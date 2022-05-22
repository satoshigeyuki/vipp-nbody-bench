#ifndef T_SNE_SOLVER_REF_HPP
#define T_SNE_SOLVER_REF_HPP

#include <barnes_hut_tree.hpp>
#include <problem.hpp>
#include <stdexcept>
#include <vantage_point_tree.hpp>

namespace t_sne {
	/// @brief BarnesHutTreeを用いた近似実装の参照実装
	template <class Output>
	struct SolverRef final {

		using image_t = typename Problem<Output>::image_t;
		using embedding_t = typename Problem<Output>::embedding_t;
		using label_t = typename Problem<Output>::label_t;

		static constexpr auto PIXEL_MAX = Problem<Output>::PIXEL_MAX;
		static constexpr auto IMAGE_SIZE = Problem<Output>::IMAGE_SIZE;
		static constexpr auto EMBEDDING_DIMENSION = Problem<Output>::EMBEDDING_DIMENSION;
		static constexpr auto P_BISECTION_ITER = Problem<Output>::P_BISECTION_ITER;
		static constexpr auto BETA_BISECTION_ITER = Problem<Output>::BETA_BISECTION_ITER;
		static constexpr auto P_MIN = Problem<Output>::P_MIN;
		static constexpr auto P_MAX = Problem<Output>::P_MAX;

		/// @exception 引力計算の際に探索する近傍画像数(k)が実際の画像数(N)と等しくなるか上回るとstd::runtime_error例外が送出される
		template <class Interpolate, class AlphaGenerator>
		void preprocess(const Problem<Output, Interpolate, AlphaGenerator> &problem, const std::vector<embedding_t> &) {
			const auto N = problem.images().size();
			_dy = std::vector<embedding_t>(N);
			_k = static_cast<std::size_t>(std::floor(problem.u() * 3));
			if (_k >= N)
				throw std::runtime_error("k must be less than the number of images");
			_p_sne_values = std::vector<Output>(N * _k);
			_p_tsne = std::vector<std::vector<std::tuple<std::size_t, Output>>>(N);
		}

	  private:
		static constexpr auto distance_x(const image_t &x1, const image_t &x2) noexcept {
			auto distance_x2 = Output{0};
			for (auto d = decltype(IMAGE_SIZE){0}; d < IMAGE_SIZE; ++d) {
				const auto difference = static_cast<Output>(x1[d]) - static_cast<Output>(x2[d]);
				distance_x2 += difference * difference;
			}
			return std::sqrt(distance_x2) / PIXEL_MAX;
		}

	  public:
		/// @brief 計算の本体
		/// @param max_iter 最大実行ステップ数
		/// @param max_pseudoF 最大pseudoF値
		/// @return 実行したステップ数を返す
		/// @details ステップ数がmax_iterに達するかpseudoF値がmax_pseudoFに達するまで最急降下法による反復を繰り返す.
		template <class Interpolate, class AlphaGenerator, class PseudoFCalculator>
		auto solve(const Problem<Output, Interpolate, AlphaGenerator> &problem, std::vector<embedding_t> &y, const std::size_t max_iter, const std::optional<Output> max_pseudoF, const PseudoFCalculator &pseudoF) {
			const auto &x = problem.images();
			const auto N = x.size();
			// y^{t} - y^{t-1}
			auto &dy = _dy;
			const auto u = problem.u();
			const auto eta = problem.eta();
			const auto theta = problem.theta();
			auto tree = VantagePointTree<image_t, distance_x>{x};
			tree.search(_k);
			// p_sneを求める
			p_sne(N, tree, _p_sne_values, u);
			// p_sneをp_tsneに変換する.
			p_tsne(N, tree, _p_sne_values, _p_tsne);
			// 反復計算
			for (auto t = decltype(max_iter){0}; t < max_iter; ++t) {
				if (max_pseudoF && pseudoF(y, problem.labels()) >= *max_pseudoF) {
					return t;
				}
				const auto alpha = problem.alpha(t);
				descent(y, dy, _p_tsne, eta, alpha, theta);
			}
			return max_iter;
		}

		template <class Interpolate, class AlphaGenerator>
		void postprocess(const Problem<Output, Interpolate, AlphaGenerator> &, std::vector<embedding_t> &) const noexcept {}

#ifdef ENABLE_TEST
		friend struct SolverRefTest;
		friend struct VantagePointTreeTest;
#endif

	  private:
		static void descent(std::vector<embedding_t> &y, std::vector<embedding_t> &dy, const std::vector<std::vector<std::tuple<std::size_t, Output>>> &p_tsne, const Output eta, const Output alpha, const Output theta) noexcept {
			const auto N = y.size();
			const auto tree = BarnesHutTree<Output>{y};
			const auto z_ = z(y, tree, theta);
#pragma omp parallel for
			for (auto i = decltype(N){0}; i < N; ++i) {
				dy[i] = subtract(multiply(alpha, dy[i]), multiply(eta, gradient(y, p_tsne, z_, i, tree, theta)));
			}
#pragma omp parallel for
			for (auto i = decltype(N){0}; i < N; ++i) {
				y[i] = add(y[i], dy[i]);
			}
		}

		static auto gradient(const std::vector<embedding_t> &y, const std::vector<std::vector<std::tuple<std::size_t, Output>>> &p_tsne, const Output z, const std::size_t i, const BarnesHutTree<Output> &tree, const Output theta) noexcept {
			return multiply(4, subtract(attractive(y, p_tsne[i], z, i), repulsive(y, z, i, tree, theta)));
		}

		static auto attractive(const std::vector<embedding_t> &y, const std::vector<std::tuple<std::size_t, Output>> &p_tsne_i, const Output z, const std::size_t i) noexcept {
			auto attractive = embedding_t{};
			for (const auto &[j, p_ij] : p_tsne_i) {
				const auto q_ij = q_tsne(y, i, j, z);
				const auto difference = subtract(y[i], y[j]);
				attractive = add(attractive, multiply(p_ij * q_ij * z, difference));
			}
			return attractive;
		}

		static auto repulsive(const std::vector<embedding_t> &y, const Output z, const std::size_t i, const BarnesHutTree<Output> &tree, const Output theta) {
			return tree.apply(fix([&y_i = y[i], z, theta, i](const auto &repulsive_i, const typename BarnesHutTree<Output>::Node &node) -> embedding_t {
				const auto y_cell = node.y_cell();
				// || y_i - y_cell ||^2
				const auto distance2 = distance_y2(y_i, y_cell);
				if (node.r_cell() < theta * std::sqrt(distance2)) {
					const auto q_i_cell = 1 / (z * (distance2 + 1));
					const auto difference = subtract(y_i, y_cell);
					return multiply(q_i_cell * q_i_cell * z * node.n_cell(), difference);
				} else {
					const auto &content = node.content();
					switch (content.index()) {
					case BarnesHutTree<Output>::Node::INDEX_EMPTY:
						return embedding_t{};
					case BarnesHutTree<Output>::Node::INDEX_POINT: {
						const auto index = std::get<BarnesHutTree<Output>::Node::INDEX_POINT>(content);
						if (index == i) {
							return embedding_t{};
						} else {
							const auto q_i_cell = 1 / (z * (distance2 + 1));
							const auto difference = subtract(y_i, y_cell);
							return multiply(q_i_cell * q_i_cell * z * node.n_cell(), difference);
						}
					}
					default: // case BarnesHutTree<Output>::Node::INDEX_CHILDREN
					{
						const auto &children = std::get<BarnesHutTree<Output>::Node::INDEX_CHILDREN>(content);
						auto repulsive = embedding_t{};
						for (const auto &child_node : children) {
							repulsive = add(repulsive, repulsive_i(repulsive_i, child_node));
						}
						return repulsive;
					}
					}
				}
			}));
		}

		static auto z(const std::vector<embedding_t> &y, const BarnesHutTree<Output> &tree, const Output theta) noexcept {
			const auto N = y.size();
			auto z = Output{0};
#pragma omp parallel for reduction(+ \
                                   : z)
			for (auto k = decltype(N){0}; k < N; ++k) {
				z += tree.apply(z_k(y, k, theta));
			}
			return z;
		}

		static decltype(auto) z_k(const std::vector<embedding_t> &y, const std::size_t k, const Output theta) noexcept {
			return fix([&y_k = y[k], theta, k](const auto &z_k, const typename BarnesHutTree<Output>::Node &node) -> Output {
				const auto y_cell = node.y_cell();
				// || y_k - y_cell ||^2
				const auto distance2 = distance_y2(y_k, y_cell);
				if (node.r_cell() < theta * std::sqrt(distance2)) {
					return Output{1} / (distance2 + 1) * node.n_cell();
				} else {
					const auto &content = node.content();
					switch (content.index()) {
					case BarnesHutTree<Output>::Node::INDEX_EMPTY:
						return Output{0};
					case BarnesHutTree<Output>::Node::INDEX_POINT: {
						const auto index = std::get<BarnesHutTree<Output>::Node::INDEX_POINT>(content);
						if (index == k) {
							return Output{0};
						} else {
							return Output{1} / (distance2 + 1);
						}
					}
					default: // case BarnesHutTree<Output>::Node::INDEX_CHILDREN
					{
						const auto &children = std::get<BarnesHutTree<Output>::Node::INDEX_CHILDREN>(content);
						auto z = Output{};
						for (const auto &child_node : children) {
							z += z_k(z_k, child_node);
						}
						return z;
					}
					}
				}
			});
		}

		// p_sneからp_tsneを求める.
		static void p_tsne(const std::size_t N, const VantagePointTree<image_t, distance_x> &tree, const std::vector<Output> &p_sne_values, std::vector<std::vector<std::tuple<std::size_t, Output>>> &p_tsne) {
			const auto k = tree.k();
			// 点i(_xでのindex)
			for (auto i = decltype(N){0}; i < N; ++i) {
				// iの近傍点(nearest_indicesでのiの近傍点の中でのindex)
				for (auto j = decltype(N){0}; j < k; ++j) {
					// iの近傍点(_xでのindex)
					const auto nearest_index = tree.nearest_index(i, j);
					if (const auto index_of_i_opt = tree.find_nearest_index(nearest_index, i)) {
						if (i < nearest_index) {
							const auto index_of_i = *index_of_i_opt;
							p_tsne[i].emplace_back(nearest_index, (p_sne_values[i * k + j] + p_sne_values[nearest_index * k + index_of_i]) / (N * 2));
							p_tsne[nearest_index].emplace_back(i, (p_sne_values[i * k + j] + p_sne_values[nearest_index * k + index_of_i]) / (N * 2));
						}
					} else {
						p_tsne[i].emplace_back(nearest_index, p_sne_values[i * k + j] / (N * 2));
						p_tsne[nearest_index].emplace_back(i, p_sne_values[i * k + j] / (N * 2));
					}
				}
			}
		}

		static auto q_tsne(const std::vector<embedding_t> &y, const std::size_t i, const std::size_t j, const Output z) noexcept {
			if (i == j)
				return Output{0};
			return Output{1} / (Output{1} + distance_y2(y[i], y[j])) / z;
		}

		static constexpr auto distance_y2(const embedding_t &y1, const embedding_t &y2) noexcept {
			const auto difference = subtract(y1, y2);
			return std::inner_product(difference.begin(), difference.end(), difference.begin(), Output{0});
		}

		// 各点iに対応するbeta_iとiのk個近傍の点jに対するp_sne{j|i}を求める.
		static void p_sne(const std::size_t N, const VantagePointTree<image_t, distance_x> &tree, std::vector<Output> &p_sne_values, const Output u) {
			const auto p = bisection_p(N, u, P_MIN, P_MAX);
			const auto N_ = static_cast<Output>(N);
			const auto beta_min = std::max(N_ * std::log2(N_ / u) / ((N_ - 1) * IMAGE_SIZE), std::sqrt(std::log2(N_ / u)) / IMAGE_SIZE);
			const auto beta_max = static_cast<Output>(PIXEL_MAX) * PIXEL_MAX * std::log2(p / (Output{1} - p) * (N - 1));
#pragma omp parallel for
			for (auto i = decltype(N){0}; i < N; ++i) {
				bisection_beta(tree, p_sne_values, i, u, beta_min, beta_max);
			}
		}

		// 点iとbeta_candidateに対応するlog_uとiのk個近傍の点jに対するp_sne{j|i}の分子,分母を求める.
		static auto entropy(const VantagePointTree<image_t, distance_x> &tree, std::vector<Output> &p_sne_values, const std::size_t i, const Output beta_candidate) noexcept {
			const auto k = tree.k();
			auto p_sne_divisor = Output{0};
			auto entropy = Output{0};
			for (auto j = decltype(k){0}; j < k; ++j) {
				// d(i,j)
				const auto distance = tree.nearest_distance(i, j);
				// exp(-beta*d^2)
				p_sne_values[i * k + j] = std::exp(-beta_candidate * distance * distance);
				p_sne_divisor += p_sne_values[i * k + j];
			}
			for (auto j = decltype(k){0}; j < k; ++j) {
				p_sne_values[i * k + j] /= p_sne_divisor;
				entropy -= p_sne_values[i * k + j] * std::log2(p_sne_values[i * k + j]);
			}
			return entropy;
		}

		// 各点iのk個近傍の点jに対するp_sne{j|i}をbetaの二分法で求める.
		static void bisection_beta(const VantagePointTree<image_t, distance_x> &tree, std::vector<Output> &p_sne_values, const std::size_t i, const Output u, Output beta_min, Output beta_max) {
			auto beta_candidate = (beta_min + beta_max) / 2;
			const auto log_u = std::log2(u);
			for (auto j = decltype(BETA_BISECTION_ITER){0}; j < BETA_BISECTION_ITER; ++j) {
				const auto entropy_ = entropy(tree, p_sne_values, i, beta_candidate);
				if (entropy_ > log_u) {
					beta_min = beta_candidate;
				} else {
					beta_max = beta_candidate;
				}
				beta_candidate = (beta_min + beta_max) / 2;
			}
		}

		// p_1の値を二分法で求める.
		static auto bisection_p(const std::size_t N, const Output u, Output p_min, Output p_max) {
			const auto N_ = static_cast<Output>(N);
			auto p_candidate = (p_min + p_max) / 2;
			const auto rhs = std::log2(std::min(std::sqrt(N_ * 2), u));
			for (auto i = decltype(P_BISECTION_ITER){0}; i < P_BISECTION_ITER; ++i) {
				const auto lhs = (Output{1} - p_candidate) * 2 * std::log2(N_ / ((Output{1} - p_candidate) * 2));
				if (lhs > rhs) {
					p_min = p_candidate;
				} else {
					p_max = p_candidate;
				}
				p_candidate = (p_min + p_max) / 2;
			}
			return p_candidate;
		}

		static constexpr auto multiply(const Output coefficient, const embedding_t &y) noexcept {
			auto result = embedding_t{};
			for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
				result[d] = coefficient * y[d];
			}
			return result;
		}

		static constexpr auto add(const embedding_t &y1, const embedding_t &y2) noexcept {
			auto result = embedding_t{};
			for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
				result[d] = y1[d] + y2[d];
			}
			return result;
		}

		static constexpr auto subtract(const embedding_t &y1, const embedding_t &y2) noexcept {
			auto result = embedding_t{};
			for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
				result[d] = y1[d] - y2[d];
			}
			return result;
		}

		template <class F>
		static decltype(auto) fix(F &&f) noexcept {
			return [f = std::forward<F>(f)](auto &&... args) {
				return f(f, std::forward<decltype(args)>(args)...);
			};
		}

		std::vector<embedding_t> _dy;
		std::size_t _k;
		std::vector<Output> _p_sne_values;
		std::vector<std::vector<std::tuple<std::size_t, Output>>> _p_tsne;
	};
} // namespace t_sne

#endif // T_SNE_SOLVER_REF_HPP
