#ifndef T_SNE_SOLVER_REF_NAIVE_HPP
#define T_SNE_SOLVER_REF_NAIVE_HPP

#include <problem.hpp>
#include <stdexcept>

namespace t_sne {
	/// @brief BarnesHutTreeを用いないナイーブな実装
	template <class Output>
	struct SolverRefNaive final {
		static constexpr auto PIXEL_MAX = Problem<Output>::PIXEL_MAX;
		static constexpr auto IMAGE_SIZE = Problem<Output>::IMAGE_SIZE;
		static constexpr auto EMBEDDING_DIMENSION = Problem<Output>::EMBEDDING_DIMENSION;
		static constexpr auto P_BISECTION_ITER = Problem<Output>::P_BISECTION_ITER;
		static constexpr auto BETA_BISECTION_ITER = Problem<Output>::BETA_BISECTION_ITER;
		static constexpr auto P_MIN = Problem<Output>::P_MIN;
		static constexpr auto P_MAX = Problem<Output>::P_MAX;

		using image_t = typename Problem<Output>::image_t;
		using embedding_t = typename Problem<Output>::embedding_t;
		using label_t = typename Problem<Output>::label_t;

		template <class Interpolate, class AlphaGenerator>
		void preprocess(const Problem<Output, Interpolate, AlphaGenerator> &problem, const std::vector<embedding_t> &) {
			const auto N = problem.images().size();
			_dy = std::vector<embedding_t>(N);
			_betas = std::vector<Output>(N);
			_p_sne_divisors = std::vector<Output>(N);
		}

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
			auto &betas = _betas;
			auto &p_sne_divisors = _p_sne_divisors;
			beta(x, betas, u);
#pragma omp parallel for
			for (auto i = decltype(N){0}; i < N; ++i) {
				p_sne_divisors[i] = p_sne_divisor(x, i, betas[i]);
			}

			for (auto t = decltype(max_iter){0}; t < max_iter; ++t) {
				if (max_pseudoF && pseudoF(y, problem.labels()) >= *max_pseudoF) {
					return t;
				}
				const auto alpha = problem.alpha(t);
				descent(x, y, dy, betas, p_sne_divisors, eta, alpha);
			}
			return max_iter;
		}

		template <class Interpolate, class AlphaGenerator>
		void postprocess(const Problem<Output, Interpolate, AlphaGenerator> &, std::vector<embedding_t> &) {}

#ifdef ENABLE_TEST
		friend struct SolverRefNaiveTest;
#endif

	  private:
		static void descent(const std::vector<image_t> &x, std::vector<embedding_t> &y, std::vector<embedding_t> &dy, const std::vector<Output> &betas, const std::vector<Output> &p_sne_divisors, const Output eta, const Output alpha) noexcept {
			const auto N = x.size();
			const auto z_ = z(y);
#pragma omp parallel for
			for (auto i = decltype(N){0}; i < N; ++i) {
				dy[i] = add(multiply(eta, gradient(x, y, betas, p_sne_divisors, z_, i)), multiply(alpha, dy[i]));
			}
#pragma omp parallel for
			for (auto i = decltype(N){0}; i < N; ++i) {
				y[i] = add(y[i], dy[i]);
			}
		}

		static auto gradient(const std::vector<image_t> &x, const std::vector<embedding_t> &y, const std::vector<Output> &betas, const std::vector<Output> &p_sne_divisors, const Output z, const std::size_t i) noexcept {
			const auto N = x.size();
			auto gradient = embedding_t();
			for (auto j = decltype(N){0}; j < N; ++j) {
				if (j == i) {
					continue;
				}
				const auto p_ij = p_tsne(x, i, j, betas[i], p_sne_divisors[i], p_sne_divisors[j]);
				const auto q_ij = q_tsne(y, i, j, z);
				const auto difference = subtract(y[i], y[j]);
				gradient = add(gradient, multiply((p_ij - q_ij) * q_ij * z, difference));
			}
			return multiply(4, gradient);
		}

		static auto z(const std::vector<embedding_t> &y) noexcept {
			const auto N = y.size();
			auto z = Output{0};
			// 高速化可能
			for (auto k = decltype(N){0}; k < N; ++k) {
				for (auto l = decltype(N){0}; l < N; ++l) {
					if (k == l)
						continue;
					z += Output{1} / (Output{1} + distance_y2(y[k], y[l]));
				}
			}
			return z;
		}

		static auto p_tsne(const std::vector<image_t> &x, const std::size_t i, const std::size_t j, const Output beta, const Output divisor_i, const Output divisor_j) noexcept {
			return (p_sne(x, i, j, beta, divisor_i) + p_sne(x, j, i, beta, divisor_j)) / (x.size() * 2);
		}

		static auto p_sne(const std::vector<image_t> &x, const std::size_t i, const std::size_t j, const Output beta, const Output divisor) noexcept {
			if (i == j)
				return Output{0};
			return std::exp(-distance_x2(x[i], x[j]) * beta) / divisor;
		}

		static auto p_sne_divisor(const std::vector<image_t> &x, const std::size_t i, const Output beta) noexcept {
			const auto N = x.size();
			auto divisor = Output{0};
			for (auto k = decltype(N){0}; k < N; ++k) {
				if (k == i)
					continue;
				divisor += std::exp(-distance_x2(x[i], x[k]) * beta);
			}
			return divisor;
		}

		static auto q_tsne(const std::vector<embedding_t> &y, const std::size_t i, const std::size_t j, const Output z) noexcept {
			if (i == j)
				return Output{0};
			return Output{1} / (Output{1} + distance_y2(y[i], y[j])) / z;
		}

		static constexpr auto distance_x2(const image_t &x1, const image_t &x2) noexcept {
			auto distance_x2 = Output{0};
			for (auto d = decltype(IMAGE_SIZE){0}; d < IMAGE_SIZE; ++d) {
				const auto difference = static_cast<Output>(x1[d]) - static_cast<Output>(x2[d]);
				distance_x2 += difference * difference;
			}
			return distance_x2 / PIXEL_MAX / PIXEL_MAX;
		}

		static constexpr auto distance_y2(const embedding_t &y1, const embedding_t &y2) noexcept {
			auto distance_y2 = Output{0};
			for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
				const auto difference = static_cast<Output>(y1[d]) - static_cast<Output>(y2[d]);
				distance_y2 += difference * difference;
			}
			return distance_y2;
		}

		static void beta(const std::vector<image_t> &x, std::vector<Output> &betas, const Output u) {
			const auto N = x.size();
			const auto N_ = static_cast<Output>(N);
			const auto p = bisection_p(N, u, P_MIN, P_MAX);
			const auto beta_min = std::max(N_ * std::log2(N_ / u) / ((N_ - 1) * IMAGE_SIZE), std::sqrt(std::log2(N_ / u)) / IMAGE_SIZE);
			const auto beta_max = Output{PIXEL_MAX} * PIXEL_MAX * std::log2(p / (Output{1} - p) * (N - 1));

#pragma omp parallel for
			for (auto i = decltype(N){0}; i < N; ++i) {
				betas[i] = bisection_beta(x, i, u, beta_min, beta_max);
			}
		}

		static auto entropy(const std::vector<image_t> &x, const std::size_t i, const Output beta_candidate) noexcept {
			const auto N = x.size();
			auto divisor = p_sne_divisor(x, i, beta_candidate);
			if (divisor < std::numeric_limits<Output>::epsilon())
				divisor = std::numeric_limits<Output>::epsilon();
			auto entropy = Output{0};
			for (auto j = decltype(N){0}; j < N; ++j) {
				if (i == j)
					continue;
				const auto p = p_sne(x, i, j, beta_candidate, divisor);
				if (p < std::numeric_limits<Output>::epsilon())
					continue;
				entropy += p * std::log2(p);
			}
			return -entropy;
		}

		static auto bisection_beta(const std::vector<image_t> &x, const std::size_t i, const Output u, Output beta_min, Output beta_max) {
			auto beta_candidate = (beta_min + beta_max) / 2;
			const auto log_u = std::log2(u);
			for (auto j = decltype(BETA_BISECTION_ITER){0}; j < BETA_BISECTION_ITER; ++j) {
				const auto entropy_ = entropy(x, i, beta_candidate);
				if (entropy_ > log_u) {
					beta_min = beta_candidate;
				} else {
					beta_max = beta_candidate;
				}
				beta_candidate = (beta_min + beta_max) / 2;
			}
			return beta_candidate;
		}

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

		std::vector<embedding_t> _dy;
		std::vector<Output> _betas;
		std::vector<Output> _p_sne_divisors;
	};
} // namespace t_sne

#endif // T_SNE_SOLVER_REF_NAIVE_HPP
