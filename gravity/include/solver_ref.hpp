#ifndef GRAVITY_SOLVER_REF_HPP
#define GRAVITY_SOLVER_REF_HPP

#include <fmm.hpp>

namespace gravity {
	/// @brief 高速多重極展開法を用いた近似実装の参照実装
	template <class M, class T, class X, class V, class A, class G>
	struct SolverRef final {
		/// @brief 前処理を行う
		void preprocess(const Problem<M, T, X, V, A, G> &, const std::vector<Vector<X>> &, const std::vector<Vector<V>> &) {}

		/// @brief 後処理を行う
		void postprocess(const Problem<M, T, X, V, A, G> &, std::vector<Vector<X>> &, std::vector<Vector<V>> &) {}

		/// @brief 計算の本体
		void solve(const Problem<M, T, X, V, A, G> &problem,
		           const T t,
		           std::vector<Vector<X>> &x,
		           std::vector<Vector<V>> &v) {
			const auto G_ = problem.get_G();
			const auto m = problem.m();
			const auto s = problem.s();
			const auto p = problem.p();
			const auto epsilon = problem.epsilon();
			runge_kutta(x, v, G_, m, epsilon, problem.delta_t() / t, s, p);
		}

#ifdef ENABLE_TEST
		friend struct SolverRefTest;
		friend struct FMMTest;
#endif

	  private:
		/// @brief runge kutta法によって時刻delta_t後の質点の座標を計算する
		static void runge_kutta(std::vector<Vector<X>> &x,
		                        std::vector<Vector<V>> &v,
		                        const G G_,
		                        const M m,
		                        const X epsilon,
		                        const T delta_t,
		                        const std::size_t s,
		                        const unsigned p) {
			const auto N = x.size();
			// 位置と速度の初期値
			const auto x0 = x;
			const auto v0 = v;
			// delta_t後の位置と速度の近似値
			auto x_out = x;
			auto v_out = v;

			// step0
			// 現時刻の加速度
			auto a = std::vector<Vector<A>>{N};
			acceleration(x, s, G_, m, epsilon, a, p);
			for (auto i = decltype(N){0}; i < N; ++i) {
				// 近似値の更新
				x_out[i] += static_cast<Vector<X>>(delta_t / 6 * v[i]);
				v_out[i] += static_cast<Vector<V>>(delta_t / 6 * a[i]);
			}
			// step1
			for (auto i = decltype(N){0}; i < N; ++i) {
				// delta_t/2後の位置と速度をeuler法で近似する.
				x[i] = static_cast<Vector<X>>(euler(x0[i], v[i], delta_t / 2));
				v[i] = static_cast<Vector<V>>(euler(v0[i], a[i], delta_t / 2));
			}
			// delta_t/2後の加速度
			acceleration(x, s, G_, m, epsilon, a, p);
			for (auto i = decltype(N){0}; i < N; ++i) {
				// 近似値の更新
				x_out[i] += static_cast<Vector<X>>(delta_t / 3 * v[i]);
				v_out[i] += static_cast<Vector<V>>(delta_t / 3 * a[i]);
			}
			// step2
			for (auto i = decltype(N){0}; i < N; ++i) {
				// delta_t/2後の位置と速度をeuler法で近似する.ただし勾配にはstep1の予測を用いる.
				x[i] = static_cast<Vector<X>>(euler(x0[i], v[i], delta_t / 2));
				v[i] = static_cast<Vector<V>>(euler(v0[i], a[i], delta_t / 2));
			}
			// step1の勾配予測を用いたときのdelta_t/2後の加速度
			acceleration(x, s, G_, m, epsilon, a, p);
			for (auto i = decltype(N){0}; i < N; ++i) {
				// 近似値の更新
				x_out[i] += static_cast<Vector<X>>(delta_t / 3 * v[i]);
				v_out[i] += static_cast<Vector<V>>(delta_t / 3 * a[i]);
			}
			// step3
			for (auto i = decltype(N){0}; i < N; ++i) {
				// delta_t後の位置と速度をeuler法で近似する.ただし勾配にはstep2の予測を用いる.
				x[i] = static_cast<Vector<X>>(euler(x0[i], v[i], delta_t));
				v[i] = static_cast<Vector<V>>(euler(v0[i], a[i], delta_t));
			}
			// step2の勾配予測を用いたときのdelta_t後の加速度
			acceleration(x, s, G_, m, epsilon, a, p);
			for (auto i = decltype(N){0}; i < N; ++i) {
				// 近似値の更新
				x_out[i] += static_cast<Vector<X>>(delta_t / 6 * v[i]);
				v_out[i] += static_cast<Vector<V>>(delta_t / 6 * a[i]);
			}
			for (auto i = decltype(N){0}; i < N; ++i) {
				x[i] = x_out[i];
				v[i] = v_out[i];
			}
		}

		/// @brief 一次オイラー法の計算
		/// @details RはVector<X>又はVector<V>
		// SはVector<V>又はVector<A>
		template <class R, class S>
		static auto euler(const R init, const S slope, const T delta_t) {
			return init + slope * delta_t;
		}

		/// @brief M2LとP2Pの適用対象を計算する.
		static void calc_link(const Cell<M, T, X, V, A, G> &cell, const std::vector<Cell<M, T, X, V, A, G>> &cells, std::vector<std::vector<std::size_t>> &approx_interact_indices_list, std::vector<std::vector<std::size_t>> &neighbor_interact_indices_list, std::vector<std::vector<std::size_t>> &neighbor_indices_list, const std::size_t level) {
			const auto index = cell.index();
			auto &neighbor_interact_indices = neighbor_interact_indices_list[index];
			auto &neighbor_indices = neighbor_indices_list[index];
			auto &approx_interact_indices = approx_interact_indices_list[index];
			if (level != 0) {
				const auto parent_index = cell.parent().value();

				const auto &parent_neighbor_interact_indices = neighbor_interact_indices_list[parent_index];
				const auto &parent_neighbor_indices = neighbor_indices_list[parent_index];

				std::copy(parent_neighbor_interact_indices.begin(), parent_neighbor_interact_indices.end(), std::back_inserter(neighbor_interact_indices));

				// 親を含む
				for (const auto parent_neighbor_index : parent_neighbor_indices) {
					// const auto cousin_index_base = cells[parent_neighbor_index].cells();
					// 兄弟と自分自身を含む
					// for (auto cousin_index = cousin_index_base; cousin_index < cousin_index_base + 8; ++cousin_index) {
					for (const auto cousin_index : cells[parent_neighbor_index].cells()) {
						const auto &cousin = cells[cousin_index];
						// 自分自身か隣接ノードの場合
						if (cell.is_adjacent_or_equal_to(cells, cousin)) {
							if (!cell.is_leaf() && !cousin.is_leaf()) {
								neighbor_indices.emplace_back(cousin_index);
							} else {
								neighbor_interact_indices.emplace_back(cousin_index);
							}
						} else {
							// 隣接していないノードはM2Lの対象となる
							approx_interact_indices.emplace_back(cousin_index);
						}
					}
				}
			} else {
				if (cell.is_leaf()) {
					neighbor_interact_indices.emplace_back(cell.index());
				} else {
					neighbor_indices.emplace_back(cell.index());
				}
			}
			if (!cell.is_leaf()) {
				const auto child_index_base = std::get<Cell<M, T, X, V, A, G>::INDEX_CELLS>(cell.content());
#pragma omp parallel for
				for (auto child_index = child_index_base; child_index < child_index_base + 8; ++child_index) {
					calc_link(cells[child_index], cells, approx_interact_indices_list, neighbor_interact_indices_list, neighbor_indices_list, level + 1);
				}
			}
		}

		static void acceleration(const std::vector<Vector<X>> &x, const std::size_t s, const G G_, const M m, const X epsilon, std::vector<Vector<A>> &a, const unsigned p) {
			std::fill(a.begin(), a.end(), Vector<A>{});
			auto tree = Octree<Cell<M, T, X, V, A, G>, Vector<X>>{x, s};
			auto &cells = tree.cells();
			// M2Lの対象index
			std::vector<std::vector<std::size_t>> approx_interact_indices_list{cells.size()};
			// P2Pの対象index
			std::vector<std::vector<std::size_t>> neighbor_interact_indices_list{cells.size()};
			// 同一ノードの隣接セルを保持
			std::vector<std::vector<std::size_t>> neighbor_indices_list{cells.size()};
			// approx_interact_indicesとneighbor_interact_indicesを計算する.
			tree.apply(calc_link, cells, approx_interact_indices_list, neighbor_interact_indices_list, neighbor_indices_list, std::size_t{0});
			fmm(x, G_, m, epsilon, a, p, tree, approx_interact_indices_list, neighbor_interact_indices_list);
		}

		/// @brief bottom up sweep (P2M, M2M), unordered sweep (M2L, P2P), top down sweep (L2L, L2P)の3パスでFMMを実行する.
		static void fmm(const std::vector<Vector<X>> &x, const G G_, const M m, const X epsilon, std::vector<Vector<A>> &a, const unsigned p, Octree<Cell<M, T, X, V, A, G>, Vector<X>> &tree, const std::vector<std::vector<std::size_t>> &approx_interact_indices_list, const std::vector<std::vector<std::size_t>> &neighbor_interact_indices_list) {
			constexpr auto OctreeLevel2Threshold = 8;
			auto &cells = tree.cells();
			// P2M & M2M
			tree.apply(Octree<Cell<M, T, X, V, A, G>, Vector<X>>::fix([&cells, &p, &m, &x](const auto &bottom_up_f, Cell<M, T, X, V, A, G> &cell, const std::size_t level) -> void {
				           const auto &content = cell.content();
				           if (content.index() == Cell<M, T, X, V, A, G>::INDEX_POINTS) {
					           if (level >= 2) {
						           p2m(cell, p, x, m);
					           }
				           } else {
					           const auto child_index_base = std::get<Cell<M, T, X, V, A, G>::INDEX_CELLS>(content);
#pragma omp parallel for
					           for (auto child_index = child_index_base; child_index < child_index_base + 8; ++child_index) {
						           bottom_up_f(bottom_up_f, cells[child_index], level + 1);
					           }
					           if (level >= 2) {
						           m2m(cell, p, cells);
					           }
				           }
			           }),
			           std::size_t{0});
			// M2L & P2P
#pragma omp parallel for
			for (auto index = decltype(cells.size()){0}; index < cells.size(); ++index) {
				auto &cell = cells[index];
				if (cell.index() > OctreeLevel2Threshold) {
					m2l(cell, p, cells, approx_interact_indices_list[index]);
				}
				if (cell.is_leaf()) {
					p2p(cell, x, a, cells, neighbor_interact_indices_list[index], G_, m, epsilon);
				}
			}
			// L2L & L2P
			tree.apply(Octree<Cell<M, T, X, V, A, G>, Vector<X>>::fix([&cells, &p, &x, &a](const auto &top_down_f, Cell<M, T, X, V, A, G> &cell, const std::size_t level) -> void {
				           const auto &content = cell.content();
				           if (level >= 3) {
					           l2l(cell, p, cells);
				           }
				           if (content.index() == Cell<M, T, X, V, A, G>::INDEX_POINTS) {
					           if (level >= 2) {
						           l2p(cell, p, x, a);
					           }
				           } else {
					           const auto child_index_base = std::get<Cell<M, T, X, V, A, G>::INDEX_CELLS>(content);
#pragma omp parallel for
					           for (auto child_index = child_index_base; child_index < child_index_base + 8; ++child_index) {
						           top_down_f(top_down_f, cells[child_index], level + 1);
					           }
				           }
			           }),
			           std::size_t{0});
		}
	};
} // namespace gravity

#endif // GRAVITY_SOLVER_REF_HPP
