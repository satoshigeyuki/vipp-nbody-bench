#ifndef GRAVITY_SOLVER_REF_NAIVE_HPP
#define GRAVITY_SOLVER_REF_NAIVE_HPP

#include <problem.hpp>

namespace gravity {
	/// @brief 直接法によるナイーブな参照実装
	template <class M, class T, class X, class V, class A, class G>
	struct SolverRefNaive final {
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
			const auto epsilon = problem.epsilon();
			runge_kutta(x, v, G_, m, epsilon, problem.delta_t() / t);
		}

#ifdef ENABLE_TEST
		friend struct SolverRefNaiveTest;
#endif

	  private:
		/// @brief runge kutta法によって時刻delta_t後の質点の座標を計算する
		static void runge_kutta(std::vector<Vector<X>> &x,
		                        std::vector<Vector<V>> &v,
		                        const G G_,
		                        const M m,
		                        const X epsilon,
		                        const T delta_t) {
			const auto N = x.size();
			// 位置と速度の初期値
			const auto x0 = x;
			const auto v0 = v;
			// delta_t後の位置と速度の近似値
			auto x_out = x;
			auto v_out = v;

			auto a = std::vector<Vector<A>>{N};

			// step0
			for (auto i = decltype(N){0}; i < N; ++i) {
				// 現時刻の加速度
				a[i] = acceleration(x, G_, m, epsilon, i);
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
			for (auto i = decltype(N){0}; i < N; ++i) {
				// delta_t/2後の加速度
				a[i] = acceleration(x, G_, m, epsilon, i);
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
			for (auto i = decltype(N){0}; i < N; ++i) {
				// step1の勾配予測を用いたときのdelta_t/2後の加速度
				a[i] = acceleration(x, G_, m, epsilon, i);
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
			for (auto i = decltype(N){0}; i < N; ++i) {
				// step2の勾配予測を用いたときのdelta_t後の加速度
				a[i] = acceleration(x, G_, m, epsilon, i);
				// 近似値の更新
				x_out[i] += static_cast<Vector<X>>(delta_t / 6 * v[i]);
				v_out[i] += static_cast<Vector<V>>(delta_t / 6 * a[i]);
			}
			for (auto i = decltype(N){0}; i < N; ++i) {
				x[i] = x_out[i];
				v[i] = v_out[i];
			}
		}

		/// @brief 質点iに対する加速度の計算
		static auto acceleration(const std::vector<Vector<X>> &x,
		                         const G G_,
		                         const M m,
		                         const X epsilon,
		                         const std::size_t i) {
			const auto N = x.size();
			auto acceleration = Vector<A>{};
			const auto &xi = x[i];
			for (auto j = decltype(N){0}; j < N; ++j) {
				if (i == j) {
					continue;
				}
				const auto &xj = x[j];
				acceleration += static_cast<Vector<A>>(G_ * m * (xj - xi) / std::pow((xj - xi).norm2() + epsilon * epsilon, X{3} / 2));
			}
			return acceleration;
		}

		/// @brief 一次オイラー法の計算
		/// @details RはVector<X>又はVector<V>
		/// SはVector<V>又はVector<A>
		template <class R, class S>
		static auto euler(const R init, const S slope, const T delta_t) {
			return init + slope * delta_t;
		}
	};
} // namespace gravity

#endif // GRAVITY_SOLVER_REF_NAIVE_HPP
