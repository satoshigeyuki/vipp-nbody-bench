#ifndef GRAVITY_BENCHMARK_HPP
#define GRAVITY_BENCHMARK_HPP

#include <iostream>
#include <solver_ref.hpp>
#ifdef ENABLE_PRINT
#include <fstream>
#include <iomanip>
#include <sstream>
#endif

namespace gravity {
#ifdef ENABLE_PRINT
	template <class X>
	void print(const std::string &filename, const std::vector<Vector<X>> &x) {
		std::ofstream ofs(filename);
		for (const auto &point : x) {
			ofs << point.x() << "," << point.y() << "," << point.z() << std::endl;
		}
	}
#endif

	/// @brief benchmark関数でSolverを評価する.
	template <class M, class T, class X, class V, class A, class G>
	struct Benchmark final {
		/// @brief 誤差Eの係数の閾値
		static constexpr auto ALPHA_THRESHOLD = static_cast<X>(1E-6);
		/// @brief 誤差Eの指数の係数の閾値
		static constexpr auto C_THRESHOLD = static_cast<X>(1E-2);

#ifdef ENABLE_DIFFERENT_PRECISION
		template <class EvaluatedSolver, template <class, class, class, class, class, class> class ReferenceSolver, class MR, class TR, class XR, class VR, class AR, class GR>
		void benchmark_different_precision(EvaluatedSolver &&solver_eval, ReferenceSolver<MR, TR, XR, VR, AR, GR> &solver_ref, const std::size_t N, const std::size_t s) {
			const auto problem_ref = Problem<MR, TR, XR, VR, AR, GR>{N, s};
			const auto problem_eval = Problem<M, T, X, V, A, G>{N, s};
			const auto t_set = std::vector<T>{1, 2, 4, 8, 16};
			auto x_eval = std::vector<Vector<X>>{N};
			auto v_eval = std::vector<Vector<V>>{N};
			auto x_ref = std::vector<Vector<XR>>{N};
			auto v_ref = std::vector<Vector<VR>>{N};
			auto e_t = std::vector<X>{};
			auto eval_duration = double{0};

			for (const auto &t : t_set) {
				problem_eval.initialize(x_eval, v_eval);
#ifdef ENABLE_PRINT
				print("before.csv", x_eval);
#endif
				solver_eval.preprocess(problem_eval, x_eval, v_eval);
				const auto start_eval = std::chrono::high_resolution_clock::now();
				solver_eval.solve(problem_eval, t, x_eval, v_eval);
				const auto end_eval = std::chrono::high_resolution_clock::now();
				solver_eval.postprocess(problem_eval, x_eval, v_eval);
				eval_duration += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_eval - start_eval).count()) / 1E9;
#ifdef ENABLE_PRINT
				std::stringstream ss;
				ss << "after" << t << ".csv";
				print(ss.str(), x_eval);
#endif

				problem_ref.initialize(x_ref, v_ref);
				solver_ref.preprocess(problem_ref, x_ref, v_ref);
				solver_ref.solve(problem_ref, static_cast<TR>(t), x_ref, v_ref);
				solver_ref.postprocess(problem_ref, x_ref, v_ref);

				auto e_t2 = std::common_type_t<X, XR>{};
				for (auto i = decltype(N){0}; i < N; ++i) {
					e_t2 += (x_eval[i] - x_ref[i]).norm2();
				}
				e_t.emplace_back(std::sqrt(e_t2));
			}

			// α,cの計算
			const auto [alpha, c] = curve_fitting(e_t, t_set);
			std::cout << "curve fitting to E(T) = αT^c α:" << alpha << " c:" << c << std::endl;
			const auto alpha_threshold = ALPHA_THRESHOLD * static_cast<X>(std::sqrt(problem_eval.N()));
			const auto c_threshold = -5 + C_THRESHOLD;
			if (alpha < alpha_threshold || c < c_threshold) {
				std::cout << "solver has passed the benchmark test" << std::endl;
			} else {
				std::cout << "solver failed the benchmark test" << std::endl;
			}

			// 実行時間
			std::cout << "calc time of evaluated solver : " << eval_duration << "(s)" << std::endl;
		}
#endif

		/// @brief solver_evalを評価する
		template <class EvaluatedSolver>
		void benchmark(EvaluatedSolver &&solver_eval, const std::size_t N, const std::size_t s, const std::optional<X> epsilon = std::nullopt) {
			constexpr auto g = G{1};
			constexpr auto m = M{1};
			constexpr auto r = X{1};
#ifdef GRAVITY_P
			constexpr auto p = unsigned(GRAVITY_P);
#else
			constexpr auto p = unsigned{4};
#endif
			const auto problem = Problem<M, T, X, V, A, G>{N, s, epsilon, g, m, r, p};

			using ref_t = double;
			constexpr auto g_ref = ref_t{g}, m_ref = ref_t{m}, r_ref = ref_t{r};
			const auto problem_ref = Problem<ref_t, ref_t, ref_t, ref_t, ref_t, ref_t>{N, s, epsilon, g_ref, m_ref, r_ref, p};
#ifdef GRAVITY_SEARCH_DELTA_T
			const auto t_set = geometric_sequence(T{1}, T{1.5}, 30);
#else
			const auto t_set = geometric_sequence(T{1}, T{2}, 5);
#endif
			auto solver_ref = SolverRef<ref_t, ref_t, ref_t, ref_t, ref_t, ref_t>{};
			auto x_eval = std::vector<Vector<X>>{N};
			auto v_eval = std::vector<Vector<V>>{N};
			auto x_ref = std::vector<Vector<ref_t>>{N};
			auto v_ref = std::vector<Vector<ref_t>>{N};
			auto e_t = std::vector<X>{};
			auto eval_duration = double{0};

			for (const auto &t : t_set) {
				problem.initialize(x_eval, v_eval);
#ifdef ENABLE_PRINT
				print("before.csv", x_eval);
#endif
				solver_eval.preprocess(problem, x_eval, v_eval);
				const auto start_eval = std::chrono::high_resolution_clock::now();
				solver_eval.solve(problem, t, x_eval, v_eval);
				const auto end_eval = std::chrono::high_resolution_clock::now();
				solver_eval.postprocess(problem, x_eval, v_eval);
				eval_duration += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_eval - start_eval).count()) / 1E9;
#ifdef ENABLE_PRINT
				std::stringstream ss;
				ss << "after" << t << ".csv";
				print(ss.str(), x_eval);
#endif

#ifdef GRAVITY_SUBDIVIDE_REF_TIMESTEP
				constexpr auto subdivide_ref_timestep = std::size_t(GRAVITY_SUBDIVIDE_REF_TIMESTEP);
#else
				constexpr auto subdivide_ref_timestep = std::size_t{1};
#endif

				problem_ref.initialize(x_ref, v_ref);
				solver_ref.preprocess(problem_ref, x_ref, v_ref);
				for (auto k = std::size_t{0}; k < subdivide_ref_timestep; ++k) {
					solver_ref.solve(problem_ref, subdivide_ref_timestep * t, x_ref, v_ref);
				}
				solver_ref.postprocess(problem_ref, x_ref, v_ref);

				auto e_t2 = X{};
				for (auto i = decltype(N){0}; i < N; ++i) {
					e_t2 += (x_eval[i] - Vector<X>(x_ref[i])).norm2();
				}
				e_t.emplace_back(std::sqrt(e_t2));

#ifdef GRAVITY_SEARCH_DELTA_T
				std::cerr << T{1} / t << "," << std::sqrt(e_t2) << std::endl;
#endif
			}

			// α,cの計算
			const auto [alpha, c] = curve_fitting(e_t, t_set);
			std::cout << "curve fitting to E(T) = αT^c α:" << alpha << " c:" << c << std::endl;
			const auto alpha_threshold = ALPHA_THRESHOLD * static_cast<X>(std::sqrt(problem.N()));
			const auto c_threshold = -5 + C_THRESHOLD;
			if (alpha < alpha_threshold || c < c_threshold) {
				std::cout << "solver has passed the benchmark test" << std::endl;
			} else {
				std::cout << "solver failed the benchmark test" << std::endl;
			}

			// 実行時間
			std::cout << "calc time of evaluated solver : " << eval_duration << "(s)" << std::endl;
#ifdef GRAVITY_CHECK_TIME
			std::cerr << N << "," << s << "," << alpha << "," << c << "," << eval_duration << std::endl;
#endif
		}

#ifdef ENABLE_TEST
		friend struct BenchmarkTest;
#endif

	  private:
		using fitting_t = decltype(std::declval<X>() * std::declval<T>());
		static constexpr inline auto C_MIN = fitting_t{-10};
		static constexpr inline auto C_MAX = fitting_t{10};
		static constexpr inline auto C_EPSILON = static_cast<fitting_t>(1E-10);

		/// @brief 等比数列 {a, a*r, a*r^2, ..., a*r^(n-1)} を生成する
		template <typename U>
		static auto geometric_sequence(const U a, const U r, const std::size_t n) {
			auto ret = std::vector<U>(n, a);
			for (auto i = std::size_t{1}; i < n; ++i) {
				ret[i] = ret[i - 1] * r;
			}
			return ret;
		}

		/// @brief T = {1,2,4,8,16}に対応するE(T)の集合を受け取ってE(T) = αT^cに曲線回帰した場合のαとcの組を返す.
		static auto curve_fitting(const std::vector<X> &e_t, const std::vector<T> &t_set) {
			auto c_min = C_MIN;
			auto c_max = C_MAX;
			auto c = (c_min + c_max) / 2;
			const auto max_iter = static_cast<std::size_t>(std::ceil(std::log2((C_MAX - C_MIN) / C_EPSILON)));
			for (auto i = decltype(max_iter){0}; i < max_iter; ++i) {
				// \sum e_t x_i^c
				auto e_t_x_i = fitting_t{0};
				// \sum x_i^2c
				auto x_i2 = fitting_t{0};
				// \sum e_t x_i^c log(x_i)
				auto e_t_x_i_log_x_i = fitting_t{0};
				// \sum x_i^2c log(x_i)
				auto x_i2_log_x_i = fitting_t{0};
				for (auto t_index = std::size_t{0}; t_index < t_set.size(); ++t_index) {
					// x_i^c
					const auto x_i = static_cast<fitting_t>(std::pow(t_set[t_index], c));
					const auto ln_t = static_cast<fitting_t>(std::log(t_set[t_index]));
					e_t_x_i += e_t[t_index] * x_i;
					x_i2 += x_i * x_i;
					e_t_x_i_log_x_i += e_t[t_index] * x_i * ln_t;
					x_i2_log_x_i += x_i * x_i * ln_t;
				}
				if (e_t_x_i * x_i2_log_x_i - x_i2 * e_t_x_i_log_x_i >= 0) {
					c_max = c;
				} else {
					c_min = c;
				}
				c = (c_min + c_max) / 2;
			}
			return std::make_tuple(calc_alpha(e_t, t_set, c), c);
		}

		static auto calc_alpha(const std::vector<X> &e_t, const std::vector<T> &t_set, const fitting_t &c) {
			auto numerator = fitting_t{0};
			auto denominator = fitting_t{0};
			for (auto t_index = std::size_t{0}; t_index < t_set.size(); ++t_index) {
				// x_i^c
				const auto x_i = static_cast<fitting_t>(std::pow(t_set[t_index], c));
				numerator += e_t[t_index] * x_i;
				denominator += x_i * x_i;
			}
			return numerator / denominator;
		}
	};
} // namespace gravity

#endif // GRAVITY_BENCHMARK_HPP
