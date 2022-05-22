#ifndef T_SNE_BENCHMARK_HPP
#define T_SNE_BENCHMARK_HPP

#include <chrono>
#include <iostream>
#include <set>
#include <solver_ref.hpp>

namespace t_sne {
#ifdef ENABLE_PRINT
	template <class Output>
	void print(const std::vector<typename Problem<Output>::embedding_t> &y, const std::vector<typename Problem<Output>::label_t> &labels) {
		for (std::size_t i = 0; i < y.size(); ++i) {
			std::cerr << y[i][0] << ", " << y[i][1] << ", " << static_cast<std::uint16_t>(labels[i]) << std::endl;
		}
	}
#endif

	/// @brief benchmark関数でSolverを評価する.
	template <class Output, class Interpolate = float>
	struct Benchmark final {
		static constexpr auto EMBEDDING_DIMENSION = Problem<Output>::EMBEDDING_DIMENSION;

		using image_t = typename Problem<Output, Interpolate>::image_t;
		template <typename U = Output>
		using embedding_t = typename Problem<U, Interpolate>::embedding_t;
		using label_t = typename Problem<Output, Interpolate>::label_t;

		template <class EvaluatedSolver, class OutputRef = double>
		void benchmark(EvaluatedSolver &&solver_eval, const std::size_t n = 1, const std::size_t t = 1000, const double a = 2) {
			constexpr auto seed = std::uint_fast64_t{0};
			constexpr auto eta = Output{200}, u = Output{50};
			constexpr auto eta_ref = OutputRef{200}, u_ref = OutputRef{50};
#ifdef TSNE_THETA
			constexpr auto theta = Output(TSNE_THETA);
			constexpr auto theta_ref = OutputRef(TSNE_THETA);
#else
			constexpr auto theta = Output{1} / 2;
			constexpr auto theta_ref = OutputRef{1} / 2;
#endif
			const auto problem = Problem<Output, Interpolate>("data", n, t, a, seed, eta, u, theta);
			const auto N = problem.images().size();
			const auto problem_ref = Problem<OutputRef, Interpolate>("data", n, t, a, seed, eta_ref, u_ref, theta_ref);
			auto solver_ref = SolverRef<OutputRef>{};

			auto y_ref = Problem<OutputRef, Interpolate>::default_y(N);
			solver_ref.preprocess(problem_ref, y_ref);
			const auto start_ref = std::chrono::high_resolution_clock::now();
			const auto t_ref = solver_ref.solve(problem_ref, y_ref, problem_ref.t(), std::nullopt, pseudoF<OutputRef>);
			const auto end_ref = std::chrono::high_resolution_clock::now();
			solver_ref.postprocess(problem_ref, y_ref);

			const auto pseudo_F_ref = pseudoF<OutputRef>(y_ref, problem_ref.labels());

			auto y_eval = Problem<Output, Interpolate>::default_y(N);
			solver_eval.preprocess(problem, y_eval);
			const auto start_eval = std::chrono::high_resolution_clock::now();
			const auto t_eval = solver_eval.solve(problem, y_eval, static_cast<std::size_t>(problem.t() * problem.a()), pseudo_F_ref, pseudoF<Output>);
			const auto end_eval = std::chrono::high_resolution_clock::now();
			solver_eval.postprocess(problem, y_eval);
#ifdef ENABLE_PRINT
			print<Output>(y_eval, problem.labels());
#endif

			const auto pseudo_F_eval = pseudoF(y_eval, problem.labels());
			// validation
			std::cout << "pseudo F of evaluated solver  : " << pseudo_F_eval << std::endl;
			std::cout << "pseudo F of reference solver  : " << pseudo_F_ref << std::endl;
			std::cout << "calc step of evaluated solver  : " << t_eval << std::endl;
			std::cout << "calc step of reference solver  : " << t_ref << std::endl;

			std::cout << "calc time of evaluated solver : " << static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_eval - start_eval).count()) / std::nano::den << "(s)" << std::endl;
			std::cout << "calc time of reference solver : " << static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_ref - start_ref).count()) / std::nano::den << "(s)" << std::endl;
#ifdef TSNE_STDERR_CSV
			std::cerr << N << ',' << theta << ',' << pseudo_F_eval << ',' << pseudo_F_ref << ',' << t_eval << ',' << t_ref << ','
                                  << static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_eval - start_eval).count()) / std::nano::den << ','
                                  << static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_ref - start_ref).count()) / std::nano::den << std::endl;
#endif
		}

#ifdef ENABLE_TEST
		friend struct BenchmarkTest;
#endif

	  private:
		/// @brief 2次元に射影された点の座標と真のラベルからpseudoF値を計算する.
		template <typename U = Output>
		static auto pseudoF(const std::vector<embedding_t<U>> &y, const std::vector<label_t> &labels) {
			const auto t = T<U>(y);
			auto wk = U{0};
			auto label_set = std::set<label_t>{};
			for (const auto &label : labels)
				label_set.insert(label);
			if (label_set.size() <= 1) {
				throw std::runtime_error("cluster size must be greater than 1.");
			}
			for (const auto &label : label_set) {
				wk += Wk<U>(y, labels, label);
			}
			return (t - wk) * (y.size() - label_set.size()) / (wk * (label_set.size() - 1));
		}

		/// @brief 全点の重心からの距離の二乗和
		template <typename U = Output>
		static auto T(const std::vector<embedding_t<U>> &y) noexcept {
			const auto N = y.size();
			auto mean = embedding_t<U>{};
			auto t = U{0};
			for (auto i = decltype(N){0}; i < N; ++i) {
				for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
					mean[d] += y[i][d];
				}
			}
			for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
				mean[d] /= N;
			}
			for (auto i = decltype(N){0}; i < N; ++i) {
				for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
					const auto diff = mean[d] - y[i][d];
					t += diff * diff;
				}
			}
			return t;
		}

		/// @brief 各ラベルの重心からの距離の二乗和
		template <typename U = Output>
		static auto Wk(const std::vector<embedding_t<U>> &y, const std::vector<label_t> &labels, const label_t label) noexcept {
			const auto N = labels.size();
			auto n = decltype(N){0};
			auto mean = embedding_t<U>{};
			auto wk = U{0};
			for (auto i = decltype(N){0}; i < N; ++i) {
				if (labels[i] != label)
					continue;
				++n;
				for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
					mean[d] += y[i][d];
				}
			}
			for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
				mean[d] /= n;
			}
			for (auto i = decltype(N){0}; i < N; ++i) {
				if (labels[i] != label)
					continue;
				for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
					const auto diff = mean[d] - y[i][d];
					wk += diff * diff;
				}
			}
			return wk;
		}
	};
} // namespace t_sne

#endif // T_SNE_BENCHMARK_HPP
