#ifndef H_MATRIX_BENCHMARK_HPP
#define H_MATRIX_BENCHMARK_HPP

#include <chrono>
#include <iostream>
#include <problem.hpp>
#include <solver_ref.hpp>

namespace h_matrix {
#ifdef ENABLE_PRINT_MATRIX
	template <class T>
	void print(const Matrix<T> &h) {
		for (auto m = decltype(h.msize()){0}; m < h.msize(); ++m) {
			for (auto n = decltype(h.nsize()){0}; n < h.nsize(); ++n) {
				std::cerr << m << "," << n << "," << h(m, n) << std::endl;
			}
			std::cerr << std::endl;
		}
	}
#endif

	/// @brief benchmark関数でSolverを評価する.
	template <class T, class T_ref = double>
	struct Benchmark {
		static constexpr auto EPSILON = static_cast<T>(1E-5);
		/// @brief solver_evalを評価する.
		template <class EvaluatedSolver>
		void benchmark(EvaluatedSolver &&solver_eval, const std::size_t N) {
			const auto problem = Problem<T>{N};

			auto h = Matrix<T>{problem.matrix_size(), problem.matrix_size()};
			auto x = Vector<T>{problem.matrix_size()};
			problem.initialize_H(h);
#ifdef ENABLE_PRINT_MATRIX
			print(h);
#endif
			problem.initialize_x(x);
			const auto y = matrix_vector_multiply(h, x);

			solver_eval.preprocess(problem, h, x);
			// H'
			auto h_eval = ApproximateMatrix<T>{problem.matrix_size(), problem.matrix_size()};
#ifdef H_MATRIX_SAME_RANK_AS
			// define reference solver
			const auto problem_ref = Problem<T_ref>{N};
			auto h_ref = Matrix<T_ref>{problem_ref.matrix_size(), problem_ref.matrix_size()};
			auto x_ref = Vector<T_ref>{problem_ref.matrix_size()};
			problem_ref.initialize_H(h_ref);
			problem_ref.initialize_x(x_ref);

			// calculate rank of reference matrix
			auto solver_ref = SolverRef<T_ref>(solver_eval.tolerance);
			auto h_eval_ref = ApproximateMatrix<double>{problem_ref.matrix_size(), problem_ref.matrix_size()};
			const auto start_approximate_ref = std::chrono::high_resolution_clock::now();
			solver_ref.low_rank_approximate(problem_ref, h_ref, h_eval_ref);
			const auto end_approximate_ref = std::chrono::high_resolution_clock::now();
			const auto t_approximate_ref = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_approximate_ref - start_approximate_ref).count()) / std::nano::den;
			std::cout << "approximate time of reference solver : " << t_approximate_ref << "(s)" << std::endl;

			// calculate approximate matrix same rank as reference
			const auto start_approximate = std::chrono::high_resolution_clock::now();
			solver_eval.low_rank_approximate(problem, h, h_eval, h_eval_ref);
			const auto end_approximate = std::chrono::high_resolution_clock::now();
#else
			const auto start_approximate = std::chrono::high_resolution_clock::now();
			solver_eval.low_rank_approximate(problem, h, h_eval);
			const auto end_approximate = std::chrono::high_resolution_clock::now();
#endif

#ifdef H_MATRIX_PRINT_APPROX_MATRIX
			h_eval.print();
#endif

			const auto start_eval = std::chrono::high_resolution_clock::now();
#ifdef H_MATRIX_MULTIPLY_REPEAT
			// 時間計測のために H_MATRIX_MULTIPLY_REPEAT 回繰り返す
			for (auto repeat_idx = std::size_t{1}; repeat_idx < std::size_t{H_MATRIX_MULTIPLY_REPEAT}; ++repeat_idx) {
				std::remove_reference_t<EvaluatedSolver>::matrix_vector_multiply(h_eval, x);
			}
#endif
			auto y_eval = std::remove_reference_t<EvaluatedSolver>::matrix_vector_multiply(h_eval, x);
			const auto end_eval = std::chrono::high_resolution_clock::now();

			solver_eval.postprocess(problem, h, x, y_eval);
			const auto res_eval = std::sqrt(vector_norm2(vector_sub(y, y_eval)) / vector_norm2(y));

			std::cout << "residual error of evaluated solver : " << res_eval << std::endl;

			if (res_eval <= solver_eval.tolerance) {
				std::cout << "solver has passed the benchmark test" << std::endl;
			} else {
				std::cout << "solver failed the benchmark test" << std::endl;
			}
			std::cout << "size of matrix : " << problem.matrix_size() << std::endl;

			const auto t_approximate = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_approximate - start_approximate).count()) / std::nano::den;
			std::cout << "approximate time of evaluated solver : " << t_approximate << "(s)" << std::endl;

#ifdef H_MATRIX_MULTIPLY_REPEAT
			const auto t_eval = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_eval - start_eval).count()) / std::nano::den / H_MATRIX_MULTIPLY_REPEAT;
			std::cout << "calc time of evaluated solver (average of " << H_MATRIX_MULTIPLY_REPEAT << " times): " << t_eval << "(s)" << std::endl;
#else
			const auto t_eval = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_eval - start_eval).count()) / std::nano::den;
			std::cout << "calc time of evaluated solver : " << t_eval << "(s)" << std::endl;
#endif

#ifdef ENABLE_PRINT_ERROR
			std::cerr << N << "," << problem.matrix_size() << "," << res_eval << "," << t_approximate << "," << t_eval << std::endl;
#endif
		}

		/// @brief 密行列とベクトルの積
		static auto matrix_vector_multiply(const Matrix<T> &matrix, const Vector<T> &vector) {
			const auto M = matrix.msize();
			const auto N = matrix.nsize();
			if (N != vector.size()) {
				throw std::runtime_error("Vector size not matched.");
			}
			auto result = Vector<T>{M};
#pragma omp parallel for
			for (auto i = decltype(M){0}; i < M; ++i) {
				for (auto j = decltype(N){0}; j < N; ++j) {
					result(i) += matrix(i, j) * vector(j);
				}
			}
			return result;
		}

		/// @brief ベクトル同士の減算
		static auto vector_sub(const Vector<T> &lhs, const Vector<T> &rhs) {
			const auto N = lhs.size();
			if (N != rhs.size()) {
				throw std::runtime_error("Vector size not matched.");
			}
			auto result = Vector<T>{N};
			for (auto i = decltype(N){0}; i < N; ++i) {
				result(i) = lhs(i) - rhs(i);
			}
			return result;
		}

		/// @brief ベクトル同士のドット積
		static auto vector_dot_product(const Vector<T> &lhs, const Vector<T> &rhs) {
			const auto N = lhs.size();
			if (N != rhs.size()) {
				throw std::runtime_error("Vector size not matched.");
			}
			auto result = T{0};
			for (auto i = decltype(N){0}; i < N; ++i) {
				result += lhs(i) * rhs(i);
			}
			return result;
		}

		/// @brief ベクトルの二乗ノルム
		static auto vector_norm2(const Vector<T> &vector) noexcept {
			return vector_dot_product(vector, vector);
		}
	};
} // namespace h_matrix

#endif // H_MATRIX_BENCHMARK_HPP
