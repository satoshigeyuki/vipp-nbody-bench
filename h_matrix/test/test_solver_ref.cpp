#define ENABLE_TEST
#ifdef ENABLE_TEST
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#endif
#include <benchmark.hpp>
#include <solver_ref.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace h_matrix {
	using T = double;
	template <class U>
	static constexpr U EPSILON;
	template <>
	static constexpr auto EPSILON<double> = 1E-7;

	struct SolverRefTest : ::testing::Test {
	};

	TEST_F(SolverRefTest, SingularValueDecomposition) {
		const auto m = std::size_t{2};
		const auto n = std::size_t{3};
		auto a = Matrix<T>{m, n};
		a(0, 0) = 1;
		a(0, 1) = 0;
		a(0, 2) = 0;
		a(1, 0) = 0;
		a(1, 1) = 1;
		a(1, 2) = 1;
		const auto [u, s, vt] = singular_value_decomposition<T>(a, 0, 0, m, n);
		EXPECT_NEAR(0, u(0, 0), EPSILON<T>);
		EXPECT_NEAR(1, u(0, 1), EPSILON<T>);
		EXPECT_NEAR(1, u(1, 0), EPSILON<T>);
		EXPECT_NEAR(0, u(1, 1), EPSILON<T>);

		EXPECT_NEAR(std::sqrt(2), s(0), EPSILON<T>);
		EXPECT_NEAR(1, s(1), EPSILON<T>);

		EXPECT_NEAR(0, vt(0, 0), EPSILON<T>);
		EXPECT_NEAR(std::sqrt(2) / 2, vt(0, 1), EPSILON<T>);
		EXPECT_NEAR(std::sqrt(2) / 2, vt(0, 2), EPSILON<T>);
		EXPECT_NEAR(1, vt(1, 0), EPSILON<T>);
		EXPECT_NEAR(0, vt(1, 1), EPSILON<T>);
		EXPECT_NEAR(0, vt(1, 2), EPSILON<T>);
		EXPECT_NEAR(0, vt(2, 0), EPSILON<T>);
		EXPECT_NEAR(-std::sqrt(2) / 2, vt(2, 1), EPSILON<T>);
		EXPECT_NEAR(std::sqrt(2) / 2, vt(2, 2), EPSILON<T>);
	}

	TEST_F(SolverRefTest, SingularValueDecompositionMultiply) {
		const auto m = std::size_t{5};
		const auto n = std::size_t{3};
		auto a = Matrix<T>{m, n};
		for (auto i = decltype(m){0}; i < m; ++i) {
			for (auto j = decltype(n){0}; j < n; ++j) {
				a(i, j) = i * n + j;
			}
		}
		auto vector = Vector<T>{n};
		for (auto i = decltype(n){0}; i < n; ++i) {
			vector(i) = m * n + i;
		}

		const auto [u, s, vt] = singular_value_decomposition<T>(a, 0, 0, m, n);
		const auto res = Benchmark<T>::vector_norm2(Benchmark<T>::vector_sub(Benchmark<T>::matrix_vector_multiply(a, vector), Benchmark<T>::matrix_vector_multiply(u, SolverRef<T>::matrix_vector_multiply(s, Benchmark<T>::matrix_vector_multiply(vt, vector)))));
		EXPECT_NEAR(0, res, EPSILON<T>);
	}

	TEST_F(SolverRefTest, SingularValueDecompositionMultiply2) {
		const auto m = std::size_t{3};
		const auto n = std::size_t{5};
		auto a = Matrix<T>{m, n};
		for (auto i = decltype(m){0}; i < m; ++i) {
			for (auto j = decltype(n){0}; j < n; ++j) {
				a(i, j) = i * n + j;
			}
		}
		auto vector = Vector<T>{n};
		for (auto i = decltype(n){0}; i < n; ++i) {
			vector(i) = m * n + i;
		}

		const auto [u, s, vt] = singular_value_decomposition<T>(a, 0, 0, m, n);
		const auto res = Benchmark<T>::vector_norm2(Benchmark<T>::vector_sub(Benchmark<T>::matrix_vector_multiply(a, vector), Benchmark<T>::matrix_vector_multiply(u, SolverRef<T>::matrix_vector_multiply(s, Benchmark<T>::matrix_vector_multiply(vt, vector)))));
		EXPECT_NEAR(0, res, EPSILON<T>);
	}

	TEST_F(SolverRefTest, SingularValueDecompositionMultiply3) {
		const auto m = std::size_t{3};
		const auto n = std::size_t{3};
		auto a = Matrix<T>{m, n};
		for (auto i = decltype(m){0}; i < m; ++i) {
			for (auto j = decltype(n){0}; j < n; ++j) {
				a(i, j) = i * n + j;
			}
		}
		auto vector = Vector<T>{n};
		for (auto i = decltype(n){0}; i < n; ++i) {
			vector(i) = m * n + i;
		}

		const auto [u, s, vt] = singular_value_decomposition<T>(a, 0, 0, m, n);
		const auto res = Benchmark<T>::vector_norm2(Benchmark<T>::vector_sub(Benchmark<T>::matrix_vector_multiply(a, vector), Benchmark<T>::matrix_vector_multiply(u, SolverRef<T>::matrix_vector_multiply(s, Benchmark<T>::matrix_vector_multiply(vt, vector)))));
		EXPECT_NEAR(0, res, EPSILON<T>);
	}

	TEST_F(SolverRefTest, DiagonalMatrixVectorMultiply) {
		auto matrix = DiagonalMatrix<int>{3, 3};
		auto vector = Vector<int>{3};
		for (auto i = std::size_t{0}; i < 3; ++i) {
			matrix(i) = static_cast<int>(i + 1);
			vector(i) = static_cast<int>(i + 1);
		}
		const auto result = SolverRef<int>::matrix_vector_multiply(matrix, vector);
		EXPECT_EQ(3, result.size());
		EXPECT_EQ(1, result(0));
		EXPECT_EQ(4, result(1));
		EXPECT_EQ(9, result(2));
	}

	TEST_F(SolverRefTest, ApproximateMatrixVectorMultiply) {
		auto matrix = ApproximateMatrix<T>{3, 4};
		matrix.split(0, 1, 1);
		auto &ul = matrix.set_dense(1);
		auto &ur = matrix.set_dense(2);
		auto &ll = matrix.set_dense(3);
		auto &lr = matrix.set_low_rank(4, 2);

		ul.matrix()(0, 0) = 1;
		auto &ur_matrix = ur.matrix();
		ur_matrix(0, 0) = 2;
		ur_matrix(0, 1) = 3;
		ur_matrix(0, 2) = 4;
		auto &ll_matrix = ll.matrix();
		ll_matrix(0, 0) = 5;
		ll_matrix(1, 0) = 6;
		auto &u = lr.u();
		auto &s = lr.s();
		auto &vt = lr.vt();
		u(0, 0) = 0;
		u(0, 1) = 1;
		u(1, 0) = 1;
		u(1, 1) = 0;
		s(0) = std::sqrt(2);
		s(1) = 1;
		vt(0, 0) = 0;
		vt(0, 1) = std::sqrt(2) / 2;
		vt(0, 2) = std::sqrt(2) / 2;
		vt(1, 0) = 1;
		vt(1, 1) = 0;
		vt(1, 2) = 0;
		auto vector = Vector<T>{4};
		for (auto i = std::size_t{0}; i < matrix.nsize(); ++i) {
			vector(i) = static_cast<int>(i + 7);
		}
		const auto result = SolverRef<T>::matrix_vector_multiply(matrix, vector);
		EXPECT_NEAR(90, result(0), EPSILON<T>);
		EXPECT_NEAR(43, result(1), EPSILON<T>);
		EXPECT_NEAR(61, result(2), EPSILON<T>);
	}

	TEST_F(SolverRefTest, ApproximateMatrixVectorMultiply2) {
		const auto level = std::size_t{8};
		auto matrix = ApproximateMatrix<int>{std::size_t{1} << level, std::size_t{1} << level};
		auto vector = Vector<int>{std::size_t{1} << level};
		auto idx = std::size_t{0};
		for (auto l = decltype(level){0}; l <= level; ++l) {
			for (auto i = 0; i < (1 << (l * 2)); ++i) {
				if (l != level) {
					matrix.split(idx++, 1 << (level - l - 1), 1 << (level - l - 1));
				} else {
					auto &dense = matrix.set_dense(idx++);
					dense.matrix()(0, 0) = 1;
				}
			}
		}
		for (auto i = std::size_t{0}; i < (1 << level); ++i) {
			vector(i) = 1;
		}
		const auto result = SolverRef<int>::matrix_vector_multiply(matrix, vector);
		for (auto i = std::size_t{0}; i < (1 << level); ++i) {
			EXPECT_EQ(1 << level, result(i));
		}
	}

	TEST_F(SolverRefTest, LowRankApproximate) {
		const auto problem = Problem<T>{};
		const auto solver = SolverRef<T>{};
		const auto M = std::size_t{128};
		const auto N = std::size_t{128};
		auto matrix = Matrix<T>{M, N};
		auto vector = Vector<T>{N};
		auto approx = ApproximateMatrix<T>{M, N};
		for (auto i = decltype(M){0}; i < M; ++i) {
			for (auto j = decltype(N){0}; j < N; ++j) {
				matrix(i, j) = 1;
			}
		}
		for (auto i = decltype(M){0}; i < N; ++i) {
			vector(i) = 1;
		}
		solver.low_rank_approximate(problem, matrix, approx);
		const auto result = SolverRef<T>::matrix_vector_multiply(approx, vector);
		for (auto i = decltype(M){0}; i < M; ++i) {
			EXPECT_NEAR(static_cast<T>(N), result(i), EPSILON<T>);
		}
	}
} // namespace h_matrix

#pragma clang diagnostic pop
