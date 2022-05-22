#define ENABLE_TEST
#ifdef ENABLE_TEST
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#endif
#include <benchmark.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace h_matrix {
	using T = double;

	struct BenchmarkTest : ::testing::Test {};

	TEST_F(BenchmarkTest, MatrixVectorMultiply) {
		auto matrix = Matrix<int>{3, 2};
		for (auto i = std::size_t{0}; i < 3; ++i) {
			for (auto j = std::size_t{0}; j < 2; ++j) {
				matrix(i, j) = static_cast<int>(i * 2 + j + 1);
			}
		}
		auto vector = Vector<int>{2};
		for (auto i = std::size_t{0}; i < 2; ++i) {
			vector(i) = static_cast<int>(i + 7);
		}
		const auto result = Benchmark<int>::matrix_vector_multiply(matrix, vector);
		EXPECT_EQ(3, result.size());
		EXPECT_EQ(23, result(0));
		EXPECT_EQ(53, result(1));
		EXPECT_EQ(83, result(2));
	}

	TEST_F(BenchmarkTest, VectorSub) {
		auto lhs = Vector<int>{3};
		auto rhs = Vector<int>{3};
		lhs(0) = 1;
		lhs(1) = 2;
		lhs(2) = 3;
		rhs(0) = 4;
		rhs(1) = 5;
		rhs(2) = 6;
		const auto result = Benchmark<int>::vector_sub(lhs, rhs);
		for (auto i = std::size_t{0}; i < 3; ++i) {
			EXPECT_EQ(-3, result(i));
		}
	}

	TEST_F(BenchmarkTest, VectorDot) {
		auto lhs = Vector<int>{3};
		auto rhs = Vector<int>{3};
		lhs(0) = 1;
		lhs(1) = 2;
		lhs(2) = 3;
		rhs(0) = 4;
		rhs(1) = 5;
		rhs(2) = 6;
		EXPECT_EQ(32, Benchmark<int>::vector_dot_product(lhs, rhs));
	}

	TEST_F(BenchmarkTest, VectorNorm2) {
		auto vector = Vector<int>{3};
		vector(0) = 1;
		vector(1) = 2;
		vector(2) = 3;

		EXPECT_EQ(14, Benchmark<int>::vector_norm2(vector));
	}

} // namespace h_matrix

#pragma clang diagnostic pop
