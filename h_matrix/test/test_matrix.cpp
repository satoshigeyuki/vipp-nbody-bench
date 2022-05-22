#include <limits>
#define ENABLE_TEST
#ifdef ENABLE_TEST
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#endif
#include <matrix.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace h_matrix {
	using T = double;

	struct MatrixTest : ::testing::Test {
		const auto &matrix_impl(const Matrix<T> &matrix) const noexcept {
			return matrix._entity;
		}

		const auto &vector_impl(const Vector<T> &vector) const noexcept {
			return vector._entity;
		}
	};

	TEST_F(MatrixTest, MatrixConstruct) {
		const auto zero = Matrix<T>{3, 2};
		const auto &zero_entity = matrix_impl(zero);
		EXPECT_EQ(3, zero.msize());
		EXPECT_EQ(2, zero.nsize());
		EXPECT_EQ(6, zero_entity.size());
		auto idx = std::size_t{0};
		for (auto i = std::size_t{0}; i < 3; ++i) {
			for (auto j = std::size_t{0}; j < 2; ++j, ++idx) {
				EXPECT_EQ(zero_entity.data() + idx, &zero(i, j));
			}
		}
	}

	TEST_F(MatrixTest, MatrixModify) {
		auto matrix = Matrix<int>{3, 2};
		matrix(0, 0) = 1;
		matrix(0, 1) = 2;
		matrix(1, 0) = 3;
		matrix(1, 1) = 4;
		matrix(2, 0) = 5;
		matrix(2, 1) = 6;
		for (auto i = std::size_t{0}; i < 3; ++i) {
			for (auto j = std::size_t{0}; j < 2; ++j) {
				EXPECT_EQ(i * 2 + j + 1, matrix(i, j));
			}
		}
	}

	TEST_F(MatrixTest, VectorConstruct) {
		const auto zero = Vector<T>{3};
		const auto &zero_entity = vector_impl(zero);
		EXPECT_EQ(3, zero.size());
		EXPECT_EQ(3, zero_entity.size());
		for (auto i = std::size_t{0}; i < 3; ++i) {
			EXPECT_EQ(zero_entity.data() + i, &zero(i));
		}
	}

	TEST_F(MatrixTest, VectorModify) {
		auto vector = Vector<int>{3};
		vector(0) = 1;
		vector(1) = 2;
		vector(2) = 3;
		for (auto i = std::size_t{0}; i < 3; ++i) {
			EXPECT_EQ(i + 1, vector(i));
		}
	}

	TEST_F(MatrixTest, EmptyApprox) {
		const auto matrix = ApproximateMatrix<T>{3, 2};
		EXPECT_EQ(3, matrix.msize());
		EXPECT_EQ(2, matrix.nsize());
		EXPECT_EQ(0, matrix.leaves().size());
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_EMPTY, matrix.root().index());
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_EMPTY, matrix.node(0).index());
		const auto &empty = std::get<ApproximateMatrix<T>::INDEX_EMPTY>(matrix.root());
		EXPECT_EQ(0, empty.idx());
		EXPECT_EQ(0, empty.min_i());
		EXPECT_EQ(0, empty.min_j());
		EXPECT_EQ(3, empty.msize());
		EXPECT_EQ(2, empty.nsize());
	}

	TEST_F(MatrixTest, DenseApprox) {
		auto matrix = ApproximateMatrix<T>{3, 2};
		const auto &dense = matrix.set_dense(0);
		EXPECT_EQ(0, dense.idx());
		EXPECT_EQ(0, dense.min_i());
		EXPECT_EQ(0, dense.min_j());
		EXPECT_EQ(3, dense.msize());
		EXPECT_EQ(2, dense.nsize());
		EXPECT_EQ(1, matrix.leaves().size());
		EXPECT_EQ(0, matrix.leaves()[0]);
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_DENSE, matrix.root().index());
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_DENSE, matrix.node(0).index());
	}

	TEST_F(MatrixTest, LowRankApprox) {
		auto matrix = ApproximateMatrix<T>{3, 2};
		const auto &low_rank = matrix.set_low_rank(0, 2);
		EXPECT_EQ(0, low_rank.idx());
		EXPECT_EQ(0, low_rank.min_i());
		EXPECT_EQ(0, low_rank.min_j());
		EXPECT_EQ(3, low_rank.msize());
		EXPECT_EQ(2, low_rank.nsize());
		EXPECT_EQ(1, matrix.leaves().size());
		EXPECT_EQ(0, matrix.leaves()[0]);
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_LOW_RANK, matrix.root().index());
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_LOW_RANK, matrix.node(0).index());
	}

	TEST_F(MatrixTest, SplitApprox) {
		auto matrix = ApproximateMatrix<T>{3, 2};
		const auto &node = matrix.split(0, 1, 1);
		EXPECT_EQ(0, node.idx());
		EXPECT_EQ(0, node.min_i());
		EXPECT_EQ(0, node.min_j());
		EXPECT_EQ(3, node.msize());
		EXPECT_EQ(2, node.nsize());
		EXPECT_EQ(0, matrix.leaves().size());
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_NODE, matrix.root().index());
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_NODE, matrix.node(0).index());
		const auto &ul = matrix.node(node.upper_left());
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_EMPTY, ul.index());
		const auto &ul_empty = std::get<ApproximateMatrix<T>::INDEX_EMPTY>(ul);
		EXPECT_EQ(1, ul_empty.idx());
		EXPECT_EQ(0, ul_empty.min_i());
		EXPECT_EQ(0, ul_empty.min_j());
		EXPECT_EQ(1, ul_empty.msize());
		EXPECT_EQ(1, ul_empty.nsize());
		const auto &ur = matrix.node(node.upper_right());
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_EMPTY, ur.index());
		const auto &ur_empty = std::get<ApproximateMatrix<T>::INDEX_EMPTY>(ur);
		EXPECT_EQ(2, ur_empty.idx());
		EXPECT_EQ(0, ur_empty.min_i());
		EXPECT_EQ(1, ur_empty.min_j());
		EXPECT_EQ(1, ur_empty.msize());
		EXPECT_EQ(1, ur_empty.nsize());
		const auto &ll = matrix.node(node.lower_left());
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_EMPTY, ll.index());
		const auto &ll_empty = std::get<ApproximateMatrix<T>::INDEX_EMPTY>(ll);
		EXPECT_EQ(3, ll_empty.idx());
		EXPECT_EQ(1, ll_empty.min_i());
		EXPECT_EQ(0, ll_empty.min_j());
		EXPECT_EQ(2, ll_empty.msize());
		EXPECT_EQ(1, ll_empty.nsize());
		const auto &lr = matrix.node(node.lower_right());
		EXPECT_EQ(ApproximateMatrix<T>::INDEX_EMPTY, lr.index());
		const auto &lr_empty = std::get<ApproximateMatrix<T>::INDEX_EMPTY>(lr);
		EXPECT_EQ(4, lr_empty.idx());
		EXPECT_EQ(1, lr_empty.min_i());
		EXPECT_EQ(1, lr_empty.min_j());
		EXPECT_EQ(2, lr_empty.msize());
		EXPECT_EQ(1, lr_empty.nsize());
	}

} // namespace h_matrix

#pragma clang diagnostic pop
