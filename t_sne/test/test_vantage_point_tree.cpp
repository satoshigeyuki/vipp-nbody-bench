#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#define ENABLE_TEST
#include <solver_ref.hpp>
#include <vantage_point_tree.hpp>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace t_sne {
	using Output = double;
	using image_t = SolverRef<Output>::image_t;
	template <class Output>
	constexpr Output EPSILON;
	template <>
	static constexpr auto EPSILON<double> = 1E-15;

	struct VantagePointTreeTest : ::testing::Test {
		using tree_t = VantagePointTree<image_t, SolverRef<Output>::distance_x>;

		const std::vector<image_t> &get_x(const tree_t &tree) {
			return tree._x;
		}

		const std::vector<std::size_t> &get_tree(const tree_t &tree) {
			return tree._tree;
		}

		const std::vector<std::size_t> &get_indices(const tree_t &tree) {
			return tree._indices;
		}

		const std::vector<Output> &get_distances(const tree_t &tree) {
			return tree._distances;
		}

		constexpr Output distance_x(const image_t &x1, const image_t &x2) noexcept {
			return SolverRef<Output>::distance_x(x1, x2);
		}

		void test_nearest(const std::size_t i, const std::size_t j, const tree_t &tree, const std::size_t expect_index) {
			const auto k = tree.k();
			EXPECT_EQ(expect_index, tree._indices[i * k + j]);
			EXPECT_NEAR(distance_x(tree._x[i], tree._x[expect_index]), tree._distances[i * k + j], EPSILON<Output>);
		}
	};

	TEST_F(VantagePointTreeTest, EmptyConstructTest) {
		const auto x = std::vector<image_t>{};
		const auto tree = tree_t{x};
		EXPECT_EQ(&x, &get_x(tree));
		const auto &tree_impl = get_tree(tree);
		EXPECT_EQ(0, tree_impl.size());
	}

	TEST_F(VantagePointTreeTest, OnePointConstructTest) {
		const auto x = std::vector<image_t>{{}};
		const auto tree = tree_t{x};
		EXPECT_EQ(&x, &get_x(tree));
		const auto &tree_impl = get_tree(tree);
		EXPECT_EQ(1, tree_impl.size());
		EXPECT_EQ(0, tree_impl[0]);
	}

	TEST_F(VantagePointTreeTest, TwoPointConstructTest) {
		const auto x = std::vector<image_t>{{}, {1}};
		const auto tree = tree_t{x};
		EXPECT_EQ(&x, &get_x(tree));
		const auto &tree_impl = get_tree(tree);
		EXPECT_EQ(2, tree_impl.size());
		EXPECT_EQ(0, tree_impl[0]);
		EXPECT_EQ(1, tree_impl[1]);
	}

	TEST_F(VantagePointTreeTest, ThreePointConstructTest) {
		const auto x = std::vector<image_t>{{}, {3}, {1}};
		const auto tree = tree_t{x};
		EXPECT_EQ(&x, &get_x(tree));
		const auto &tree_impl = get_tree(tree);
		EXPECT_EQ(3, tree_impl.size());
		EXPECT_EQ(0, tree_impl[0]);
		EXPECT_EQ(2, tree_impl[1]);
		EXPECT_EQ(1, tree_impl[2]);
	}

	TEST_F(VantagePointTreeTest, FourPointConstructTest) {
		const auto x = std::vector<image_t>{{}, {3}, {1}, {5}};
		const auto tree = tree_t{x};
		EXPECT_EQ(&x, &get_x(tree));
		const auto &tree_impl = get_tree(tree);
		EXPECT_EQ(4, tree_impl.size());
		EXPECT_EQ(0, tree_impl[0]);
		EXPECT_EQ(2, tree_impl[1]);
		EXPECT_EQ(1, tree_impl[2]);
		EXPECT_EQ(3, tree_impl[3]);
	}

	TEST_F(VantagePointTreeTest, TenPointConstructTest) {
		const auto x = std::vector<image_t>{{}, {7}, {4}, {9}, {3}, {8}, {2}, {6}, {5}, {1}};
		const auto tree = tree_t{x};
		EXPECT_EQ(&x, &get_x(tree));
		const auto &tree_impl = get_tree(tree);
		EXPECT_EQ(10, tree_impl.size());
		EXPECT_EQ(0, tree_impl[0]);
		EXPECT_EQ(8, tree_impl[5]);

		EXPECT_TRUE(distance_x(x[tree_impl[0]], x[tree_impl[1]]) <= distance_x(x[tree_impl[0]], x[tree_impl[5]]));
		EXPECT_TRUE(distance_x(x[tree_impl[0]], x[tree_impl[2]]) <= distance_x(x[tree_impl[0]], x[tree_impl[5]]));
		EXPECT_TRUE(distance_x(x[tree_impl[0]], x[tree_impl[3]]) <= distance_x(x[tree_impl[0]], x[tree_impl[5]]));
		EXPECT_TRUE(distance_x(x[tree_impl[0]], x[tree_impl[4]]) <= distance_x(x[tree_impl[0]], x[tree_impl[5]]));
		EXPECT_TRUE(distance_x(x[tree_impl[0]], x[tree_impl[6]]) >= distance_x(x[tree_impl[0]], x[tree_impl[5]]));
		EXPECT_TRUE(distance_x(x[tree_impl[0]], x[tree_impl[7]]) >= distance_x(x[tree_impl[0]], x[tree_impl[5]]));
		EXPECT_TRUE(distance_x(x[tree_impl[0]], x[tree_impl[8]]) >= distance_x(x[tree_impl[0]], x[tree_impl[5]]));
		EXPECT_TRUE(distance_x(x[tree_impl[0]], x[tree_impl[9]]) >= distance_x(x[tree_impl[0]], x[tree_impl[5]]));

		EXPECT_TRUE(distance_x(x[tree_impl[1]], x[tree_impl[2]]) <= distance_x(x[tree_impl[1]], x[tree_impl[3]]));
		EXPECT_TRUE(distance_x(x[tree_impl[1]], x[tree_impl[4]]) >= distance_x(x[tree_impl[1]], x[tree_impl[3]]));
		EXPECT_TRUE(distance_x(x[tree_impl[5]], x[tree_impl[6]]) <= distance_x(x[tree_impl[5]], x[tree_impl[8]]));
		EXPECT_TRUE(distance_x(x[tree_impl[5]], x[tree_impl[7]]) <= distance_x(x[tree_impl[5]], x[tree_impl[8]]));
		EXPECT_TRUE(distance_x(x[tree_impl[5]], x[tree_impl[9]]) >= distance_x(x[tree_impl[5]], x[tree_impl[8]]));
	}

	TEST_F(VantagePointTreeTest, EmptySearchTest) {
		const auto x = std::vector<image_t>{};
		auto tree = tree_t{x};
		EXPECT_THROW(tree.search(1), std::runtime_error);
	}

	TEST_F(VantagePointTreeTest, OnePointSearchTest) {
		const auto x = std::vector<image_t>{{}};
		auto tree = tree_t{x};
		EXPECT_THROW(tree.search(1), std::runtime_error);
	}

	TEST_F(VantagePointTreeTest, TwoPointSearchTest) {
		const auto x = std::vector<image_t>{{}, {1}};
		auto tree = tree_t{x};
		tree.search(1);
		const auto &indices = get_indices(tree);
		const auto &distances = get_distances(tree);
		EXPECT_EQ(2, indices.size());
		EXPECT_EQ(2, distances.size());
		const auto &index_answer = std::vector<std::vector<std::size_t>>{{1}, {0}};
		for (auto i = decltype(index_answer.size()){0}; i < index_answer.size(); ++i) {
			for (auto j = decltype(index_answer[0].size()){0}; j < index_answer[i].size(); ++j) {
				test_nearest(i, j, tree, index_answer[i][j]);
			}
		}
	}

	TEST_F(VantagePointTreeTest, ThreePointSearchTest) {
		const auto x = std::vector<image_t>{{}, {3}, {1}};
		auto tree = tree_t{x};
		tree.search(2);
		const auto &indices = get_indices(tree);
		const auto &distances = get_distances(tree);
		EXPECT_EQ(6, indices.size());
		EXPECT_EQ(6, distances.size());
		const auto &index_answer = std::vector<std::vector<std::size_t>>{{2, 1}, {2, 0}, {0, 1}};
		for (auto i = decltype(index_answer.size()){0}; i < index_answer.size(); ++i) {
			for (auto j = decltype(index_answer[0].size()){0}; j < index_answer[i].size(); ++j) {
				test_nearest(i, j, tree, index_answer[i][j]);
			}
		}
	}

	TEST_F(VantagePointTreeTest, FourPointSearchTest) {
		const auto x = std::vector<image_t>{{}, {3}, {1}, {5}};
		auto tree = tree_t{x};
		tree.search(3);
		const auto &indices = get_indices(tree);
		const auto &distances = get_distances(tree);
		EXPECT_EQ(12, indices.size());
		EXPECT_EQ(12, distances.size());
		const auto &index_answer = std::vector<std::vector<std::size_t>>{{2, 1, 3}, {3, 2, 0}, {0, 1, 3}, {1, 2, 0}};
		for (auto i = decltype(index_answer.size()){0}; i < index_answer.size(); ++i) {
			for (auto j = decltype(index_answer[0].size()){0}; j < index_answer[i].size(); ++j) {
				test_nearest(i, j, tree, index_answer[i][j]);
			}
		}
	}

	TEST_F(VantagePointTreeTest, TenPointSearchTest) {
		const auto x = std::vector<image_t>{{}, {7}, {4}, {9}, {3}, {8}, {2}, {6}, {5}, {1}};
		auto tree = tree_t{x};
		tree.search(4);
		const auto &indices = get_indices(tree);
		const auto &distances = get_distances(tree);
		EXPECT_EQ(40, indices.size());
		EXPECT_EQ(40, distances.size());
		const auto &index_answer = std::vector<std::vector<std::size_t>>{{9, 6, 4, 2}, {7, 5, 3, 8}, {4, 8, 7, 6}, {5, 1, 7, 8}, {2, 6, 8, 9}, {1, 3, 7, 8}, {9, 4, 2, 0}, {8, 1, 2, 5}, {7, 2, 4, 1}, {6, 0, 4, 2}};
		for (auto i = decltype(index_answer.size()){0}; i < index_answer.size(); ++i) {
			for (auto j = decltype(index_answer[0].size()){0}; j < index_answer[i].size(); ++j) {
				test_nearest(i, j, tree, index_answer[i][j]);
			}
		}
	}
} // namespace t_sne
#pragma clang diagnostic pop
