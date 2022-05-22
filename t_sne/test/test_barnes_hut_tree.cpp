#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#define ENABLE_TEST
#include <barnes_hut_tree.hpp>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace t_sne {
	using Output = double;
	using embedding_t = BarnesHutTree<Output>::embedding_t;
	template <class Output>
	static constexpr Output EPSILON;
	template <>
	static constexpr auto EPSILON<double> = 1E-15;

	struct BarnesHutTreeTest : ::testing::Test {
		using Node = BarnesHutTree<Output>::Node;
		static constexpr auto WIDTH_PADDING = Node::WIDTH_PADDING;
		static constexpr auto INDEX_EMPTY = Node::INDEX_EMPTY;
		static constexpr auto INDEX_POINT = Node::INDEX_POINT;
		static constexpr auto INDEX_CHILDREN = Node::INDEX_CHILDREN;

		static const Node &get_root(const BarnesHutTree<Output> &tree) {
			return tree._root;
		}

		static void test_node(const Node &node, const Node *parent, const std::vector<embedding_t> *y, const embedding_t min, const embedding_t max, const std::size_t variant) {
			if (parent != nullptr) {
				EXPECT_TRUE(node.parent().has_value());
				EXPECT_EQ(&node.parent()->get(), parent);
			} else {
				EXPECT_FALSE(node.parent().has_value());
			}
			EXPECT_EQ(&node.y(), y);
			EXPECT_NEAR(min[0], node.min()[0], EPSILON<Output>);
			EXPECT_NEAR(min[1], node.min()[1], EPSILON<Output>);
			EXPECT_NEAR(max[0], node.max()[0], EPSILON<Output>);
			EXPECT_NEAR(max[1], node.max()[1], EPSILON<Output>);
			EXPECT_EQ(variant, node.content().index());
		}

		static void test_node(const Node &node, const Node *parent, const std::vector<embedding_t> *y, const embedding_t min, const embedding_t max, const std::size_t variant, const embedding_t y_cell, const std::size_t n_cell, const Output r_cell) {
			if (parent != nullptr) {
				EXPECT_TRUE(node.parent().has_value());
				EXPECT_EQ(&node.parent()->get(), parent);
			} else {
				EXPECT_FALSE(node.parent().has_value());
			}
			EXPECT_EQ(&node.y(), y);
			EXPECT_NEAR(min[0], node.min()[0], EPSILON<Output>);
			EXPECT_NEAR(min[1], node.min()[1], EPSILON<Output>);
			EXPECT_NEAR(max[0], node.max()[0], EPSILON<Output>);
			EXPECT_NEAR(max[1], node.max()[1], EPSILON<Output>);
			EXPECT_EQ(variant, node.content().index());
			const auto y_cell_node = node.y_cell();
			for (auto i = decltype(Node::EMBEDDING_DIMENSION){0}; i < Node::EMBEDDING_DIMENSION; ++i) {
				EXPECT_NEAR(y_cell[i], y_cell_node[i], EPSILON<Output>);
			}
			EXPECT_EQ(n_cell, node.n_cell());
			EXPECT_NEAR(r_cell, node.r_cell(), EPSILON<Output>);
		}

		template <class F>
		static decltype(auto) fix(F &&f) noexcept {
			return [f = std::forward<F>(f)](auto &&... args) {
				return f(f, std::forward<decltype(args)>(args)...);
			};
		}
	};

	TEST_F(BarnesHutTreeTest, Apply) {
		const auto init = std::vector<embedding_t>{{3, 3}, {2, 3}, {3, 2}, {2, 2}, {1, 3}, {0, 3}, {1, 2}, {0, 2}, {3, 1}, {2, 1}, {3, 0}, {2, 0}, {1, 1}, {0, 1}, {1, 0}, {0, 0}};
		const auto tree = BarnesHutTree<Output>(init);
		const auto points = tree.apply(fix([](auto count_point, const typename BarnesHutTree<Output>::Node &node) -> std::size_t {
			const auto &content = node.content();
			switch (content.index()) {
			case BarnesHutTree<Output>::Node::INDEX_EMPTY:
				return 0;
			case BarnesHutTree<Output>::Node::INDEX_POINT:
				return 1;
			default: // case BarnesHutTree<Output>::Node::INDEX_CHILDREN:
			{
				const auto &children = std::get<BarnesHutTree<Output>::Node::INDEX_CHILDREN>(content);
				auto count = std::size_t{0};
				for (const auto &child_node : children) {
					count += count_point(count_point, child_node);
				}
				return count;
			}
			}
		}));
		EXPECT_EQ(init.size(), points);
	}

	TEST_F(BarnesHutTreeTest, EmptyTree) {
		const auto init = std::vector<embedding_t>{};
		const auto tree = BarnesHutTree<Output>(init);
		const auto &root = get_root(tree);
		test_node(root, nullptr, &init, {0, 0}, {0, 0}, INDEX_EMPTY);
	}

	TEST_F(BarnesHutTreeTest, OnePointTree) {
		const auto init = std::vector<embedding_t>{{0, 0}};
		const auto tree = BarnesHutTree<Output>(init);
		const auto &root = get_root(tree);
		test_node(root, nullptr, &init, {-WIDTH_PADDING / 2, -WIDTH_PADDING / 2}, {WIDTH_PADDING / 2, WIDTH_PADDING / 2}, INDEX_POINT, {0, 0}, 1, WIDTH_PADDING * std::sqrt(2));
		EXPECT_EQ(0, std::get<INDEX_POINT>(root.content()));
	}

	TEST_F(BarnesHutTreeTest, TwoPointTree) {
		const auto init = std::vector<embedding_t>{{0, 0}, {1, 1}};
		const auto tree = BarnesHutTree<Output>(init);
		const auto &root = get_root(tree);
		test_node(root, nullptr, &init, {-WIDTH_PADDING / 2, -WIDTH_PADDING / 2}, {1 + WIDTH_PADDING / 2, 1 + WIDTH_PADDING / 2}, INDEX_CHILDREN, {Output{1} / 2, Output{1} / 2}, 2, (WIDTH_PADDING + 1) * std::sqrt(2));
		const auto &children = std::get<INDEX_CHILDREN>(root.content());
		test_node(children[0], &root, &init, {Output{1} / 2, Output{1} / 2}, {1 + WIDTH_PADDING / 2, 1 + WIDTH_PADDING / 2}, INDEX_POINT, {1, 1}, 1, (WIDTH_PADDING + 1) * std::sqrt(2) / 2);
		EXPECT_EQ(1, std::get<INDEX_POINT>(children[0].content()));
		test_node(children[1], &root, &init, {-WIDTH_PADDING / 2, Output{1} / 2}, {Output{1} / 2, 1 + WIDTH_PADDING / 2}, INDEX_EMPTY);
		test_node(children[2], &root, &init, {Output{1} / 2, -WIDTH_PADDING / 2}, {1 + WIDTH_PADDING / 2, Output{1} / 2}, INDEX_EMPTY);
		test_node(children[3], &root, &init, {-WIDTH_PADDING / 2, -WIDTH_PADDING / 2}, {Output{1} / 2, Output{1} / 2}, INDEX_POINT, {0, 0}, 1, (WIDTH_PADDING + 1) * std::sqrt(2) / 2);
		EXPECT_EQ(0, std::get<INDEX_POINT>(children[3].content()));
	} // namespace t_sne

	TEST_F(BarnesHutTreeTest, TwoPointTree_MidPoint) {
		const auto init = std::vector<embedding_t>{{0, 0}, {1, 0}};
		const auto tree = BarnesHutTree<Output>(init);
		const auto &root = get_root(tree);
		test_node(root, nullptr, &init, {-WIDTH_PADDING / 2, -WIDTH_PADDING / 2}, {WIDTH_PADDING / 2 + 1, WIDTH_PADDING / 2}, INDEX_CHILDREN, {Output{1} / 2, 0}, 2, std::sqrt((WIDTH_PADDING + 1) * (WIDTH_PADDING + 1) + WIDTH_PADDING * WIDTH_PADDING));
		const auto &children = std::get<INDEX_CHILDREN>(root.content());
		test_node(children[0], &root, &init, {Output{1} / 2, 0}, {WIDTH_PADDING / 2 + 1, WIDTH_PADDING / 2}, INDEX_POINT, {1, 0}, 1, std::sqrt((WIDTH_PADDING + 1) * (WIDTH_PADDING + 1) + WIDTH_PADDING * WIDTH_PADDING) / 2);
		EXPECT_EQ(1, std::get<INDEX_POINT>(children[0].content()));
		test_node(children[1], &root, &init, {-WIDTH_PADDING / 2, 0}, {Output{1} / 2, WIDTH_PADDING / 2}, INDEX_POINT, {0, 0}, 1, std::sqrt((WIDTH_PADDING + 1) * (WIDTH_PADDING + 1) + WIDTH_PADDING * WIDTH_PADDING) / 2);
		EXPECT_EQ(0, std::get<INDEX_POINT>(children[1].content()));
		test_node(children[2], &root, &init, {Output{1} / 2, -WIDTH_PADDING / 2}, {WIDTH_PADDING / 2 + 1, 0}, INDEX_EMPTY);
		test_node(children[3], &root, &init, {-WIDTH_PADDING / 2, -WIDTH_PADDING / 2}, {Output{1} / 2, 0}, INDEX_EMPTY);
	}

	TEST_F(BarnesHutTreeTest, ThreePointTree_Depth1) {
		const auto init = std::vector<embedding_t>{{0, 0}, {1, 1}, {1, 0}};
		const auto tree = BarnesHutTree<Output>(init);
		const auto &root = get_root(tree);
		test_node(root, nullptr, &init, {-WIDTH_PADDING / 2, -Output{1} / 3 - WIDTH_PADDING / 2}, {Output{4} / 3 + WIDTH_PADDING / 2, 1 + WIDTH_PADDING / 2}, INDEX_CHILDREN, {Output{2} / 3, Output{1} / 3}, 3, (WIDTH_PADDING + Output{4} / 3) * std::sqrt(2));
		const auto &children = std::get<INDEX_CHILDREN>(root.content());
		test_node(children[0], &root, &init, {Output{2} / 3, Output{1} / 3}, {Output{4} / 3 + WIDTH_PADDING / 2, 1 + WIDTH_PADDING / 2}, INDEX_POINT, {1, 1}, 1, (WIDTH_PADDING + Output{4} / 3) * std::sqrt(2) / 2);
		EXPECT_EQ(1, std::get<INDEX_POINT>(children[0].content()));
		test_node(children[1], &root, &init, {-WIDTH_PADDING / 2, Output{1} / 3}, {Output{2} / 3, 1 + WIDTH_PADDING / 2}, INDEX_EMPTY);
		test_node(children[2], &root, &init, {Output{2} / 3, -Output{1} / 3 - WIDTH_PADDING / 2}, {Output{4} / 3 + WIDTH_PADDING / 2, Output{1} / 3}, INDEX_POINT, {1, 0}, 1, (WIDTH_PADDING + Output{4} / 3) * std::sqrt(2) / 2);
		EXPECT_EQ(2, std::get<INDEX_POINT>(children[2].content()));
		test_node(children[3], &root, &init, {-WIDTH_PADDING / 2, -Output{1} / 3 - WIDTH_PADDING / 2}, {Output{2} / 3, Output{1} / 3}, INDEX_POINT, {0, 0}, 1, (WIDTH_PADDING + Output{4} / 3) * std::sqrt(2) / 2);
		EXPECT_EQ(0, std::get<INDEX_POINT>(children[3].content()));
	}

	TEST_F(BarnesHutTreeTest, ThreePointTree_Depth2) {
		const auto init = std::vector<embedding_t>{{0, 0}, {1, 1}, {3, 3}};
		const auto tree = BarnesHutTree<Output>(init);
		const auto &root = get_root(tree);
		test_node(root, nullptr, &init, {-Output{1} / 3 - WIDTH_PADDING / 2, -Output{1} / 3 - WIDTH_PADDING / 2}, {3 + WIDTH_PADDING / 2, 3 + WIDTH_PADDING / 2}, INDEX_CHILDREN, {Output{4} / 3, Output{4} / 3}, 3, (WIDTH_PADDING + Output{10} / 3) * std::sqrt(2));
		const auto &children = std::get<INDEX_CHILDREN>(root.content());
		test_node(children[0], &root, &init, {Output{4} / 3, Output{4} / 3}, {3 + WIDTH_PADDING / 2, 3 + WIDTH_PADDING / 2}, INDEX_POINT, {3, 3}, 1, (WIDTH_PADDING + Output{10} / 3) * std::sqrt(2) / 2);
		EXPECT_EQ(2, std::get<INDEX_POINT>(children[0].content()));
		test_node(children[1], &root, &init, {-Output{1} / 3 - WIDTH_PADDING / 2, Output{4} / 3}, {Output{4} / 3, 3 + WIDTH_PADDING / 2}, INDEX_EMPTY);
		test_node(children[2], &root, &init, {Output{4} / 3, -Output{1} / 3 - WIDTH_PADDING / 2}, {3 + WIDTH_PADDING / 2, Output{4} / 3}, INDEX_EMPTY);
		test_node(children[3], &root, &init, {-Output{1} / 3 - WIDTH_PADDING / 2, -Output{1} / 3 - WIDTH_PADDING / 2}, {Output{4} / 3, Output{4} / 3}, INDEX_CHILDREN, {Output{1} / 2, Output{1} / 2}, 2, (WIDTH_PADDING + Output{10} / 3) * std::sqrt(2) / 2);
		const auto &children3 = std::get<INDEX_CHILDREN>(children[3].content());
		test_node(children3[0], &children[3], &init, {Output{1} / 2 - WIDTH_PADDING / 4, Output{1} / 2 - WIDTH_PADDING / 4}, {Output{4} / 3, Output{4} / 3}, INDEX_POINT, {1, 1}, 1, (WIDTH_PADDING + Output{10} / 3) * std::sqrt(2) / 4);
		EXPECT_EQ(1, std::get<INDEX_POINT>(children3[0].content()));
		test_node(children3[1], &children[3], &init, {-Output{1} / 3 - WIDTH_PADDING / 2, Output{1} / 2 - WIDTH_PADDING / 4}, {Output{1} / 2 - WIDTH_PADDING / 4, Output{4} / 3}, INDEX_EMPTY);
		test_node(children3[2], &children[3], &init, {Output{1} / 2 - WIDTH_PADDING / 4, -Output{1} / 3 - WIDTH_PADDING / 2}, {Output{4} / 3, Output{1} / 2 - WIDTH_PADDING / 4}, INDEX_EMPTY);
		test_node(children3[3], &children[3], &init, {-Output{1} / 3 - WIDTH_PADDING / 2, -Output{1} / 3 - WIDTH_PADDING / 2}, {Output{1} / 2 - WIDTH_PADDING / 4, Output{1} / 2 - WIDTH_PADDING / 4}, INDEX_POINT, {0, 0}, 1, (WIDTH_PADDING + Output{10} / 3) * std::sqrt(2) / 4);
		EXPECT_EQ(0, std::get<INDEX_POINT>(children3[3].content()));
	}

	TEST_F(BarnesHutTreeTest, SixteenPointTree_Depth2) {
		const auto init = std::vector<embedding_t>{{3, 3}, {2, 3}, {3, 2}, {2, 2}, {1, 3}, {0, 3}, {1, 2}, {0, 2}, {3, 1}, {2, 1}, {3, 0}, {2, 0}, {1, 1}, {0, 1}, {1, 0}, {0, 0}};
		const auto tree = BarnesHutTree<Output>(init);
		const auto &root = get_root(tree);
		test_node(root, nullptr, &init, {-WIDTH_PADDING / 2, -WIDTH_PADDING / 2}, {WIDTH_PADDING / 2 + 3, WIDTH_PADDING / 2 + 3}, INDEX_CHILDREN, {Output{3} / 2, Output{3} / 2}, 16, (WIDTH_PADDING + 3) * std::sqrt(2));
		const auto &children = std::get<INDEX_CHILDREN>(root.content());

		test_node(children[0], &root, &init, {Output{3} / 2, Output{3} / 2}, {WIDTH_PADDING / 2 + 3, WIDTH_PADDING / 2 + 3}, INDEX_CHILDREN, {Output{5} / 2, Output{5} / 2}, 4, (WIDTH_PADDING + 3) * std::sqrt(2) / 2);
		const auto &children0 = std::get<INDEX_CHILDREN>(children[0].content());
		test_node(children0[0], &children[0], &init, {WIDTH_PADDING / 4 + Output{9} / 4, WIDTH_PADDING / 4 + Output{9} / 4}, {WIDTH_PADDING / 2 + 3, WIDTH_PADDING / 2 + 3}, INDEX_POINT, {3, 3}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(0, std::get<INDEX_POINT>(children0[0].content()));
		test_node(children0[1], &children[0], &init, {Output{3} / 2, WIDTH_PADDING / 4 + Output{9} / 4}, {WIDTH_PADDING / 4 + Output{9} / 4, WIDTH_PADDING / 2 + 3}, INDEX_POINT, {2, 3}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(1, std::get<INDEX_POINT>(children0[1].content()));
		test_node(children0[2], &children[0], &init, {WIDTH_PADDING / 4 + Output{9} / 4, Output{3} / 2}, {WIDTH_PADDING / 2 + 3, WIDTH_PADDING / 4 + Output{9} / 4}, INDEX_POINT, {3, 2}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(2, std::get<INDEX_POINT>(children0[2].content()));
		test_node(children0[3], &children[0], &init, {Output{3} / 2, Output{3} / 2}, {WIDTH_PADDING / 4 + Output{9} / 4, WIDTH_PADDING / 4 + Output{9} / 4}, INDEX_POINT, {2, 2}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(3, std::get<INDEX_POINT>(children0[3].content()));

		test_node(children[1], &root, &init, {-WIDTH_PADDING / 2, Output{3} / 2}, {Output{3} / 2, 3 + WIDTH_PADDING / 2}, INDEX_CHILDREN, {Output{1} / 2, Output{5} / 2}, 4, (WIDTH_PADDING + 3) * std::sqrt(2) / 2);
		const auto &children1 = std::get<INDEX_CHILDREN>(children[1].content());
		test_node(children1[0], &children[1], &init, {Output{3} / 4 - WIDTH_PADDING / 4, Output{9} / 4 + WIDTH_PADDING / 4}, {Output{3} / 2, 3 + WIDTH_PADDING / 2}, INDEX_POINT, {1, 3}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(4, std::get<INDEX_POINT>(children1[0].content()));
		test_node(children1[1], &children[1], &init, {-WIDTH_PADDING / 2, Output{9} / 4 + WIDTH_PADDING / 4}, {Output{3} / 4 - WIDTH_PADDING / 4, 3 + WIDTH_PADDING / 2}, INDEX_POINT, {0, 3}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(5, std::get<INDEX_POINT>(children1[1].content()));
		test_node(children1[2], &children[1], &init, {Output{3} / 4 - WIDTH_PADDING / 4, Output{3} / 2}, {Output{3} / 2, Output{9} / 4 + WIDTH_PADDING / 4}, INDEX_POINT, {1, 2}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(6, std::get<INDEX_POINT>(children1[2].content()));
		test_node(children1[3], &children[1], &init, {-WIDTH_PADDING / 2, Output{3} / 2}, {Output{3} / 4 - WIDTH_PADDING / 4, Output{9} / 4 + WIDTH_PADDING / 4}, INDEX_POINT, {0, 2}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(7, std::get<INDEX_POINT>(children1[3].content()));

		test_node(children[2], &root, &init, {Output{3} / 2, -WIDTH_PADDING / 2}, {3 + WIDTH_PADDING / 2, Output{3} / 2}, INDEX_CHILDREN, {Output{5} / 2, Output{1} / 2}, 4, (WIDTH_PADDING + 3) * std::sqrt(2) / 2);
		const auto &children2 = std::get<INDEX_CHILDREN>(children[2].content());
		test_node(children2[0], &children[2], &init, {Output{9} / 4 + WIDTH_PADDING / 4, Output{3} / 4 - WIDTH_PADDING / 4}, {3 + WIDTH_PADDING / 2, Output{3} / 2}, INDEX_POINT, {3, 1}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(8, std::get<INDEX_POINT>(children2[0].content()));
		test_node(children2[1], &children[2], &init, {Output{3} / 2, Output{3} / 4 - WIDTH_PADDING / 4}, {Output{9} / 4 + WIDTH_PADDING / 4, Output{3} / 2}, INDEX_POINT, {2, 1}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(9, std::get<INDEX_POINT>(children2[1].content()));
		test_node(children2[2], &children[2], &init, {Output{9} / 4 + WIDTH_PADDING / 4, -WIDTH_PADDING / 2}, {3 + WIDTH_PADDING / 2, Output{3} / 4 - WIDTH_PADDING / 4}, INDEX_POINT, {3, 0}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(10, std::get<INDEX_POINT>(children2[2].content()));
		test_node(children2[3], &children[2], &init, {Output{3} / 2, -WIDTH_PADDING / 2}, {Output{9} / 4 + WIDTH_PADDING / 4, Output{3} / 4 - WIDTH_PADDING / 4}, INDEX_POINT, {2, 0}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(11, std::get<INDEX_POINT>(children2[3].content()));

		test_node(children[3], &root, &init, {-WIDTH_PADDING / 2, -WIDTH_PADDING / 2}, {Output{3} / 2, Output{3} / 2}, INDEX_CHILDREN, {Output{1} / 2, Output{1} / 2}, 4, (WIDTH_PADDING + 3) * std::sqrt(2) / 2);
		const auto &children3 = std::get<INDEX_CHILDREN>(children[3].content());
		test_node(children3[0], &children[3], &init, {Output{3} / 4 - WIDTH_PADDING / 4, Output{3} / 4 - WIDTH_PADDING / 4}, {Output{3} / 2, Output{3} / 2}, INDEX_POINT, {1, 1}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(12, std::get<INDEX_POINT>(children3[0].content()));
		test_node(children3[1], &children[3], &init, {-WIDTH_PADDING / 2, Output{3} / 4 - WIDTH_PADDING / 4}, {Output{3} / 4 - WIDTH_PADDING / 4, Output{3} / 2}, INDEX_POINT, {0, 1}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(13, std::get<INDEX_POINT>(children3[1].content()));
		test_node(children3[2], &children[3], &init, {Output{3} / 4 - WIDTH_PADDING / 4, -WIDTH_PADDING / 2}, {Output{3} / 2, Output{3} / 4 - WIDTH_PADDING / 4}, INDEX_POINT, {1, 0}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(14, std::get<INDEX_POINT>(children3[2].content()));
		test_node(children3[3], &children[3], &init, {-WIDTH_PADDING / 2, -WIDTH_PADDING / 2}, {Output{3} / 4 - WIDTH_PADDING / 4, Output{3} / 4 - WIDTH_PADDING / 4}, INDEX_POINT, {0, 0}, 1, (WIDTH_PADDING + 3) * std::sqrt(2) / 4);
		EXPECT_EQ(15, std::get<INDEX_POINT>(children3[3].content()));
	}
} // namespace t_sne
#pragma clang diagnostic pop
