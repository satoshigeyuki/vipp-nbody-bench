#define ENABLE_TEST
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-template"
#include <fmm.hpp>
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace gravity {
	using M = double;
	using T = double;
	using X = double;
	using V = double;
	using A = double;
	using G = double;
	template <class X>
	static constexpr X EPSILON;
	template <>
	static constexpr auto EPSILON<double> = 1E-15;

	struct OctreeTest : ::testing::Test {
		using Cell = Cell<M, T, X, V, A, G>;

		static constexpr auto WIDTH_PADDING = Cell::WIDTH_PADDING;
		static constexpr auto INDEX_POINTS = Octree<Cell, Vector<X>>::INDEX_POINTS;
		static constexpr auto INDEX_CELLS = Octree<Cell, Vector<X>>::INDEX_CELLS;

		static void test_vector(const Vector<X> &required, const Vector<X> &real) {
			EXPECT_NEAR(required.x(), real.x(), EPSILON<X>);
			EXPECT_NEAR(required.y(), real.y(), EPSILON<X>);
			EXPECT_NEAR(required.z(), real.z(), EPSILON<X>);
		}

		// root(points)
		static void test_cell(const Cell &cell, const std::size_t index, const Vector<X> min, const Vector<X> max, const std::vector<std::size_t> content_i) {
			EXPECT_EQ(index, cell.index());
			test_vector(min, cell.min());
			test_vector(max, cell.max());
			const auto &content = cell.content();
			EXPECT_EQ(INDEX_POINTS, content.index());
			const auto &real_i = std::get<INDEX_POINTS>(content);
			const auto N = content_i.size();
			EXPECT_EQ(N, real_i.size());
			for (auto i = decltype(N){0}; i < N; ++i) {
				EXPECT_EQ(content_i[i], real_i[i]);
			}
		}

		// root(cells)
		static auto test_cell(const Cell &cell, const std::size_t index, const Vector<X> min, const Vector<X> max, const std::size_t content_index_base) {
			EXPECT_EQ(index, cell.index());
			test_vector(min, cell.min());
			test_vector(max, cell.max());
			const auto &content = cell.content();
			EXPECT_EQ(INDEX_CELLS, content.index());
			EXPECT_EQ(content_index_base, std::get<INDEX_CELLS>(content));
			return std::get<INDEX_CELLS>(content);
		}

		// children(points)
		static void test_cell(const Cell &cell, const std::size_t parent, const std::size_t index, const Vector<X> min, const Vector<X> max, const std::vector<std::size_t> content_i, const std::size_t n) {
			EXPECT_EQ(parent, cell.parent());
			EXPECT_EQ(index, cell.index());
			test_vector(min, cell.min());
			test_vector(max, cell.max());
			const auto &content = cell.content();
			EXPECT_EQ(INDEX_POINTS, content.index());
			const auto &real_i = std::get<INDEX_POINTS>(content);
			const auto N = content_i.size();
			EXPECT_EQ(N, real_i.size());
			for (auto i = decltype(N){0}; i < N; ++i) {
				EXPECT_EQ(content_i[i], real_i[i]);
			}
			EXPECT_EQ(n, N);
		}

		// children(cells)
		static auto test_cell(const Cell &cell, const std::size_t parent, const std::size_t index, const Vector<X> min, const Vector<X> max, const std::size_t content_index_base) {
			EXPECT_EQ(parent, cell.parent());
			EXPECT_EQ(index, cell.index());
			test_vector(min, cell.min());
			test_vector(max, cell.max());
			const auto &content = cell.content();
			EXPECT_EQ(INDEX_CELLS, content.index());
			EXPECT_EQ(content_index_base, std::get<INDEX_CELLS>(content));
			return std::get<INDEX_CELLS>(content);
		}
	};

	TEST_F(OctreeTest, EmptyTree) {
		const auto x = std::vector<Vector<X>>{};
		const auto s = std::size_t{1};
		const auto tree = Octree<Cell, Vector<X>>{x, s};
		const auto &root = tree.cells().front();
		test_cell(root, 0, {0, 0, 0}, {0, 0, 0}, std::vector<std::size_t>{});
	}

	TEST_F(OctreeTest, OnePointTree) {
		const auto x = std::vector<Vector<X>>{{0, 0, 0}};
		const auto s = std::size_t{1};
		const auto tree = Octree<Cell, Vector<X>>{x, s};
		const auto &root = tree.cells().front();
		test_cell(root, 0, {-WIDTH_PADDING, -WIDTH_PADDING, -WIDTH_PADDING}, {WIDTH_PADDING, WIDTH_PADDING, WIDTH_PADDING}, std::vector<std::size_t>{0});
	}

	TEST_F(OctreeTest, ThreePointTree) {
		const auto x = std::vector<Vector<X>>{{1, 1, 1}, {-1, -1, -1}, {1, -1, 1}};
		const auto s = std::size_t{1};
		const auto tree = Octree<Cell, Vector<X>>{x, s};
		const auto &cells = tree.cells();
		const auto &root = tree.cells().front();
		const auto &children_index_base = test_cell(root, 0, {-WIDTH_PADDING - 1, -WIDTH_PADDING - 1, -WIDTH_PADDING - 1}, {WIDTH_PADDING + 1, WIDTH_PADDING + 1, WIDTH_PADDING + 1}, 1);
		test_cell(cells[children_index_base + 0], 0, 1, {0, 0, 0}, {WIDTH_PADDING + 1, WIDTH_PADDING + 1, WIDTH_PADDING + 1}, std::vector<std::size_t>{0}, 1);
		test_cell(cells[children_index_base + 1], 0, 2, {-WIDTH_PADDING - 1, 0, 0}, {0, WIDTH_PADDING + 1, WIDTH_PADDING + 1}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base + 2], 0, 3, {0, -WIDTH_PADDING - 1, 0}, {WIDTH_PADDING + 1, 0, WIDTH_PADDING + 1}, std::vector<std::size_t>{2}, 1);
		test_cell(cells[children_index_base + 3], 0, 4, {-WIDTH_PADDING - 1, -WIDTH_PADDING - 1, 0}, {0, 0, WIDTH_PADDING + 1}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base + 4], 0, 5, {0, 0, -WIDTH_PADDING - 1}, {WIDTH_PADDING + 1, WIDTH_PADDING + 1, 0}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base + 5], 0, 6, {-WIDTH_PADDING - 1, 0, -WIDTH_PADDING - 1}, {0, WIDTH_PADDING + 1, 0}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base + 6], 0, 7, {0, -WIDTH_PADDING - 1, -WIDTH_PADDING - 1}, {WIDTH_PADDING + 1, 0, 0}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base + 7], 0, 8, {-WIDTH_PADDING - 1, -WIDTH_PADDING - 1, -WIDTH_PADDING - 1}, {0, 0, 0}, std::vector<std::size_t>{1}, 1);
	}

	TEST_F(OctreeTest, FourPointTree) {
		const auto x = std::vector<Vector<X>>{{4, 4, 4}, {-4, -4, -4}, {2, 2, 2}, {3, 3, 3}};
		const auto s = std::size_t{1};
		const auto tree = Octree<Cell, Vector<X>>{x, s};
		const auto &cells = tree.cells();
		const auto &root = tree.cells().front();
		const auto &children_index_base = test_cell(root, 0, {-WIDTH_PADDING - 4, -WIDTH_PADDING - 4, -WIDTH_PADDING - 4}, {WIDTH_PADDING + 4, WIDTH_PADDING + 4, WIDTH_PADDING + 4}, 1);
		const auto &children_index_base2 = test_cell(cells[children_index_base + 0], 0, 1, {0, 0, 0}, {WIDTH_PADDING + 4, WIDTH_PADDING + 4, WIDTH_PADDING + 4}, 9);
		test_cell(cells[children_index_base + 1], 0, 2, {-WIDTH_PADDING - 4, 0, 0}, {0, WIDTH_PADDING + 4, WIDTH_PADDING + 4}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base + 2], 0, 3, {0, -WIDTH_PADDING - 4, 0}, {WIDTH_PADDING + 4, 0, WIDTH_PADDING + 4}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base + 3], 0, 4, {-WIDTH_PADDING - 4, -WIDTH_PADDING - 4, 0}, {0, 0, WIDTH_PADDING + 4}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base + 4], 0, 5, {0, 0, -WIDTH_PADDING - 4}, {WIDTH_PADDING + 4, WIDTH_PADDING + 4, 0}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base + 5], 0, 6, {-WIDTH_PADDING - 4, 0, -WIDTH_PADDING - 4}, {0, WIDTH_PADDING + 4, 0}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base + 6], 0, 7, {0, -WIDTH_PADDING - 4, -WIDTH_PADDING - 4}, {WIDTH_PADDING + 4, 0, 0}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base + 7], 0, 8, {-WIDTH_PADDING - 4, -WIDTH_PADDING - 4, -WIDTH_PADDING - 4}, {0, 0, 0}, std::vector<std::size_t>{1}, 1);

		test_cell(cells[children_index_base2 + 0], 1, 9, {(WIDTH_PADDING + 4) / 2, (WIDTH_PADDING + 4) / 2, (WIDTH_PADDING + 4) / 2}, {WIDTH_PADDING + 4, WIDTH_PADDING + 4, WIDTH_PADDING + 4}, std::vector<std::size_t>{0, 3}, 2);
		test_cell(cells[children_index_base2 + 1], 1, 10, {0, (WIDTH_PADDING + 4) / 2, (WIDTH_PADDING + 4) / 2}, {(WIDTH_PADDING + 4) / 2, WIDTH_PADDING + 4, WIDTH_PADDING + 4}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base2 + 2], 1, 11, {(WIDTH_PADDING + 4) / 2, 0, (WIDTH_PADDING + 4) / 2}, {WIDTH_PADDING + 4, (WIDTH_PADDING + 4) / 2, WIDTH_PADDING + 4}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base2 + 3], 1, 12, {0, 0, (WIDTH_PADDING + 4) / 2}, {(WIDTH_PADDING + 4) / 2, (WIDTH_PADDING + 4) / 2, WIDTH_PADDING + 4}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base2 + 4], 1, 13, {(WIDTH_PADDING + 4) / 2, (WIDTH_PADDING + 4) / 2, 0}, {WIDTH_PADDING + 4, WIDTH_PADDING + 4, (WIDTH_PADDING + 4) / 2}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base2 + 5], 1, 14, {0, (WIDTH_PADDING + 4) / 2, 0}, {(WIDTH_PADDING + 4) / 2, WIDTH_PADDING + 4, (WIDTH_PADDING + 4) / 2}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base2 + 6], 1, 15, {(WIDTH_PADDING + 4) / 2, 0, 0}, {WIDTH_PADDING + 4, (WIDTH_PADDING + 4) / 2, (WIDTH_PADDING + 4) / 2}, std::vector<std::size_t>{}, 0);
		test_cell(cells[children_index_base2 + 7], 1, 16, {0, 0, 0}, {(WIDTH_PADDING + 4) / 2, (WIDTH_PADDING + 4) / 2, (WIDTH_PADDING + 4) / 2}, std::vector<std::size_t>{2}, 1);
	}
} // namespace gravity

#pragma clang diagnostic pop
