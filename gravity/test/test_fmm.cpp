#define ENABLE_TEST
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#include <solver_ref.hpp>
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

	struct FMMTest : ::testing::Test {
		using Cell = Cell<M, T, X, V, A, G>;

		static constexpr auto WIDTH_PADDING = Cell::WIDTH_PADDING;

		static void calc_link(const Cell &cell, const std::vector<Cell> &cells, std::vector<std::vector<std::size_t>> &approx_interact_indices_list, std::vector<std::vector<std::size_t>> &neighbor_interact_indices_list, std::vector<std::vector<std::size_t>> &neighbor_indices_list, const std::size_t level) {
			SolverRef<M, T, X, V, A, G>::calc_link(cell, cells, approx_interact_indices_list, neighbor_interact_indices_list, neighbor_indices_list, level);
		}

		static void calcM(Octree<Cell, Vector<X>> &tree, const unsigned p, const M m, const std::vector<Vector<X>> &x) {
			auto &cells = tree.cells();
			tree.apply(Octree<Cell, Vector<X>>::fix([&cells, &p, &m, &x](const auto &bottom_up_f, Cell &cell, const std::size_t level) -> void {
				           const auto &content = cell.content();
				           if (content.index() == Cell::INDEX_POINTS) {
					           if (level >= 2) {
						           p2m(cell, p, x, m);
					           }
				           } else {
					           const auto child_index_base = std::get<Cell::INDEX_CELLS>(content);
#pragma omp parallel for
					           for (auto child_index = child_index_base; child_index < child_index_base + 8; ++child_index) {
						           bottom_up_f(bottom_up_f, cells[child_index], level + 1);
					           }
					           if (level >= 2) {
						           m2m(cell, p, cells);
					           }
				           }
			           }),
			           std::size_t{0});
		}

		static void calcL(Octree<Cell, Vector<X>> &tree, const unsigned p) {
			auto &cells = tree.cells();
			std::vector<std::vector<std::size_t>> approx_interact_indices_list{tree.cells().size()};
			std::vector<std::vector<std::size_t>> neighbor_interact_indices_list{cells.size()};
			std::vector<std::vector<std::size_t>> neighbor_indices_list{tree.cells().size()};
			tree.apply(calc_link, cells, approx_interact_indices_list, neighbor_interact_indices_list, neighbor_indices_list, std::size_t{0});
#pragma omp parallel for
			for (auto index = decltype(cells.size()){0}; index < cells.size(); ++index) {
				auto &cell = cells[index];
				if (cell.index() > 8) {
					m2l(cell, p, cells, approx_interact_indices_list[index]);
				}
			}
			tree.apply(Octree<Cell, Vector<X>>::fix([&cells, &p](const auto &top_down_f, Cell &cell, const std::size_t level) -> void {
				           const auto &content = cell.content();
				           if (content.index() == Cell::INDEX_CELLS) {
					           if (level >= 3) {
						           l2l(cell, p, cells);
					           }
					           const auto child_index_base = std::get<Cell::INDEX_CELLS>(content);
#pragma omp parallel for
					           for (auto child_index = child_index_base; child_index < child_index_base + 8; ++child_index) {
						           top_down_f(top_down_f, cells[child_index], level + 1);
					           }
				           }
			           }),
			           std::size_t{0});
		}

		static auto getM(const Cell &cell, const unsigned p, const unsigned k, const signed m) {
			return cell.multipole(p, k, m);
		}

		static auto getL(const Cell &cell, const unsigned p, const unsigned j, const signed k) {
			return cell.local(p, j, k);
		}

		template <class U>
		static auto test_complex(const std::complex<U> &required, const std::complex<U> &calculated) {
			EXPECT_NEAR(required.real(), calculated.real(), EPSILON<U>);
			EXPECT_NEAR(required.imag(), calculated.imag(), EPSILON<U>);
		}

		static void divide(Octree<Cell, Vector<X>> &tree, const std::size_t index, const std::size_t threshold) {
			tree.subdivide(index, threshold);
		}
	};

	TEST_F(FMMTest, ManyPoints) {
		const auto problem = Problem<M, T, X, V, A, G>{1000, 10};
		auto x = std::vector<Vector<X>>{};
		auto v = std::vector<Vector<V>>{};
		problem.initialize(x, v);
		const auto s = std::size_t{5};
		const auto p = unsigned{5};
		const auto m_i = M{1} / x.size();
		auto tree = Octree<Cell, Vector<X>>{x, s};
		calcM(tree, p, m_i, x);
		calcL(tree, p);
		const auto &cells = tree.cells();
		for (const auto &cell : cells) {
			if (cell.index() <= 8) {
				continue;
			}
			for (auto k = decltype(p){0}; k <= p; ++k) {
				for (auto m = -static_cast<signed>(k); m <= static_cast<signed>(k); ++m) {
					auto m_value = getM(cell, p, k, m);
					auto l_value = getL(cell, p, k, m);
					EXPECT_FALSE(std::isnan(m_value.real()));
					EXPECT_FALSE(std::isnan(m_value.imag()));
					EXPECT_FALSE(std::isnan(l_value.real()));
					EXPECT_FALSE(std::isnan(l_value.imag()));
				}
			}
		}
	}

	TEST_F(FMMTest, PM1) {
		const auto p = P<X>{5, 5, -1};
		EXPECT_NEAR(1, p(0, 0), EPSILON<X>);
		EXPECT_NEAR(-1, p(1, 0), EPSILON<X>);
		EXPECT_NEAR(1, p(2, 0), EPSILON<X>);
		EXPECT_NEAR(-1, p(3, 0), EPSILON<X>);
		EXPECT_NEAR(1, p(4, 0), EPSILON<X>);
		EXPECT_NEAR(-1, p(5, 0), EPSILON<X>);

		EXPECT_NEAR(0, p(1, 1), EPSILON<X>);
		EXPECT_NEAR(0, p(2, 1), EPSILON<X>);
		EXPECT_NEAR(0, p(3, 1), EPSILON<X>);
		EXPECT_NEAR(0, p(4, 1), EPSILON<X>);
		EXPECT_NEAR(0, p(5, 1), EPSILON<X>);

		EXPECT_NEAR(0, p(2, 2), EPSILON<X>);
		EXPECT_NEAR(0, p(3, 2), EPSILON<X>);
		EXPECT_NEAR(0, p(4, 2), EPSILON<X>);
		EXPECT_NEAR(0, p(5, 2), EPSILON<X>);

		EXPECT_NEAR(0, p(3, 3), EPSILON<X>);
		EXPECT_NEAR(0, p(4, 3), EPSILON<X>);
		EXPECT_NEAR(0, p(5, 3), EPSILON<X>);

		EXPECT_NEAR(0, p(4, 4), EPSILON<X>);
		EXPECT_NEAR(0, p(5, 4), EPSILON<X>);

		EXPECT_NEAR(0, p(5, 5), EPSILON<X>);
	}

	TEST_F(FMMTest, P0) {
		const auto p = P<X>{5, 5, 0};
		EXPECT_NEAR(1, p(0, 0), EPSILON<X>);
		EXPECT_NEAR(0, p(1, 0), EPSILON<X>);
		EXPECT_NEAR(X{-1} / 2, p(2, 0), EPSILON<X>);
		EXPECT_NEAR(0, p(3, 0), EPSILON<X>);
		EXPECT_NEAR(X{3} / 8, p(4, 0), EPSILON<X>);
		EXPECT_NEAR(0, p(5, 0), EPSILON<X>);

		EXPECT_NEAR(-1, p(1, 1), EPSILON<X>);
		EXPECT_NEAR(0, p(2, 1), EPSILON<X>);
		EXPECT_NEAR(X{3} / 2, p(3, 1), EPSILON<X>);
		EXPECT_NEAR(0, p(4, 1), EPSILON<X>);
		EXPECT_NEAR(X{-15} / 8, p(5, 1), EPSILON<X>);

		EXPECT_NEAR(3, p(2, 2), EPSILON<X>);
		EXPECT_NEAR(0, p(3, 2), EPSILON<X>);
		EXPECT_NEAR(X{-15} / 2, p(4, 2), EPSILON<X>);
		EXPECT_NEAR(0, p(5, 2), EPSILON<X>);

		EXPECT_NEAR(-15, p(3, 3), EPSILON<X>);
		EXPECT_NEAR(0, p(4, 3), EPSILON<X>);
		EXPECT_NEAR(X{105} / 2, p(5, 3), EPSILON<X>);

		EXPECT_NEAR(105, p(4, 4), EPSILON<X>);
		EXPECT_NEAR(0, p(5, 4), EPSILON<X>);

		EXPECT_NEAR(-945, p(5, 5), EPSILON<X>);
	}

	TEST_F(FMMTest, P1) {
		const auto p = P<X>{5, 5, 1};
		EXPECT_NEAR(1, p(0, 0), EPSILON<X>);
		EXPECT_NEAR(1, p(1, 0), EPSILON<X>);
		EXPECT_NEAR(1, p(2, 0), EPSILON<X>);
		EXPECT_NEAR(1, p(3, 0), EPSILON<X>);
		EXPECT_NEAR(1, p(4, 0), EPSILON<X>);
		EXPECT_NEAR(1, p(5, 0), EPSILON<X>);

		EXPECT_NEAR(0, p(1, 1), EPSILON<X>);
		EXPECT_NEAR(0, p(2, 1), EPSILON<X>);
		EXPECT_NEAR(0, p(3, 1), EPSILON<X>);
		EXPECT_NEAR(0, p(4, 1), EPSILON<X>);
		EXPECT_NEAR(0, p(5, 1), EPSILON<X>);

		EXPECT_NEAR(0, p(2, 2), EPSILON<X>);
		EXPECT_NEAR(0, p(3, 2), EPSILON<X>);
		EXPECT_NEAR(0, p(4, 2), EPSILON<X>);
		EXPECT_NEAR(0, p(5, 2), EPSILON<X>);

		EXPECT_NEAR(0, p(3, 3), EPSILON<X>);
		EXPECT_NEAR(0, p(4, 3), EPSILON<X>);
		EXPECT_NEAR(0, p(5, 3), EPSILON<X>);

		EXPECT_NEAR(0, p(4, 4), EPSILON<X>);
		EXPECT_NEAR(0, p(5, 4), EPSILON<X>);

		EXPECT_NEAR(0, p(5, 5), EPSILON<X>);
	}

	TEST_F(FMMTest, Y0) {
		const auto y = Y<X>{5, 5, 0, 0};
		for (auto k = unsigned{0}; k < 5; ++k) {
			for (auto m = -static_cast<signed>(k); m < static_cast<signed>(k); ++m) {
				if (m == 0) {
					test_complex({1, 0}, y(k, m));
				} else {
					test_complex({0, 0}, y(k, m));
				}
			}
		}
	}

	TEST_F(FMMTest, Y_Q_PI) {
		const auto theta = Constant<X>::PI / 4;
		const auto y0 = Y<X>{5, 5, theta, 0};
		test_complex({1, 0}, y0(0, 0));

		test_complex({-X{1} / 2, 0}, y0(1, -1));
		test_complex({std::sqrt(X{2}) / 2, 0}, y0(1, 0));
		test_complex({-X{1} / 2, 0}, y0(1, 1));

		test_complex({X{3} / 2 / std::sqrt(24), 0}, y0(2, -2));
		test_complex({X{-3} / 2 / std::sqrt(6), 0}, y0(2, -1));
		test_complex({X{1} / 4, 0}, y0(2, 0));
		test_complex({X{-3} / 2 / std::sqrt(6), 0}, y0(2, 1));
		test_complex({X{3} / 2 / std::sqrt(24), 0}, y0(2, 2));

		test_complex({std::sqrt(X{2}) * -15 / 4 / std::sqrt(720), 0}, y0(3, -3));
		test_complex({std::sqrt(X{2}) * 15 / 4 / std::sqrt(120), 0}, y0(3, -2));
		test_complex({std::sqrt(X{2}) * -9 / 8 / std::sqrt(12), 0}, y0(3, -1));
		test_complex({std::sqrt(X{2}) / -8, 0}, y0(3, 0));
		test_complex({std::sqrt(X{2}) * -9 / 8 / std::sqrt(12), 0}, y0(3, 1));
		test_complex({std::sqrt(X{2}) * 15 / 4 / std::sqrt(120), 0}, y0(3, 2));
		test_complex({std::sqrt(X{2}) * -15 / 4 / std::sqrt(720), 0}, y0(3, 3));

		test_complex({X{105} / 4 / std::sqrt(40320), 0}, y0(4, -4));
		test_complex({X{-105} / 4 / std::sqrt(5040), 0}, y0(4, -3));
		test_complex({X{75} / 8 / std::sqrt(360), 0}, y0(4, -2));
		test_complex({X{-5} / 8 / std::sqrt(20), 0}, y0(4, -1));
		test_complex({X{-13} / 32, 0}, y0(4, 0));
		test_complex({X{-5} / 8 / std::sqrt(20), 0}, y0(4, 1));
		test_complex({X{75} / 8 / std::sqrt(360), 0}, y0(4, 2));
		test_complex({X{-105} / 4 / std::sqrt(5040), 0}, y0(4, 3));
		test_complex({X{105} / 4 / std::sqrt(40320), 0}, y0(4, 4));

		test_complex({std::sqrt(X{2}) * -945 / 8 / std::sqrt(3628800), 0}, y0(5, -5));
		test_complex({std::sqrt(X{2}) * 945 / 8 / std::sqrt(362880), 0}, y0(5, -4));
		test_complex({std::sqrt(X{2}) * -735 / 16 / std::sqrt(20160), 0}, y0(5, -3));
		test_complex({std::sqrt(X{2}) * 105 / 16 / std::sqrt(840), 0}, y0(5, -2));
		test_complex({std::sqrt(X{2}) * 45 / 64 / std::sqrt(30), 0}, y0(5, -1));
		test_complex({std::sqrt(X{2}) * -17 / 64, 0}, y0(5, 0));
		test_complex({std::sqrt(X{2}) * 45 / 64 / std::sqrt(30), 0}, y0(5, 1));
		test_complex({std::sqrt(X{2}) * 105 / 16 / std::sqrt(840), 0}, y0(5, 2));
		test_complex({std::sqrt(X{2}) * -735 / 16 / std::sqrt(20160), 0}, y0(5, 3));
		test_complex({std::sqrt(X{2}) * 945 / 8 / std::sqrt(362880), 0}, y0(5, 4));
		test_complex({std::sqrt(X{2}) * -945 / 8 / std::sqrt(3628800), 0}, y0(5, 5));
	}

	TEST_F(FMMTest, Y_H_PI) {
		const auto theta = Constant<X>::PI / 2;
		const auto y0 = Y<X>{5, 5, theta, 0};
		test_complex({1, 0}, y0(0, 0));

		test_complex({X{-1} / std::sqrt(2), 0}, y0(1, -1));
		test_complex({0, 0}, y0(1, 0));
		test_complex({X{-1} / std::sqrt(2), 0}, y0(1, 1));

		test_complex({X{3} / std::sqrt(24), 0}, y0(2, -2));
		test_complex({0, 0}, y0(2, -1));
		test_complex({X{-1} / 2, 0}, y0(2, 0));
		test_complex({0, 0}, y0(2, 1));
		test_complex({X{3} / std::sqrt(24), 0}, y0(2, 2));

		test_complex({X{-15} / std::sqrt(720), 0}, y0(3, -3));
		test_complex({0, 0}, y0(3, -2));
		test_complex({X{3} / 2 / std::sqrt(12), 0}, y0(3, -1));
		test_complex({0, 0}, y0(3, 0));
		test_complex({X{3} / 2 / std::sqrt(12), 0}, y0(3, 1));
		test_complex({0, 0}, y0(3, 2));
		test_complex({X{-15} / std::sqrt(720), 0}, y0(3, 3));

		test_complex({X{105} / std::sqrt(40320), 0}, y0(4, -4));
		test_complex({0, 0}, y0(4, -3));
		test_complex({X{-15} / 2 / std::sqrt(360), 0}, y0(4, -2));
		test_complex({0, 0}, y0(4, -1));
		test_complex({X{3} / 8, 0}, y0(4, 0));
		test_complex({0, 0}, y0(4, 1));
		test_complex({X{-15} / 2 / std::sqrt(360), 0}, y0(4, 2));
		test_complex({0, 0}, y0(4, 3));
		test_complex({X{105} / std::sqrt(40320), 0}, y0(4, 4));

		test_complex({-945 / std::sqrt(3628800), 0}, y0(5, -5));
		test_complex({0, 0}, y0(5, -4));
		test_complex({X{105} / 2 / std::sqrt(20160), 0}, y0(5, -3));
		test_complex({0, 0}, y0(5, -2));
		test_complex({X{-15} / 8 / std::sqrt(30), 0}, y0(5, -1));
		test_complex({0, 0}, y0(5, 0));
		test_complex({X{-15} / 8 / std::sqrt(30), 0}, y0(5, 1));
		test_complex({0, 0}, y0(5, 2));
		test_complex({X{105} / 2 / std::sqrt(20160), 0}, y0(5, 3));
		test_complex({0, 0}, y0(5, 4));
		test_complex({-945 / std::sqrt(3628800), 0}, y0(5, 5));
	}

	TEST_F(FMMTest, M) {
		const auto x = std::vector<Vector<X>>{{1, 1, 1}};
		const auto s = std::size_t{1};
		const auto p = unsigned{2};
		const auto m = M{1} / x.size();
		auto tree = Octree<Cell, Vector<X>>{x, s};
		auto &root = tree.cells().front();
		p2m(root, p, x, m);

		test_complex({1, 0}, getM(root, p, 0, 0));
		test_complex({0, 0}, getM(root, p, 1, 0));
		test_complex({0, 0}, getM(root, p, 1, 1));
	}

	TEST_F(FMMTest, M2) {
		const auto x = std::vector<Vector<X>>{{1, 0, 0}};
		const auto s = std::size_t{1};
		const auto p = unsigned{2};
		const auto m = M{1} / x.size();
		auto tree = Octree<Cell, Vector<X>>{x, s};
		divide(tree, 0, 2);
		auto &root = tree.cells().front();
		for (const auto child_index : root.cells()) {
			p2m(tree.cells()[child_index], p, x, m);
		}
		m2m(root, p, tree.cells());

		test_complex({1, 0}, getM(root, p, 0, 0));
		test_complex({0, 0}, getM(root, p, 1, 0));
	}

	TEST_F(FMMTest, L) {
		const auto x = std::vector<Vector<X>>{{1, 1, 1}, {1, -1, -1}, {-1, -1, -1}};
		const auto s = std::size_t{1};
		const auto p = unsigned{2};
		const auto m = M{1} / x.size();
		auto tree = Octree<Cell, Vector<X>>{x, s};
		const auto &cells = tree.cells();
		divide(tree, 1, 2);
		divide(tree, 8, 2);
		calcM(tree, p, m, x);
		calcL(tree, p);
		const auto l_1_1 = getL(cells[9], p, 1, 1);
		using U = typename decltype(l_1_1)::value_type;
		const auto e = static_cast<U>(WIDTH_PADDING);
		const auto rho = (1 + e) * 3 / 2 * std::sqrt(3);
		const auto sq2 = std::sqrt(U{2});
		const auto sq3 = std::sqrt(U{3});
		const auto sq5 = std::sqrt(U{5});
		const auto sq6 = std::sqrt(U{6});
		const auto required =
		    std::complex<U>{U{1} / 3, 0} * (-1 / sq2) * std::complex<U>{1 / sq6, -1 / sq6} / (-sq2 / 2 * rho * rho) +
		    std::complex<U>{sq2 / 24 - sq2 / 8 * e, sq2 / 24 - sq2 / 8 * e} * (U{1} / 2) * std::complex<U>{0, -1 / sq6} / (-sq6 / 12 * rho * rho * rho) +
		    std::complex<U>{U{1} / 12 - U{1} / 4 * e, 0} * (-sq2 / 2) * std::complex<U>{-1 / sq6, 1 / sq6} / (-sq6 / 6 * rho * rho * rho) +
		    std::complex<U>{sq2 / 24 - sq2 / 8 * e, -sq2 / 24 + sq2 / 8 * e} * (U{-1} / 2) * std::complex<U>{0, 0} / (U{-1} / 2 * rho * rho * rho) +
		    std::complex<U>{0, sq6 / 96 - sq6 / 16 * e + 3 * sq6 / 32 * e * e} * (-sq3 / 12) * std::complex<U>{-sq3 * sq5 / 18, -sq3 * sq5 / 18} / (U{-1} / 12 / sq5 * rho * rho * rho * rho) +
		    std::complex<U>{-sq6 / 96 + sq6 / 16 * e - 3 * sq6 / 32 * e * e, -sq6 / 96 + sq6 / 16 * e - 3 * sq6 / 32 * e * e} * (-sq3 / 6) * std::complex<U>{0, sq2 * sq5 / 6} / (U{-1} / 2 / sq2 / sq3 / sq5 * rho * rho * rho * rho) +
		    std::complex<U>{0, 0} * (-sq2 / 4) * std::complex<U>{U{1} / 6, U{-1} / 6} / (-sq3 / 12 * rho * rho * rho * rho) +
		    std::complex<U>{-sq6 / 96 + sq6 / 16 * e - 3 * sq6 / 32 * e * e, sq6 / 96 - sq6 / 16 * e + 3 * sq6 / 32 * e * e} * (sq3 / 6) * std::complex<U>{2 * sq3 / 9, 0} / (U{-1} / 6 * rho * rho * rho * rho) +
		    std::complex<U>{0, -sq6 / 96 + sq6 / 16 * e - 3 * sq6 / 32 * e * e} * (sq3 / 12) * std::complex<U>{U{1} / 6, U{1} / 6} / (-sq3 / 12 * rho * rho * rho * rho);
		test_complex(required, l_1_1);
	}

} // namespace gravity

#pragma clang diagnostic pop
