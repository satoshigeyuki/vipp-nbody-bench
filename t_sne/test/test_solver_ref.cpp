#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#define ENABLE_TEST
#include <solver_ref.hpp>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace t_sne {
	using Output = double;
	using image_t = SolverRef<Output>::image_t;
	using embedding_t = SolverRef<Output>::embedding_t;
	template <class Output>
	static constexpr Output EPSILON;
	template <>
	static constexpr auto EPSILON<double> = 1E-7;

	struct SolverRefTest : ::testing::Test {
		static VantagePointTree<image_t, SolverRef<Output>::distance_x> create_tree(const std::size_t k, const std::vector<std::size_t> &indices, const std::vector<image_t> &x) {
			auto tree = VantagePointTree<image_t, SolverRef<Output>::distance_x>{x};
			tree._k = k;
			tree._indices = indices;
			return tree;
		}
		static VantagePointTree<image_t, SolverRef<Output>::distance_x> create_tree(const std::size_t k, const std::vector<Output> &distances, const std::vector<image_t> &x) {
			auto tree = VantagePointTree<image_t, SolverRef<Output>::distance_x>{x};
			tree._k = k;
			tree._distances = distances;
			return tree;
		}

		static embedding_t gradient(const std::vector<embedding_t> &y, const std::vector<std::vector<std::tuple<std::size_t, Output>>> &p_tsne, const Output z, const std::size_t i, const BarnesHutTree<Output> &tree, const Output theta) noexcept {
			return SolverRef<Output>::gradient(y, p_tsne, z, i, tree, theta);
		}

		static embedding_t attractive(const std::vector<embedding_t> &y, const std::vector<std::tuple<std::size_t, Output>> &p_tsne_i, const Output z, const std::size_t i) noexcept {
			return SolverRef<Output>::attractive(y, p_tsne_i, z, i);
		}

		static embedding_t repulsive(const std::vector<embedding_t> &y, const Output z, const std::size_t i, const BarnesHutTree<Output> &tree, const Output theta) {
			return SolverRef<Output>::repulsive(y, z, i, tree, theta);
		}

		static Output z(const std::vector<embedding_t> &y, const BarnesHutTree<Output> &tree, const Output theta) noexcept {
			return SolverRef<Output>::z(y, tree, theta);
		}

		static constexpr Output distance_x(const image_t &x1, const image_t &x2) noexcept {
			return SolverRef<Output>::distance_x(x1, x2);
		}

		static void p_tsne(const std::size_t N, const VantagePointTree<image_t, SolverRef<Output>::distance_x> &tree, const std::vector<Output> &p_sne_values, std::vector<std::vector<std::tuple<std::size_t, Output>>> &p_tsne) {
			SolverRef<Output>::p_tsne(N, tree, p_sne_values, p_tsne);
		}

		static void p_sne(const std::size_t N, const VantagePointTree<image_t, SolverRef<Output>::distance_x> &tree, std::vector<Output> &p_sne_values, const Output u) {
			SolverRef<Output>::p_sne(N, tree, p_sne_values, u);
		}
	};

	TEST_F(SolverRefTest, Gradient_Z1_1) {
		const auto p_tsne = std::vector<std::vector<std::tuple<std::size_t, Output>>>{{{1, 1}}, {{0, 1}}};
		const auto y = std::vector<embedding_t>{{1, 1}, {0, 0}};
		const auto z = Output{1};
		const auto tree = BarnesHutTree<Output>(y);
		const auto theta = Output{1} / 2;
		const auto gradient0 = gradient(y, p_tsne, z, 0, tree, theta);
		EXPECT_NEAR(Output{8} / 9, gradient0[0], EPSILON<Output>);
		EXPECT_NEAR(Output{8} / 9, gradient0[1], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Gradient_Z1_2) {
		const auto p_tsne = std::vector<std::vector<std::tuple<std::size_t, Output>>>{{{1, 1}}, {{2, 1}}, {{1, 1}}};
		const auto y = std::vector<embedding_t>{{3, 3}, {1, 1}, {0, 0}};
		const auto z = Output{1};
		const auto tree = BarnesHutTree<Output>(y);
		const auto theta = Output{7} / 10;
		const auto gradient0 = gradient(y, p_tsne, z, 0, tree, theta);
		EXPECT_NEAR(Output{568} / 729, gradient0[0], EPSILON<Output>);
		EXPECT_NEAR(Output{568} / 729, gradient0[1], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Gradient_Z1_3) {
		const auto p_tsne = std::vector<std::vector<std::tuple<std::size_t, Output>>>{{{1, 1}, {2, 1}}, {{0, 1}, {3, 1}}, {{0, 1}, {3, 1}}, {{1, 1}, {2, 1}}, {{5, 1}, {6, 1}}, {{4, 1}, {7, 1}}, {{4, 1}, {7, 1}}, {{5, 1}, {6, 1}}, {{9, 1}, {10, 1}}, {{8, 1}, {11, 1}}, {{8, 1}, {11, 1}}, {{9, 1}, {10, 1}}, {{13, 1}, {14, 1}}, {{12, 1}, {15, 1}}, {{12, 1}, {15, 1}}, {{13, 1}, {14, 1}}};
		const auto y = std::vector<embedding_t>{{3, 3}, {2, 3}, {3, 2}, {2, 2}, {1, 3}, {0, 3}, {1, 2}, {0, 2}, {3, 1}, {2, 1}, {3, 0}, {2, 0}, {1, 1}, {0, 1}, {1, 0}, {0, 0}};
		const auto z = Output{1};
		const auto tree = BarnesHutTree<Output>(y);
		const auto theta = Output{9} / 10;
		const auto gradient0 = gradient(y, p_tsne, z, 0, tree, theta);
		EXPECT_NEAR(Output{-9427} / 18225, gradient0[0], EPSILON<Output>);
		EXPECT_NEAR(Output{-9427} / 18225, gradient0[1], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Gradient_Z2_1) {
		const auto p_tsne = std::vector<std::vector<std::tuple<std::size_t, Output>>>{{{1, 1}}, {{0, 1}}};
		const auto y = std::vector<embedding_t>{{1, 1}, {0, 0}};
		const auto z = Output{2};
		const auto tree = BarnesHutTree<Output>(y);
		const auto theta = Output{1} / 2;
		const auto gradient0 = gradient(y, p_tsne, z, 0, tree, theta);
		EXPECT_NEAR(Output{10} / 9, gradient0[0], EPSILON<Output>);
		EXPECT_NEAR(Output{10} / 9, gradient0[1], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Gradient_Z2_2) {
		const auto p_tsne = std::vector<std::vector<std::tuple<std::size_t, Output>>>{{{1, 1}}, {{2, 1}}, {{1, 1}}};
		const auto y = std::vector<embedding_t>{{3, 3}, {1, 1}, {0, 0}};
		const auto z = Output{2};
		const auto tree = BarnesHutTree<Output>(y);
		const auto theta = Output{7} / 10;
		const auto gradient0 = gradient(y, p_tsne, z, 0, tree, theta);
		EXPECT_NEAR(Output{608} / 729, gradient0[0], EPSILON<Output>);
		EXPECT_NEAR(Output{608} / 729, gradient0[1], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Gradient_Z2_3) {
		const auto p_tsne = std::vector<std::vector<std::tuple<std::size_t, Output>>>{{{1, 1}, {2, 1}}, {{0, 1}, {3, 1}}, {{0, 1}, {3, 1}}, {{1, 1}, {2, 1}}, {{5, 1}, {6, 1}}, {{4, 1}, {7, 1}}, {{4, 1}, {7, 1}}, {{5, 1}, {6, 1}}, {{9, 1}, {10, 1}}, {{8, 1}, {11, 1}}, {{8, 1}, {11, 1}}, {{9, 1}, {10, 1}}, {{13, 1}, {14, 1}}, {{12, 1}, {15, 1}}, {{12, 1}, {15, 1}}, {{13, 1}, {14, 1}}};
		const auto y = std::vector<embedding_t>{{3, 3}, {2, 3}, {3, 2}, {2, 2}, {1, 3}, {0, 3}, {1, 2}, {0, 2}, {3, 1}, {2, 1}, {3, 0}, {2, 0}, {1, 1}, {0, 1}, {1, 0}, {0, 0}};
		const auto z = Output{2};
		const auto tree = BarnesHutTree<Output>(y);
		const auto theta = Output{9} / 10;
		const auto gradient0 = gradient(y, p_tsne, z, 0, tree, theta);
		EXPECT_NEAR(Output{27023} / 36450, gradient0[0], EPSILON<Output>);
		EXPECT_NEAR(Output{27023} / 36450, gradient0[1], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Attractive0) {
		const auto y = std::vector<embedding_t>{{1, 2}, {4, 3}, {5, 6}};
		const auto p_tsne = std::vector<std::vector<std::tuple<std::size_t, Output>>>{{{1, 1}, {2, 1}}, {{0, 1}, {2, 1}}, {{0, 1}, {1, 1}}};
		const auto z = Output{1};
		const auto attractive0 = attractive(y, p_tsne[0], z, 0);
		EXPECT_NEAR(Output{-13} / 33, attractive0[0], EPSILON<Output>);
		EXPECT_NEAR(Output{-7} / 33, attractive0[1], EPSILON<Output>);
		const auto attractive1 = attractive(y, p_tsne[1], z, 1);
		EXPECT_NEAR(Output{2} / 11, attractive1[0], EPSILON<Output>);
		EXPECT_NEAR(Output{-2} / 11, attractive1[1], EPSILON<Output>);
		const auto attractive2 = attractive(y, p_tsne[2], z, 2);
		EXPECT_NEAR(Output{7} / 33, attractive2[0], EPSILON<Output>);
		EXPECT_NEAR(Output{13} / 33, attractive2[1], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Attractive1) {
		const auto y = std::vector<embedding_t>{{1, 2}, {4, 3}, {5, 6}};
		const auto p_tsne = std::vector<std::vector<std::tuple<std::size_t, Output>>>{{{1, 1}, {2, 2}}, {{0, 3}, {2, 4}}, {{0, 5}, {1, 6}}};
		const auto z = Output{2};
		const auto attractive0 = attractive(y, p_tsne[0], z, 0);
		EXPECT_NEAR(Output{-17} / 33, attractive0[0], EPSILON<Output>);
		EXPECT_NEAR(Output{-1} / 3, attractive0[1], EPSILON<Output>);
		const auto attractive1 = attractive(y, p_tsne[1], z, 1);
		EXPECT_NEAR(Output{5} / 11, attractive1[0], EPSILON<Output>);
		EXPECT_NEAR(Output{-9} / 11, attractive1[1], EPSILON<Output>);
		const auto attractive2 = attractive(y, p_tsne[2], z, 2);
		EXPECT_NEAR(Output{38} / 33, attractive2[0], EPSILON<Output>);
		EXPECT_NEAR(Output{74} / 33, attractive2[1], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Repulsive_Z1) {
		const auto y = std::vector<embedding_t>{{3, 3}, {2, 3}, {3, 2}, {2, 2}, {1, 3}, {0, 3}, {1, 2}, {0, 2}, {3, 1}, {2, 1}, {3, 0}, {2, 0}, {1, 1}, {0, 1}, {1, 0}, {0, 0}};
		const auto z = Output{1};
		const auto tree = BarnesHutTree<Output>(y);
		const auto theta = Output{9} / 10;
		const auto repulsive0 = repulsive(y, z, 0, tree, theta);
		EXPECT_NEAR(Output{45877} / 72900, repulsive0[0], EPSILON<Output>);
		EXPECT_NEAR(Output{45877} / 72900, repulsive0[1], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Repulsive_Z2) {
		const auto y = std::vector<embedding_t>{{3, 3}, {2, 3}, {3, 2}, {2, 2}, {1, 3}, {0, 3}, {1, 2}, {0, 2}, {3, 1}, {2, 1}, {3, 0}, {2, 0}, {1, 1}, {0, 1}, {1, 0}, {0, 0}};
		const auto z = Output{2};
		const auto tree = BarnesHutTree<Output>(y);
		const auto theta = Output{9} / 10;
		const auto repulsive0 = repulsive(y, z, 0, tree, theta);
		EXPECT_NEAR(Output{45877} / 145800, repulsive0[0], EPSILON<Output>);
		EXPECT_NEAR(Output{45877} / 145800, repulsive0[1], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Z1) {
		const auto y = std::vector<embedding_t>{{1, 1}, {0, 0}};
		const auto tree = BarnesHutTree<Output>(y);
		const auto theta = Output{1} / 2;
		EXPECT_NEAR(Output{2} / 3, z(y, tree, theta), EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Z2) {
		const auto y = std::vector<embedding_t>{{3, 3}, {1, 1}, {0, 0}};
		const auto tree = BarnesHutTree<Output>(y);
		const auto theta = Output{7} / 10;
		EXPECT_NEAR(Output{502} / 513, z(y, tree, theta), EPSILON<Output>);
	}

	TEST_F(SolverRefTest, Z3) {
		const auto y = std::vector<embedding_t>{{3, 3}, {2, 3}, {3, 2}, {2, 2}, {1, 3}, {0, 3}, {1, 2}, {0, 2}, {3, 1}, {2, 1}, {3, 0}, {2, 0}, {1, 1}, {0, 1}, {1, 0}, {0, 0}};
		const auto tree = BarnesHutTree<Output>(y);
		const auto theta = Output{9} / 10;
		EXPECT_NEAR(Output{145516} / 2565, z(y, tree, theta), EPSILON<Output>);
	}

	TEST_F(SolverRefTest, P_TSNE) {
		const auto N = std::size_t{5};
		const auto dummy_x = std::vector<image_t>(N);
		const auto k = std::size_t{3};
		const auto nearest_indices = std::vector<std::size_t>{1, 2, 3, 2, 3, 4, 3, 4, 0, 4, 0, 1, 0, 1, 2};
		const auto tree = create_tree(k, nearest_indices, dummy_x);
		const auto p_sne_values = std::vector<Output>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
		auto p_tsne_ = std::vector<std::vector<std::tuple<std::size_t, Output>>>(N);
		const auto p_tsne_size_answer = std::vector<std::size_t>{4, 4, 4, 4, 4};
		const auto p_tsne_indices_answer = std::vector<std::vector<std::size_t>>{{1, 2, 3, 4}, {0, 2, 3, 4}, {0, 1, 3, 4}, {0, 1, 2, 4}, {1, 2, 3, 0}};
		const auto p_tsne_values_answer = std::vector<std::vector<Output>>{
		    {
		        Output{1} / 10,
		        Output{11} / 10,
		        Output{14} / 10,
		        Output{13} / 10,
		    },
		    {
		        Output{1} / 10,
		        Output{4} / 10,
		        Output{17} / 10,
		        Output{20} / 10,
		    },
		    {
		        Output{11} / 10,
		        Output{4} / 10,
		        Output{7} / 10,
		        Output{23} / 10,
		    },
		    {
		        Output{14} / 10,
		        Output{17} / 10,
		        Output{7} / 10,
		        Output{10} / 10,
		    },
		    {
		        Output{20} / 10,
		        Output{23} / 10,
		        Output{10} / 10,
		        Output{13} / 10,
		    }};
		p_tsne(N, tree, p_sne_values, p_tsne_);
		for (auto i = decltype(N){0}; i < N; ++i) {
			EXPECT_EQ(p_tsne_size_answer[i], p_tsne_[i].size());
		}
		for (auto i = decltype(N){0}; i < N; ++i) {
			for (auto j = decltype(p_tsne_size_answer[0]){0}; j < p_tsne_size_answer[i]; ++j) {
				EXPECT_EQ(p_tsne_indices_answer[i][j], std::get<0>(p_tsne_[i][j]));
				EXPECT_NEAR(p_tsne_values_answer[i][j], std::get<1>(p_tsne_[i][j]), EPSILON<Output>);
			}
		}
	}

	TEST_F(SolverRefTest, P_SNE_Calculation1) {
		const auto N = std::size_t{3};
		const auto dummy_x = std::vector<image_t>(N);
		const auto nearest_distances = std::vector<Output>{1, 2, 1, 3, 2, 3};
		auto p_sne_values = std::vector<Output>(6, 0);
		const auto k = std::size_t{2};
		const auto tree = create_tree(k, nearest_distances, dummy_x);
		const auto beta = Output{1} / 2;
		const auto distance2_0 = nearest_distances[0] * nearest_distances[0];
		const auto distance2_1 = nearest_distances[1] * nearest_distances[1];
		const auto numerator0 = std::exp(-beta * distance2_0);
		const auto numerator1 = std::exp(-beta * distance2_1);
		const auto divisor0 = numerator0 + numerator1;
		const auto p_0 = numerator0 / divisor0;
		const auto p_1 = numerator1 / divisor0;
		const auto u0 = std::pow(Output{2}, -p_0 * std::log2(p_0) - p_1 * std::log2(p_1));
		p_sne(N, tree, p_sne_values, u0);
		EXPECT_NEAR(numerator0 / divisor0, p_sne_values[0], EPSILON<Output>);
		EXPECT_NEAR(numerator1 / divisor0, p_sne_values[1], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, P_SNE_Calculation2) {
		const auto N = std::size_t{3};
		const auto dummy_x = std::vector<image_t>(N);
		const auto nearest_distances = std::vector<Output>{1, 2, 1, 3, 2, 3};
		auto p_sne_values = std::vector<Output>(6, 0);
		const auto k = std::size_t{2};
		const auto tree = create_tree(k, nearest_distances, dummy_x);
		const auto beta = Output{1} / 10;
		const auto distance2_0 = nearest_distances[2] * nearest_distances[2];
		const auto distance2_1 = nearest_distances[3] * nearest_distances[3];
		const auto numerator0 = std::exp(-beta * distance2_0);
		const auto numerator1 = std::exp(-beta * distance2_1);
		const auto divisor0 = numerator0 + numerator1;
		const auto p_0 = numerator0 / divisor0;
		const auto p_1 = numerator1 / divisor0;
		const auto u0 = std::pow(Output{2}, -p_0 * std::log2(p_0) - p_1 * std::log2(p_1));
		p_sne(N, tree, p_sne_values, u0);
		EXPECT_NEAR(numerator0 / divisor0, p_sne_values[2], EPSILON<Output>);
		EXPECT_NEAR(numerator1 / divisor0, p_sne_values[3], EPSILON<Output>);
	}

	TEST_F(SolverRefTest, P_SNE_Calculation3) {
		const auto N = std::size_t{3};
		const auto dummy_x = std::vector<image_t>(N);
		const auto nearest_distances = std::vector<Output>{1, 2, 1, 3, 2, 3};
		auto p_sne_values = std::vector<Output>(6, 0);
		const auto k = std::size_t{2};
		const auto tree = create_tree(k, nearest_distances, dummy_x);
		const auto beta = Output{1} / 10;
		const auto distance2_0 = nearest_distances[4] * nearest_distances[4];
		const auto distance2_1 = nearest_distances[5] * nearest_distances[5];
		const auto numerator0 = std::exp(-beta * distance2_0);
		const auto numerator1 = std::exp(-beta * distance2_1);
		const auto divisor0 = numerator0 + numerator1;
		const auto p_0 = numerator0 / divisor0;
		const auto p_1 = numerator1 / divisor0;
		const auto u0 = std::pow(Output{2}, -p_0 * std::log2(p_0) - p_1 * std::log2(p_1));
		p_sne(N, tree, p_sne_values, u0);
		EXPECT_NEAR(numerator0 / divisor0, p_sne_values[4], EPSILON<Output>);
		EXPECT_NEAR(numerator1 / divisor0, p_sne_values[5], EPSILON<Output>);
	}

} // namespace t_sne
#pragma clang diagnostic pop
