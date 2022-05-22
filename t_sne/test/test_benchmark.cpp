#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#define ENABLE_TEST
#include <benchmark.hpp>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace t_sne {
	using Output = double;
	using Interpolate = float;
	using image_t = Benchmark<Output, Interpolate>::image_t;
	using embedding_t = Benchmark<Output, Interpolate>::embedding_t<>;
	using label_t = Benchmark<Output, Interpolate>::label_t;
	template <class Output>
	static constexpr Output EPSILON;
	template <>
	static constexpr auto EPSILON<double> = 1E-16;

	template struct Benchmark<float, float>;
	template struct Benchmark<float, double>;
	template struct Benchmark<double, float>;
	template struct Benchmark<double, double>;

	struct BenchmarkTest : ::testing::Test {
	  protected:
		static Output pseudoF(const std::vector<embedding_t> &y, const std::vector<label_t> &labels) {
			return Benchmark<Output, Interpolate>::pseudoF(y, labels);
		}

		static Output T(const std::vector<embedding_t> &y) noexcept {
			return Benchmark<Output, Interpolate>::T(y);
		}

		static Output Wk(const std::vector<embedding_t> &y, const std::vector<label_t> &labels, const label_t label) noexcept {
			return Benchmark<Output, Interpolate>::Wk(y, labels, label);
		}
	};

	TEST_F(BenchmarkTest, PseudoF1) {
		const auto y = std::vector<embedding_t>{{0, 1}, {3, 2}, {4, 5}, {7, 6}, {8, 9}, {11, 10}, {12, 13}, {15, 14}, {16, 17}, {19, 18}, {20, 21}, {23, 22}};
		const auto labels = std::vector<label_t>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
		EXPECT_NEAR(Output{15} / 56, pseudoF(y, labels), EPSILON<Output>);
	}

	TEST_F(BenchmarkTest, PseudoF2) {
		const auto y = std::vector<embedding_t>{{0, 1}, {3, 2}, {4, 5}, {7, 6}, {8, 9}, {11, 10}, {12, 13}, {15, 14}, {16, 17}, {19, 18}, {20, 21}, {23, 22}};
		const auto labels = std::vector<label_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
		EXPECT_NEAR(Output{4320} / 143, pseudoF(y, labels), EPSILON<Output>);
	}

	TEST_F(BenchmarkTest, PseudoF3) {
		const auto y = std::vector<embedding_t>{{0, 1}, {3, 2}, {4, 5}, {7, 6}, {8, 9}, {11, 10}, {12, 13}, {15, 14}, {16, 17}, {19, 18}, {20, 21}, {23, 22}};
		const auto labels = std::vector<label_t>{1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2};
		EXPECT_NEAR(Output{15} / 56, pseudoF(y, labels), EPSILON<Output>);
	}

	TEST_F(BenchmarkTest, PseudoF_Fail) {
		const auto y = std::vector<embedding_t>{{0, 1}};
		const auto labels = std::vector<label_t>{0};
		ASSERT_THROW(pseudoF(y, labels), std::runtime_error);
	}

	TEST_F(BenchmarkTest, T) {
		const auto y = std::vector<embedding_t>{{0, 1}, {3, 2}, {4, 5}, {7, 6}, {8, 9}, {11, 10}, {12, 13}, {15, 14}, {16, 17}, {19, 18}, {20, 21}, {23, 22}};
		EXPECT_NEAR(1150, T(y), EPSILON<Output>);
	}

	TEST_F(BenchmarkTest, Wk1) {
		const auto y = std::vector<embedding_t>{{0, 1}, {3, 2}, {4, 5}, {7, 6}, {8, 9}, {11, 10}, {12, 13}, {15, 14}, {16, 17}, {19, 18}, {20, 21}, {23, 22}};
		const auto labels = std::vector<label_t>{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
		EXPECT_NEAR(560, Wk(y, labels, 0), EPSILON<Output>);
		EXPECT_NEAR(560, Wk(y, labels, 1), EPSILON<Output>);
	}

	TEST_F(BenchmarkTest, Wk2) {
		const auto y = std::vector<embedding_t>{{0, 1}, {3, 2}, {4, 5}, {7, 6}, {8, 9}, {11, 10}, {12, 13}, {15, 14}, {16, 17}, {19, 18}, {20, 21}, {23, 22}};
		const auto labels = std::vector<label_t>{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
		EXPECT_NEAR(143, Wk(y, labels, 0), EPSILON<Output>);
		EXPECT_NEAR(143, Wk(y, labels, 1), EPSILON<Output>);
	}
} // namespace t_sne
#pragma clang diagnostic pop
