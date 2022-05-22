#define ENABLE_TEST
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-template"
#include <benchmark.hpp>
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
	template <class U>
	static constexpr U EPSILON;
	template <>
	static constexpr auto EPSILON<double> = 1E-9;

	struct BenchmarkTest : ::testing::Test {
		static auto curve_fitting(const std::vector<X> &e_t, const std::vector<T> &t_set) {
			return Benchmark<M, T, X, V, A, G>::curve_fitting(e_t, t_set);
		}

		static auto geometric_sequence(const T a, const T r, const std::size_t n) {
			return Benchmark<M, T, X, V, A, G>::geometric_sequence(a, r, n);
		}
	};

	TEST_F(BenchmarkTest, GeometricSequenceOriginal) {
		const auto t_set = geometric_sequence(T{1}, T{2}, 5);
		const auto default_t_set = std::vector<T>{1, 2, 4, 8, 16};
		for (auto i = std::size_t{0}; i < 5; ++i) {
			EXPECT_NEAR(t_set[i], default_t_set[i], EPSILON<T>);
		}
	}

	TEST_F(BenchmarkTest, CurveFittingPerfect) {
		const auto e_t = std::vector<X>{0, 0, 0, 0, 0};
		const auto t_set = std::vector<T>{1, 2, 4, 8, 16};
		const auto [alpha, c] = curve_fitting(e_t, t_set);
		EXPECT_NEAR(0, alpha, EPSILON<X>);
	}

	TEST_F(BenchmarkTest, CurveFittingAlpha1C1) {
		const auto e_t = std::vector<X>{1, 2, 4, 8, 16};
		const auto t_set = std::vector<T>{1, 2, 4, 8, 16};
		const auto [alpha, c] = curve_fitting(e_t, t_set);
		EXPECT_NEAR(1, alpha, EPSILON<X>);
		EXPECT_NEAR(1, c, EPSILON<X>);
	}

	TEST_F(BenchmarkTest, CurveFittingAlpha2C1) {
		const auto e_t = std::vector<X>{2, 4, 8, 16, 32};
		const auto t_set = std::vector<T>{1, 2, 4, 8, 16};
		const auto [alpha, c] = curve_fitting(e_t, t_set);
		EXPECT_NEAR(2, alpha, EPSILON<X>);
		EXPECT_NEAR(1, c, EPSILON<X>);
	}

	TEST_F(BenchmarkTest, CurveFittingAlpha2C2) {
		const auto e_t = std::vector<X>{2, 8, 32, 128, 512};
		const auto t_set = std::vector<T>{1, 2, 4, 8, 16};
		const auto [alpha, c] = curve_fitting(e_t, t_set);
		EXPECT_NEAR(2, alpha, EPSILON<X>);
		EXPECT_NEAR(2, c, EPSILON<X>);
	}

	TEST_F(BenchmarkTest, CurveFittingAlpha2CM2) {
		const auto e_t = std::vector<X>{2, X{1} / 2, X{1} / 8, X{1} / 32, X{1} / 128};
		const auto t_set = std::vector<T>{1, 2, 4, 8, 16};
		const auto [alpha, c] = curve_fitting(e_t, t_set);
		EXPECT_NEAR(2, alpha, EPSILON<X>);
		EXPECT_NEAR(-2, c, EPSILON<X>);
	}
} // namespace gravity
#pragma clang diagnostic pop
