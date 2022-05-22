#include <limits>
#define ENABLE_TEST
#ifdef ENABLE_TEST
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#endif
#include <problem.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace h_matrix {
	using T = double;
	template <class U>
	static constexpr U EPSILON;
	template <>
	static constexpr auto EPSILON<double> = 1E-7;

	struct ProblemTest : ::testing::Test {
		auto get_d(const Problem<T> &problem) {
			return problem._d;
		}
		auto get_h(const Problem<T> &problem) {
			return problem._h;
		}

		template <class F>
		static auto midpoint_rule(const T a, const T b, const std::size_t n, const F &f) {
			return Problem<T>::midpoint_rule(a, b, n, f);
		}
	};

	TEST_F(ProblemTest, HTest) {
		const auto problem = Problem<T>{};
		auto H = Matrix<T>{problem.matrix_size(), problem.matrix_size()};
		problem.initialize_H(H);

		// 下底面-下底面
		EXPECT_NEAR(0, H(0, 0), EPSILON<T>);
		EXPECT_NEAR(0, H(0, 1), EPSILON<T>);
		// 上底面-上底面
		EXPECT_NEAR(0, H(71, 71), EPSILON<T>);
		EXPECT_NEAR(0, H(71, 70), EPSILON<T>);
		// 下底面-側面
		EXPECT_NEAR(0.24578608632104, H(0, 6), EPSILON<T>);
		EXPECT_NEAR(0.24578608632104, H(1, 7), EPSILON<T>);
		// 上底面-側面
		EXPECT_NEAR(0.24578608632104, H(66, 60), EPSILON<T>);
		EXPECT_NEAR(0.24578608632104, H(67, 61), EPSILON<T>);
		// 下底面-上底面
		EXPECT_NEAR(3.794631110077729E-4, H(0, 66), EPSILON<T>);
		EXPECT_NEAR(3.794631110077729E-4, H(1, 67), EPSILON<T>);
		// 上底面-下底面
		EXPECT_NEAR(3.794631110077729E-4, H(71, 5), EPSILON<T>);
		EXPECT_NEAR(3.794631110077729E-4, H(70, 4), EPSILON<T>);
		// 側面-下底面
		EXPECT_NEAR(0.03647034458360627, H(6, 0), EPSILON<T>);
		EXPECT_NEAR(0.03647034458360627, H(7, 1), EPSILON<T>);
		// 側面-上底面
		EXPECT_NEAR(0.03647034458360627, H(60, 66), EPSILON<T>);
		EXPECT_NEAR(0.03647034458360627, H(61, 67), EPSILON<T>);
		// 側面-側面
		EXPECT_NEAR(0.04456568168243134, H(9, 6), EPSILON<T>);
		EXPECT_NEAR(0.04456568168243134, H(6, 9), EPSILON<T>);
	}

	TEST_F(ProblemTest, MidPoint) {
		const auto a = static_cast<T>(1);
		const auto b = static_cast<T>(2);
		const auto f = [](T x) { return static_cast<T>(1) / x; };
		EXPECT_NEAR(std::log(2), midpoint_rule(a, b, 1000, f), EPSILON<T>);
	}
} // namespace h_matrix

#pragma clang diagnostic pop
