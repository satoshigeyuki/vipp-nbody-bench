#define ENABLE_TEST
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#include <problem.hpp>
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
	static constexpr auto EPSILON<double> = 1E-15;

	struct ProblemTest : ::testing::Test {
		static auto construct(const std::size_t N) {
			return Problem<M, T, X, V, A, G>{N, 10};
		}

		static auto r(const X x_0) {
			return Problem<M, T, X, V, A, G>::r(x_0);
		}

		static auto v_e(const X r) {
			return Problem<M, T, X, V, A, G>::v_e(r);
		}

		static auto P(const V q) {
			return Problem<M, T, X, V, A, G>::P(q);
		}

		template <class U>
		static auto calc_xyz(const U norm, const U r, const U theta) {
			return Problem<M, T, X, V, A, G>::calc_xyz<U>(norm, r, theta);
		}

		template <class U>
		static auto test_xyz(const U norm, const U r, const U theta, const U x, const U y, const U z) {
			const auto xyz = calc_xyz<U>(norm, r, theta);
			EXPECT_NEAR(x, xyz.x(), EPSILON<U>);
			EXPECT_NEAR(y, xyz.y(), EPSILON<U>);
			EXPECT_NEAR(z, xyz.z(), EPSILON<U>);
		}
	};

	TEST_F(ProblemTest, Construction) {
		EXPECT_THROW(construct(0), std::runtime_error);
		EXPECT_NO_THROW(construct(1));
	}

	TEST_F(ProblemTest, Epsilon) {
		const auto problem1 = Problem<M, T, X, V, A, G>{1, 10};
		EXPECT_NEAR(X{63} / 100, problem1.epsilon(), EPSILON<X>);
		const auto problem2 = Problem<M, T, X, V, A, G>{2, 10};
		EXPECT_NEAR(X{63} / 100 * std::pow(2, X{-22} / 100), problem2.epsilon(), EPSILON<X>);
	}

	TEST_F(ProblemTest, R) {
		EXPECT_NEAR(std::sqrt(X{3}) / 3, r(X{1} / 8), EPSILON<X>);
		EXPECT_NEAR(std::sqrt(X{2}) / 4, r(X{1} / 27), EPSILON<X>);
	}

	TEST_F(ProblemTest, VE) {
		EXPECT_NEAR(std::sqrt(V{2}), v_e(X{0}), EPSILON<V>);
		EXPECT_NEAR(std::pow(V{2}, V{1} / 4), v_e(X{1}), EPSILON<V>);
	}

	TEST_F(ProblemTest, P) {
		EXPECT_NEAR(0, P(V{0}), EPSILON<V>);
		EXPECT_NEAR(V{147456} / 1953125, P(V{3} / 5), EPSILON<V>);
		EXPECT_NEAR(0, P(V{1}), EPSILON<V>);
	}

	TEST_F(ProblemTest, XYZ) {
		test_xyz<X>(X{1}, X{0}, X{0}, X{1}, X{0}, X{0});
		test_xyz<X>(X{1}, X{1} / 2, X{0}, X{0}, X{1}, X{0});
		test_xyz<X>(X{1}, X{1}, X{0}, X{-1}, X{0}, X{0});
		test_xyz<X>(X{1}, X{1} / 2, X{1} / 4, X{0}, X{0}, X{1});
		test_xyz<X>(X{1}, X{1} / 2, X{1} / 2, X{0}, X{-1}, X{0});
		test_xyz<X>(X{1}, X{1} / 2, X{3} / 4, X{0}, X{0}, X{-1});
	}

	TEST_F(ProblemTest, Initialize) {
		const auto N = std::size_t{100};
		const auto problem = Problem<M, T, X, V, A, G>{N, 10};
		auto x = std::vector<Vector<X>>{};
		auto v = std::vector<Vector<V>>{};
		problem.initialize(x, v);
		EXPECT_EQ(N, x.size());
		EXPECT_EQ(N, v.size());
	}
} // namespace gravity
#pragma clang diagnostic pop
