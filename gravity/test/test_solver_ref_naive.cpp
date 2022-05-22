#define ENABLE_TEST
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#include <solver_ref_naive.hpp>
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

	struct SolverRefNaiveTest : ::testing::Test {
		static void runge_kutta(std::vector<Vector<X>> &x,
		                        std::vector<Vector<V>> &v,
		                        const G G_,
		                        const M m,
		                        const X epsilon,
		                        const T delta_t) {
			return SolverRefNaive<M, T, X, V, A, G>::runge_kutta(x, v, G_, m, epsilon, delta_t);
		}

		static auto acceleration(const std::vector<Vector<X>> &x,
		                         const G G_,
		                         const M m,
		                         const X epsilon,
		                         const std::size_t i) {
			return SolverRefNaive<M, T, X, V, A, G>::acceleration(x, G_, m, epsilon, i);
		}

		template <class R, class S>
		static auto euler(const R init, const S slope, const T delta_t) {
			return SolverRefNaive<M, T, X, V, A, G>::euler<R, S>(init, slope, delta_t);
		}
	};

	TEST_F(SolverRefNaiveTest, RungeKutta) {
		auto x = std::vector<Vector<X>>{{0, 0, 0}, {10, 0, 0}};
		auto v = std::vector<Vector<V>>{{0, 0, 0}, {0, 0, 0}};
		const auto G_ = G{1};
		const auto m = M{1};
		const auto epsilon = X{1} / 10000;
		const auto delta_t = T{1};
		runge_kutta(x, v, G_, m, epsilon, delta_t);
		const auto d = X{10};
		const auto d2e2 = d * d + epsilon * epsilon;
		const auto l1 = V{0};
		const auto k1 = d / std::pow(d2e2, X{3} / 2);
		const auto l2 = k1 / 2;
		const auto k2 = k1;
		const auto l3 = l2;
		const auto k3 = (d - l2) / std::pow((d - l2) * (d - l2) + epsilon * epsilon, X{3} / 2);
		const auto l4 = k3;
		const auto k4 = (d - l3 * 2) / std::pow((d - l3 * 2) * (d - l3 * 2) + epsilon * epsilon, X{3} / 2);
		const auto xdiff = (l1 + l2 * 2 + l3 * 2 + l4) / 6;
		const auto vdiff = (k1 + k2 * 2 + k3 * 2 + k4) / 6;
		EXPECT_NEAR(xdiff, x[0].x(), EPSILON<X>);
		EXPECT_NEAR(0, x[0].y(), EPSILON<X>);
		EXPECT_NEAR(0, x[0].z(), EPSILON<X>);
		EXPECT_NEAR(d - xdiff, x[1].x(), EPSILON<X>);
		EXPECT_NEAR(0, x[1].y(), EPSILON<X>);
		EXPECT_NEAR(0, x[1].z(), EPSILON<X>);
		EXPECT_NEAR(vdiff, v[0].x(), EPSILON<V>);
		EXPECT_NEAR(0, v[0].y(), EPSILON<V>);
		EXPECT_NEAR(0, v[0].z(), EPSILON<V>);
		EXPECT_NEAR(-vdiff, v[1].x(), EPSILON<V>);
		EXPECT_NEAR(0, v[1].y(), EPSILON<V>);
		EXPECT_NEAR(0, v[1].z(), EPSILON<V>);
	}

	TEST_F(SolverRefNaiveTest, Acceleration1) {
		const auto x = std::vector<Vector<X>>{{0, 0, 0}, {10, 0, 0}};
		const auto G_ = G{1};
		const auto m = M{1};
		const auto epsilon = X{1} / 10000;
		const auto a0 = acceleration(x, G_, m, epsilon, 0);
		EXPECT_NEAR(X{10} / std::pow(X{10} * 10 + epsilon * epsilon, X{3} / 2), a0.x(), EPSILON<A>);
		EXPECT_NEAR(0, a0.y(), EPSILON<A>);
		EXPECT_NEAR(0, a0.z(), EPSILON<A>);
	}

	TEST_F(SolverRefNaiveTest, Acceleration2) {
		const auto x = std::vector<Vector<X>>{{0, 0, 0}, {5, 0, 0}, {-5, 0, 0}};
		const auto G_ = G{1};
		const auto m = M{1};
		const auto epsilon = X{1} / 10000;
		const auto a0 = acceleration(x, G_, m, epsilon, 0);
		EXPECT_NEAR(0, a0.x(), EPSILON<A>);
		EXPECT_NEAR(0, a0.y(), EPSILON<A>);
		EXPECT_NEAR(0, a0.z(), EPSILON<A>);
		const auto a1 = acceleration(x, G_, m, epsilon, 1);
		EXPECT_NEAR(X{-5} / std::pow(X{5} * 5 + epsilon * epsilon, X{3} / 2) + X{-10} / std::pow(X{10} * 10 + epsilon * epsilon, X{3} / 2), a1.x(), EPSILON<A>);
		EXPECT_NEAR(0, a1.y(), EPSILON<A>);
		EXPECT_NEAR(0, a1.z(), EPSILON<A>);
	}

	TEST_F(SolverRefNaiveTest, Euler) {
		EXPECT_NEAR(0, euler(0, 0, 0), EPSILON<X>);
		EXPECT_NEAR(0, euler(0, 0, 1), EPSILON<X>);
		EXPECT_NEAR(0, euler(0, 1, 0), EPSILON<X>);
		EXPECT_NEAR(1, euler(0, 1, 1), EPSILON<X>);
		EXPECT_NEAR(2, euler(0, 1, 2), EPSILON<X>);
		EXPECT_NEAR(2, euler(1, 1, 1), EPSILON<X>);
		EXPECT_NEAR(3, euler(1, 2, 1), EPSILON<X>);
	}

	TEST_F(SolverRefNaiveTest, EulerVector) {
		const auto x0 = Vector<X>{0, 1, 2};
		const auto s0 = Vector<V>{1, 2, 3};
		const auto t = T{2};
		const auto e = euler(x0, s0, t);
		EXPECT_NEAR(2, e.x(), EPSILON<X>);
		EXPECT_NEAR(5, e.y(), EPSILON<X>);
		EXPECT_NEAR(8, e.z(), EPSILON<X>);
	}
} // namespace gravity
#pragma clang diagnostic pop
