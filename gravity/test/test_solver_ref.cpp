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

	struct SolverRefTest : ::testing::Test {
		static void runge_kutta(std::vector<Vector<X>> &x,
		                        std::vector<Vector<V>> &v,
		                        const G G_,
		                        const M m,
		                        const X epsilon,
		                        const T delta_t,
		                        const std::size_t s,
		                        const unsigned p) {
			return SolverRef<M, T, X, V, A, G>::runge_kutta(x, v, G_, m, epsilon, delta_t, s, p);
		}
	};

	TEST_F(SolverRefTest, OnlyRun) {
		auto problem = Problem<M, T, X, V, A, G>{1000, 10};
		auto x = std::vector<Vector<X>>{};
		auto v = std::vector<Vector<V>>{};
		problem.initialize(x, v);
		const auto G_ = G{1};
		const auto m = M{1};
		const auto epsilon = X{1} / 10000;
		const auto delta_t = T{1};
		const auto s = std::size_t{1};
		const auto p = unsigned{4};
		runge_kutta(x, v, G_, m, epsilon, delta_t, s, p);
	}

} // namespace gravity
#pragma clang diagnostic pop
