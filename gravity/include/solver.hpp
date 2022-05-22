#ifndef GRAVITY_SOLVER_HPP
#define GRAVITY_SOLVER_HPP

#include <solver_ref.hpp>

namespace gravity {
	template <class M, class T, class X, class V, class A, class G>
	using Solver = SolverRef<M, T, X, V, A, G>;
}

#endif // GRAVITY_SOLVER_HPP
