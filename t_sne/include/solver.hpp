#ifndef T_SNE_SOLVER_HPP
#define T_SNE_SOLVER_HPP

#include <solver_ref.hpp>

namespace t_sne {
	// 任意のSolverに変更可能
	template <class Output>
	using Solver = SolverRef<Output>;
} // namespace t_sne

#endif // T_SNE_SOLVER_HPP
