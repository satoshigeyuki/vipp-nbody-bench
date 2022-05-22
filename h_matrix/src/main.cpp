#include <benchmark.hpp>
#include <solver.hpp>

int main(int argc, char *argv[]) {
	// テストしたいSolverを生成し,Benchmark::benchmarkに渡す.
	// nは円柱の分割数を制御する.
	const std::size_t n = argc < 2 ? 6 : std::stoul(argv[1]);
	const long p = argc < 3 ? 5 : std::stol(argv[2]);
#ifdef H_MATRIX_FLOAT
	using T = float;
#else
	using T = double;
#endif
	h_matrix::Solver<T> solver(static_cast<T>(std::pow(10,-p)));
	h_matrix::Benchmark<T>().benchmark(solver, n);
	return 0;
}
