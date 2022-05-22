#include <benchmark.hpp>
#include <solver.hpp>

int main(int argc, char *argv[]) {
	// テストしたいSolverを生成し,Benchmark::benchmarkに渡す.
	// nは使用される画像の枚数を制御する(n*70000枚)
	const std::size_t n = argc < 2 ? 1 : std::stoul(argv[1]);
	const std::size_t t = argc < 3 ? 1000 : std::stoul(argv[2]);
	const double a = argc < 4 ? 2.0 : std::stod(argv[3]);
#ifdef TSNE_FLOAT
	using Output = float;
#else
	using Output = double;
#endif
	t_sne::Solver<Output> solver;
	t_sne::Benchmark<Output>().benchmark(solver, n, t, a);
	return 0;
}
