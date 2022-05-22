#include <benchmark.hpp>
#include <solver.hpp>

int main(int argc, char *argv[]) {
	// テストしたいSolverを生成し,Benchmark::benchmarkに渡す.
	// nは使用する質点の数を制御する,sは高速多重極展開法においてノードに平均的に含まれる質点の数を制御する.
	const std::size_t n = argc < 2 ? 10000 : std::stoul(argv[1]);
	const std::size_t s = argc < 3 ? 10 : std::stoul(argv[2]);
#ifdef GRAVITY_FLOAT
	using M = float;
	using T = float;
	using X = float;
	using V = float;
	using A = float;
	using G = float;
#else
#ifdef GRAVITY_TIME_DOUBLE
	using M = float;
	using T = double;
	using X = float;
	using V = float;
	using A = float;
	using G = float;
#else
	using M = double;
	using T = double;
	using X = double;
	using V = double;
	using A = double;
	using G = double;
#endif
#endif
	gravity::Solver<M, T, X, V, A, G> solver;
#ifdef ENABLE_DIFFERENT_PRECISION
	gravity::Solver<float, float, float, float, float, float> reference;
	gravity::Benchmark<M, T, X, V, A, G>().benchmark_different_precision(solver, reference, n, s);
#else
	// ソフトニングパラメータ
	const std::optional<X> epsilon = argc < 4 ? std::nullopt : std::optional<X>(std::stod(argv[3]));
	gravity::Benchmark<M, T, X, V, A, G>().benchmark(solver, n, s, epsilon);
#endif
	return 0;
}
