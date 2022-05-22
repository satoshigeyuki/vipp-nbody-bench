#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#define ENABLE_TEST
#include <solver_ref_naive.hpp>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace t_sne {
	using Output = double;
	using image_t = SolverRefNaive<Output>::image_t;
	using embedding_t = SolverRefNaive<Output>::embedding_t;
	constexpr auto IMAGE_SIZE = Problem<Output>::IMAGE_SIZE;
	constexpr auto P_MIN = Problem<Output>::P_MIN;
	constexpr auto P_MAX = Problem<Output>::P_MAX;
	template <class Output>
	static constexpr Output EPSILON;
	template <>
	static constexpr auto EPSILON<double> = 1E-15;
	template <class Output>
	static constexpr Output EPSILON_BETA;
	template <>
	static constexpr auto EPSILON_BETA<double> = 1E-6;

	struct SolverRefNaiveTest : ::testing::Test {
	  protected:
		static void descent(const std::vector<image_t> &x, std::vector<embedding_t> &y, std::vector<embedding_t> &dy, const std::vector<Output> &betas, const std::vector<Output> &p_sne_divisors, const Output eta, const Output alpha) noexcept {
			return SolverRefNaive<Output>::descent(x, y, dy, betas, p_sne_divisors, eta, alpha);
		}

		static embedding_t gradient(const std::vector<image_t> &x, const std::vector<embedding_t> &y, const std::vector<Output> &betas, const std::vector<Output> &p_sne_divisors, const Output z, const std::size_t i) noexcept {
			return SolverRefNaive<Output>::gradient(x, y, betas, p_sne_divisors, z, i);
		}

		static Output z(const std::vector<embedding_t> &y) noexcept {
			return SolverRefNaive<Output>::z(y);
		}

		static Output p_tsne(const std::vector<image_t> &x, const std::size_t i, const std::size_t j, const Output beta, const Output divisor_i, const Output divisor_j) noexcept {
			return SolverRefNaive<Output>::p_tsne(x, i, j, beta, divisor_i, divisor_j);
		}

		static Output p_sne(const std::vector<image_t> &x, const std::size_t i, const std::size_t j, const Output beta, const Output divisor) noexcept {
			return SolverRefNaive<Output>::p_sne(x, i, j, beta, divisor);
		}

		static Output p_sne_divisor(const std::vector<image_t> &x, const std::size_t i, const Output beta) noexcept {
			return SolverRefNaive<Output>::p_sne_divisor(x, i, beta);
		}

		static Output q_tsne(const std::vector<embedding_t> &y, const std::size_t i, const std::size_t j, const Output z) noexcept {
			return SolverRefNaive<Output>::q_tsne(y, i, j, z);
		}

		static constexpr Output distance_x2(const image_t &x1, const image_t &x2) noexcept {
			return SolverRefNaive<Output>::distance_x2(x1, x2);
		}

		static constexpr Output distance_y2(const embedding_t &y1, const embedding_t &y2) noexcept {
			return SolverRefNaive<Output>::distance_y2(y1, y2);
		}

		static void beta(const std::vector<image_t> &x, std::vector<Output> &betas, const Output u) {
			SolverRefNaive<Output>::beta(x, betas, u);
		}

		static Output bisection_p(const std::size_t N, const Output u, const Output p_min, const Output p_max) {
			return SolverRefNaive<Output>::bisection_p(N, u, p_min, p_max);
		}
	};

	TEST_F(SolverRefNaiveTest, Descent_ETA_IS_1_ALPHA_IS_1) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		auto y0 = std::vector<embedding_t>{{1, 2}, {4, 3}, {5, 6}};
		auto dy0 = std::vector<embedding_t>{{0, 0}, {0, 0}, {0, 0}};
		const auto betas = std::vector<Output>{Output{1} / 2, Output{1} / 2, Output{1} / 2};
		const auto p_sne_divisors = std::vector<Output>{1, 1, 1};
		constexpr auto eta0 = Output{1};
		constexpr auto alpha0 = Output{1};
		const auto e_half = std::exp(Output{-1} / 2 / 65025);
		const auto e2 = std::exp(Output{-2} / 65025);

		descent(x0, y0, dy0, betas, p_sne_divisors, eta0, alpha0);

		EXPECT_NEAR(-e_half * 4 / 11 - e2 * 16 / 99 + Output{62} / 231, dy0[0][0], EPSILON<Output>);
		EXPECT_NEAR(-e_half * 4 / 33 - e2 * 16 / 99 + Output{26} / 231, dy0[0][1], EPSILON<Output>);
		EXPECT_NEAR(e_half * 8 / 33 - Output{12} / 77, dy0[1][0], EPSILON<Output>);
		EXPECT_NEAR(-e_half * 8 / 33 + Output{12} / 77, dy0[1][1], EPSILON<Output>);
		EXPECT_NEAR(e_half * 4 / 33 + e2 * 16 / 99 - Output{26} / 231, dy0[2][0], EPSILON<Output>);
		EXPECT_NEAR(e_half * 4 / 11 + e2 * 16 / 99 - Output{62} / 231, dy0[2][1], EPSILON<Output>);

		EXPECT_NEAR(-e_half * 4 / 11 - e2 * 16 / 99 + Output{62} / 231 + Output{1}, y0[0][0], EPSILON<Output>);
		EXPECT_NEAR(-e_half * 4 / 33 - e2 * 16 / 99 + Output{26} / 231 + Output{2}, y0[0][1], EPSILON<Output>);
		EXPECT_NEAR(e_half * 8 / 33 - Output{12} / 77 + Output{4}, y0[1][0], EPSILON<Output>);
		EXPECT_NEAR(-e_half * 8 / 33 + Output{12} / 77 + Output{3}, y0[1][1], EPSILON<Output>);
		EXPECT_NEAR(e_half * 4 / 33 + e2 * 16 / 99 - Output{26} / 231 + Output{5}, y0[2][0], EPSILON<Output>);
		EXPECT_NEAR(e_half * 4 / 11 + e2 * 16 / 99 - Output{62} / 231 + Output{6}, y0[2][1], EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, Descent_ETA_IS_2_ALPHA_IS_HALF) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		auto y1 = std::vector<embedding_t>{{1, 2}, {4, 3}, {5, 6}};
		auto dy1 = std::vector<embedding_t>{{1, 2}, {3, 4}, {5, 6}};
		const auto betas = std::vector<Output>{Output{1} / 2, Output{1} / 2, Output{1} / 2};
		const auto p_sne_divisors = std::vector<Output>{1, 1, 1};
		constexpr auto eta1 = Output{2};
		constexpr auto alpha1 = Output{1} / 2;
		const auto e_half = std::exp(Output{-1} / 2 / 65025);
		const auto e2 = std::exp(Output{-2} / 65025);

		descent(x0, y1, dy1, betas, p_sne_divisors, eta1, alpha1);

		EXPECT_NEAR(eta1 * (-e_half * 4 / 11 - e2 * 16 / 99 + Output{62} / 231) + alpha1 * (Output{1}), dy1[0][0], EPSILON<Output>);
		EXPECT_NEAR(eta1 * (-e_half * 4 / 33 - e2 * 16 / 99 + Output{26} / 231) + alpha1 * (Output{2}), dy1[0][1], EPSILON<Output>);
		EXPECT_NEAR(eta1 * (e_half * 8 / 33 - Output{12} / 77) + alpha1 * (Output{3}), dy1[1][0], EPSILON<Output>);
		EXPECT_NEAR(eta1 * (-e_half * 8 / 33 + Output{12} / 77) + alpha1 * (Output{4}), dy1[1][1], EPSILON<Output>);
		EXPECT_NEAR(eta1 * (e_half * 4 / 33 + e2 * 16 / 99 - Output{26} / 231) + alpha1 * (Output{5}), dy1[2][0], EPSILON<Output>);
		EXPECT_NEAR(eta1 * (e_half * 4 / 11 + e2 * 16 / 99 - Output{62} / 231) + alpha1 * (Output{6}), dy1[2][1], EPSILON<Output>);

		EXPECT_NEAR(eta1 * (-e_half * 4 / 11 - e2 * 16 / 99 + Output{62} / 231) + alpha1 * (Output{1}) + Output{1}, y1[0][0], EPSILON<Output>);
		EXPECT_NEAR(eta1 * (-e_half * 4 / 33 - e2 * 16 / 99 + Output{26} / 231) + alpha1 * (Output{2}) + Output{2}, y1[0][1], EPSILON<Output>);
		EXPECT_NEAR(eta1 * (e_half * 8 / 33 - Output{12} / 77) + alpha1 * (Output{3}) + Output{4}, y1[1][0], EPSILON<Output>);
		EXPECT_NEAR(eta1 * (-e_half * 8 / 33 + Output{12} / 77) + alpha1 * (Output{4}) + Output{3}, y1[1][1], EPSILON<Output>);
		EXPECT_NEAR(eta1 * (e_half * 4 / 33 + e2 * 16 / 99 - Output{26} / 231) + alpha1 * (Output{5}) + Output{5}, y1[2][0], EPSILON<Output>);
		EXPECT_NEAR(eta1 * (e_half * 4 / 11 + e2 * 16 / 99 - Output{62} / 231) + alpha1 * (Output{6}) + Output{6}, y1[2][1], EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, Gradient_Z_IS_1) {
		const auto x = std::vector<image_t>{{1}, {2}, {3}};
		const auto y = std::vector<embedding_t>{{1, 2}, {4, 3}, {5, 6}};
		const auto betas = std::vector<Output>{Output{1} / 2, Output{1} / 2, Output{1} / 2};
		const auto p_sne_divisors = std::vector<Output>{1, 1, 1};
		constexpr auto z = Output{1};
		const auto e_half = std::exp(Output{-1} / 2 / 65025);
		const auto e2 = std::exp(Output{-2} / 65025);

		const auto gradient0 = gradient(x, y, betas, p_sne_divisors, z, 0);
		EXPECT_NEAR(-e_half * 4 / 11 - e2 * 16 / 99 + Output{124} / 1089, gradient0[0], EPSILON<Output>);
		EXPECT_NEAR(-e_half * 4 / 33 - e2 * 16 / 99 + Output{52} / 1089, gradient0[1], EPSILON<Output>);

		const auto gradient1 = gradient(x, y, betas, p_sne_divisors, z, 1);
		EXPECT_NEAR(e_half * 8 / 33 - Output{8} / 121, gradient1[0], EPSILON<Output>);
		EXPECT_NEAR(-e_half * 8 / 33 + Output{8} / 121, gradient1[1], EPSILON<Output>);

		const auto gradient2 = gradient(x, y, betas, p_sne_divisors, z, 2);
		EXPECT_NEAR(e_half * 4 / 33 + e2 * 16 / 99 - Output{52} / 1089, gradient2[0], EPSILON<Output>);
		EXPECT_NEAR(e_half * 4 / 11 + e2 * 16 / 99 - Output{124} / 1089, gradient2[1], EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, Gradient_Real_Z) {
		const auto x = std::vector<image_t>{{1}, {2}, {3}};
		const auto y = std::vector<embedding_t>{{1, 2}, {4, 3}, {5, 6}};
		const auto betas = std::vector<Output>{Output{1} / 2, Output{1} / 2, Output{1} / 2};
		const auto p_sne_divisors = std::vector<Output>{1, 1, 1};
		constexpr auto z_real = Output{14} / 33;
		const auto e_half = std::exp(Output{-1} / 2 / 65025);
		const auto e2 = std::exp(Output{-2} / 65025);

		const auto rzgradient0 = gradient(x, y, betas, p_sne_divisors, z_real, 0);
		EXPECT_NEAR(-e_half * 4 / 11 - e2 * 16 / 99 + Output{62} / 231, rzgradient0[0], EPSILON<Output>);
		EXPECT_NEAR(-e_half * 4 / 33 - e2 * 16 / 99 + Output{26} / 231, rzgradient0[1], EPSILON<Output>);

		const auto rzgradient1 = gradient(x, y, betas, p_sne_divisors, z_real, 1);
		EXPECT_NEAR(e_half * 8 / 33 - Output{12} / 77, rzgradient1[0], EPSILON<Output>);
		EXPECT_NEAR(-e_half * 8 / 33 + Output{12} / 77, rzgradient1[1], EPSILON<Output>);

		const auto rzgradient2 = gradient(x, y, betas, p_sne_divisors, z_real, 2);
		EXPECT_NEAR(e_half * 4 / 33 + e2 * 16 / 99 - Output{26} / 231, rzgradient2[0], EPSILON<Output>);
		EXPECT_NEAR(e_half * 4 / 11 + e2 * 16 / 99 - Output{62} / 231, rzgradient2[1], EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, Z) {
		const auto y0 = std::vector<embedding_t>{{0, 0}, {0, 0}};
		const auto y1 = std::vector<embedding_t>{{1, 2}, {4, 3}};
		const auto y2 = std::vector<embedding_t>{{1, 2}, {4, 3}, {5, 6}};

		EXPECT_NEAR(2, z(y0), EPSILON<Output>);
		EXPECT_NEAR(Output{2} / 11, z(y1), EPSILON<Output>);
		EXPECT_NEAR(Output{14} / 33, z(y2), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, P_TSNE_Sigma2_IS_1_Divisor_IS_1) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		constexpr auto beta = Output{1} / 2;
		const auto e_half = std::exp(Output{-1} / 2 / 65025);
		const auto e2 = std::exp(Output{-2} / 65025);
		constexpr auto divisor = Output{1};

		EXPECT_NEAR(0, p_tsne(x0, 0, 0, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_half / 3, p_tsne(x0, 0, 1, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e2 / 3, p_tsne(x0, 0, 2, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_half / 3, p_tsne(x0, 1, 0, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(0, p_tsne(x0, 1, 1, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_half / 3, p_tsne(x0, 1, 2, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e2 / 3, p_tsne(x0, 2, 0, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_half / 3, p_tsne(x0, 2, 1, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(0, p_tsne(x0, 2, 2, beta, divisor, divisor), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, P_TSNE_Sigma2_IS_2_Divisor_IS_1) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		constexpr auto beta = Output{1} / 4;
		const auto e_quarter = std::exp(Output{-1} / 4 / 65025);
		const auto e1 = std::exp(Output{-1} / 65025);
		constexpr auto divisor = Output{1};

		EXPECT_NEAR(0, p_tsne(x0, 0, 0, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_quarter / 3, p_tsne(x0, 0, 1, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e1 / 3, p_tsne(x0, 0, 2, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_quarter / 3, p_tsne(x0, 1, 0, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(0, p_tsne(x0, 1, 1, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_quarter / 3, p_tsne(x0, 1, 2, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e1 / 3, p_tsne(x0, 2, 0, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_quarter / 3, p_tsne(x0, 2, 1, beta, divisor, divisor), EPSILON<Output>);
		EXPECT_NEAR(0, p_tsne(x0, 2, 2, beta, divisor, divisor), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, P_TSNE_Sigma2_IS_1_Real_Divisor) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		constexpr auto beta = Output{1} / 2;
		const auto e_half = std::exp(Output{-1} / 2 / 65025);
		const auto e2 = std::exp(Output{-2} / 65025);
		const auto divisor_real0 = e_half + e2;
		const auto divisor_real1 = e_half + e_half;
		const auto divisor_real2 = e2 + e_half;

		EXPECT_NEAR(0, p_tsne(x0, 0, 0, beta, divisor_real0, divisor_real0), EPSILON<Output>);
		EXPECT_NEAR((e_half / divisor_real0 + e_half / divisor_real1) / 6, p_tsne(x0, 0, 1, beta, divisor_real0, divisor_real1), EPSILON<Output>);
		EXPECT_NEAR((e2 / divisor_real0 + e2 / divisor_real2) / 6, p_tsne(x0, 0, 2, beta, divisor_real0, divisor_real2), EPSILON<Output>);
		EXPECT_NEAR((e_half / divisor_real0 + e_half / divisor_real1) / 6, p_tsne(x0, 1, 0, beta, divisor_real1, divisor_real0), EPSILON<Output>);
		EXPECT_NEAR(0, p_tsne(x0, 1, 1, beta, divisor_real1, divisor_real1), EPSILON<Output>);
		EXPECT_NEAR((e_half / divisor_real1 + e_half / divisor_real2) / 6, p_tsne(x0, 1, 2, beta, divisor_real1, divisor_real2), EPSILON<Output>);
		EXPECT_NEAR((e2 / divisor_real0 + e2 / divisor_real2) / 6, p_tsne(x0, 2, 0, beta, divisor_real2, divisor_real0), EPSILON<Output>);
		EXPECT_NEAR((e_half / divisor_real1 + e_half / divisor_real2) / 6, p_tsne(x0, 2, 1, beta, divisor_real2, divisor_real1), EPSILON<Output>);
		EXPECT_NEAR(0, p_tsne(x0, 2, 2, beta, divisor_real2, divisor_real2), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, P_TSNE_Sigma2_IS_2_Real_Divisor) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		constexpr auto beta = Output{1} / 4;
		const auto e_quarter = std::exp(Output{-1} / 4 / 65025);
		const auto e1 = std::exp(Output{-1} / 65025);
		const auto divisor2_real0 = e_quarter + e1;
		const auto divisor2_real1 = e_quarter + e_quarter;
		const auto divisor2_real2 = e1 + e_quarter;

		EXPECT_NEAR(0, p_tsne(x0, 0, 0, beta, divisor2_real0, divisor2_real0), EPSILON<Output>);
		EXPECT_NEAR((e_quarter / divisor2_real0 + e_quarter / divisor2_real1) / 6, p_tsne(x0, 0, 1, beta, divisor2_real0, divisor2_real1), EPSILON<Output>);
		EXPECT_NEAR((e1 / divisor2_real0 + e1 / divisor2_real2) / 6, p_tsne(x0, 0, 2, beta, divisor2_real0, divisor2_real2), EPSILON<Output>);
		EXPECT_NEAR((e_quarter / divisor2_real0 + e_quarter / divisor2_real1) / 6, p_tsne(x0, 1, 0, beta, divisor2_real1, divisor2_real0), EPSILON<Output>);
		EXPECT_NEAR(0, p_tsne(x0, 1, 1, beta, divisor2_real1, divisor2_real1), EPSILON<Output>);
		EXPECT_NEAR((e_quarter / divisor2_real1 + e_quarter / divisor2_real2) / 6, p_tsne(x0, 1, 2, beta, divisor2_real1, divisor2_real2), EPSILON<Output>);
		EXPECT_NEAR((e1 / divisor2_real0 + e1 / divisor2_real2) / 6, p_tsne(x0, 2, 0, beta, divisor2_real2, divisor2_real0), EPSILON<Output>);
		EXPECT_NEAR((e_quarter / divisor2_real1 + e_quarter / divisor2_real2) / 6, p_tsne(x0, 2, 1, beta, divisor2_real2, divisor2_real1), EPSILON<Output>);
		EXPECT_NEAR(0, p_tsne(x0, 2, 2, beta, divisor2_real2, divisor2_real2), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, P_SNE_Sigma2_IS_1_Divisor_IS_1) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		constexpr auto beta = Output{1} / 2;
		const auto e_half = std::exp(Output{-1} / 2 / 65025);
		const auto e2 = std::exp(Output{-2} / 65025);
		constexpr auto divisor = Output{1};

		EXPECT_NEAR(0, p_sne(x0, 0, 0, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_half, p_sne(x0, 0, 1, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e2, p_sne(x0, 0, 2, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_half, p_sne(x0, 1, 0, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(0, p_sne(x0, 1, 1, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_half, p_sne(x0, 1, 2, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e2, p_sne(x0, 2, 0, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_half, p_sne(x0, 2, 1, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(0, p_sne(x0, 2, 2, beta, divisor), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, P_SNE_Sigma2_IS_2_Divisor_IS_1) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		constexpr auto beta = Output{1} / 4;
		const auto e_quarter = std::exp(Output{-1} / 4 / 65025);
		const auto e1 = std::exp(Output{-1} / 65025);
		constexpr auto divisor = Output{1};

		EXPECT_NEAR(0, p_sne(x0, 0, 0, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_quarter, p_sne(x0, 0, 1, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e1, p_sne(x0, 0, 2, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_quarter, p_sne(x0, 1, 0, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(0, p_sne(x0, 1, 1, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_quarter, p_sne(x0, 1, 2, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e1, p_sne(x0, 2, 0, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(e_quarter, p_sne(x0, 2, 1, beta, divisor), EPSILON<Output>);
		EXPECT_NEAR(0, p_sne(x0, 2, 2, beta, divisor), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, P_SNE_Sigma2_IS_1_Real_Divisor) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		constexpr auto beta = Output{1} / 2;
		const auto e_half = std::exp(Output{-1} / 2 / 65025);
		const auto e2 = std::exp(Output{-2} / 65025);
		const auto divisor_real0 = e_half + e2;
		const auto divisor_real1 = e_half + e_half;
		const auto divisor_real2 = e2 + e_half;

		EXPECT_NEAR(0, p_sne(x0, 0, 0, beta, divisor_real0), EPSILON<Output>);
		EXPECT_NEAR(e_half / divisor_real0, p_sne(x0, 0, 1, beta, divisor_real0), EPSILON<Output>);
		EXPECT_NEAR(e2 / divisor_real0, p_sne(x0, 0, 2, beta, divisor_real0), EPSILON<Output>);
		EXPECT_NEAR(e_half / divisor_real1, p_sne(x0, 1, 0, beta, divisor_real1), EPSILON<Output>);
		EXPECT_NEAR(0, p_sne(x0, 1, 1, beta, divisor_real1), EPSILON<Output>);
		EXPECT_NEAR(e_half / divisor_real1, p_sne(x0, 1, 2, beta, divisor_real1), EPSILON<Output>);
		EXPECT_NEAR(e2 / divisor_real2, p_sne(x0, 2, 0, beta, divisor_real2), EPSILON<Output>);
		EXPECT_NEAR(e_half / divisor_real2, p_sne(x0, 2, 1, beta, divisor_real2), EPSILON<Output>);
		EXPECT_NEAR(0, p_sne(x0, 2, 2, beta, divisor_real2), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, P_SNE_Sigma2_IS_2_Real_Divisor) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		constexpr auto beta = Output{1} / 4;
		const auto e_quarter = std::exp(Output{-1} / 4 / 65025);
		const auto e1 = std::exp(Output{-1} / 65025);
		const auto divisor2_real0 = e_quarter + e1;
		const auto divisor2_real1 = e_quarter + e_quarter;
		const auto divisor2_real2 = e1 + e_quarter;

		EXPECT_NEAR(0, p_sne(x0, 0, 0, beta, divisor2_real0), EPSILON<Output>);
		EXPECT_NEAR(e_quarter / divisor2_real0, p_sne(x0, 0, 1, beta, divisor2_real0), EPSILON<Output>);
		EXPECT_NEAR(e1 / divisor2_real0, p_sne(x0, 0, 2, beta, divisor2_real0), EPSILON<Output>);
		EXPECT_NEAR(e_quarter / divisor2_real1, p_sne(x0, 1, 0, beta, divisor2_real1), EPSILON<Output>);
		EXPECT_NEAR(0, p_sne(x0, 1, 1, beta, divisor2_real1), EPSILON<Output>);
		EXPECT_NEAR(e_quarter / divisor2_real1, p_sne(x0, 1, 2, beta, divisor2_real1), EPSILON<Output>);
		EXPECT_NEAR(e1 / divisor2_real2, p_sne(x0, 2, 0, beta, divisor2_real2), EPSILON<Output>);
		EXPECT_NEAR(e_quarter / divisor2_real2, p_sne(x0, 2, 1, beta, divisor2_real2), EPSILON<Output>);
		EXPECT_NEAR(0, p_sne(x0, 2, 2, beta, divisor2_real2), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, P_SNEDivisor_Sigma2_IS_1) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		constexpr auto beta = Output{1} / 2;
		const auto e_half = std::exp(Output{-1} / 2 / 65025);
		const auto e2 = std::exp(Output{-2} / 65025);
		EXPECT_NEAR(e_half + e2, p_sne_divisor(x0, 0, beta), EPSILON<Output>);
		EXPECT_NEAR(e_half + e_half, p_sne_divisor(x0, 1, beta), EPSILON<Output>);
		EXPECT_NEAR(e2 + e_half, p_sne_divisor(x0, 2, beta), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, P_SNEDivisor_Sigma2_IS_2) {
		const auto x0 = std::vector<image_t>{{1}, {2}, {3}};
		constexpr auto beta = Output{1} / 4;
		const auto e_quarter = std::exp(Output{-1} / 4 / 65025);
		const auto e1 = std::exp(Output{-1} / 65025);
		EXPECT_NEAR(e_quarter + e1, p_sne_divisor(x0, 0, beta), EPSILON<Output>);
		EXPECT_NEAR(e_quarter + e_quarter, p_sne_divisor(x0, 1, beta), EPSILON<Output>);
		EXPECT_NEAR(e1 + e_quarter, p_sne_divisor(x0, 2, beta), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, Q_TSNE1_Z_IS_1) {
		const auto y0 = std::vector<embedding_t>{{0, 0}, {0, 0}};

		EXPECT_NEAR(0, q_tsne(y0, 0, 0, 1), EPSILON<Output>);
		EXPECT_NEAR(1, q_tsne(y0, 0, 1, 1), EPSILON<Output>);
		EXPECT_NEAR(1, q_tsne(y0, 1, 0, 1), EPSILON<Output>);
		EXPECT_NEAR(0, q_tsne(y0, 1, 1, 1), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, Q_TSNE1_REAL_Z) {
		const auto y0 = std::vector<embedding_t>{{0, 0}, {0, 0}};

		EXPECT_NEAR(0, q_tsne(y0, 0, 0, z(y0)), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 2, q_tsne(y0, 0, 1, z(y0)), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 2, q_tsne(y0, 1, 0, z(y0)), EPSILON<Output>);
		EXPECT_NEAR(0, q_tsne(y0, 1, 1, z(y0)), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, Q_TSNE2_Z_IS_1) {
		const auto y1 = std::vector<embedding_t>{{1, 2}, {4, 3}};

		EXPECT_NEAR(0, q_tsne(y1, 0, 0, 1), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 11, q_tsne(y1, 0, 1, 1), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 11, q_tsne(y1, 1, 0, 1), EPSILON<Output>);
		EXPECT_NEAR(0, q_tsne(y1, 1, 1, 1), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, Q_TSNE2_REAL_Z) {
		const auto y1 = std::vector<embedding_t>{{1, 2}, {4, 3}};

		EXPECT_NEAR(0, q_tsne(y1, 0, 0, z(y1)), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 2, q_tsne(y1, 0, 1, z(y1)), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 2, q_tsne(y1, 1, 0, z(y1)), EPSILON<Output>);
		EXPECT_NEAR(0, q_tsne(y1, 1, 1, z(y1)), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, Q_TSNE3_Z_IS_1) {
		const auto y2 = std::vector<embedding_t>{{1, 2}, {4, 3}, {5, 6}};

		EXPECT_NEAR(0, q_tsne(y2, 0, 0, 1), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 11, q_tsne(y2, 0, 1, 1), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 33, q_tsne(y2, 0, 2, 1), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 11, q_tsne(y2, 1, 0, 1), EPSILON<Output>);
		EXPECT_NEAR(0, q_tsne(y2, 1, 1, 1), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 11, q_tsne(y2, 1, 2, 1), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 33, q_tsne(y2, 2, 0, 1), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 11, q_tsne(y2, 2, 1, 1), EPSILON<Output>);
		EXPECT_NEAR(0, q_tsne(y2, 2, 2, 1), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, Q_TSNE3_REAL_Z) {
		const auto y2 = std::vector<embedding_t>{{1, 2}, {4, 3}, {5, 6}};

		EXPECT_NEAR(0, q_tsne(y2, 0, 0, z(y2)), EPSILON<Output>);
		EXPECT_NEAR(Output{3} / 14, q_tsne(y2, 0, 1, z(y2)), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 14, q_tsne(y2, 0, 2, z(y2)), EPSILON<Output>);
		EXPECT_NEAR(Output{3} / 14, q_tsne(y2, 1, 0, z(y2)), EPSILON<Output>);
		EXPECT_NEAR(0, q_tsne(y2, 1, 1, z(y2)), EPSILON<Output>);
		EXPECT_NEAR(Output{3} / 14, q_tsne(y2, 1, 2, z(y2)), EPSILON<Output>);
		EXPECT_NEAR(Output{1} / 14, q_tsne(y2, 2, 0, z(y2)), EPSILON<Output>);
		EXPECT_NEAR(Output{3} / 14, q_tsne(y2, 2, 1, z(y2)), EPSILON<Output>);
		EXPECT_NEAR(0, q_tsne(y2, 2, 2, z(y2)), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, Beta1) {
		auto x0 = std::vector<image_t>(3);
		auto betas = std::vector<Output>(3);
		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			x0[0][i] = 0;
			x0[1][i] = 16;
			x0[2][i] = 24;
		}
		constexpr auto beta0 = Output{1} / 10;
		const auto divisor0 = p_sne_divisor(x0, 0, beta0);
		const auto p_sne01 = p_sne(x0, 0, 1, beta0, divisor0);
		const auto p_sne02 = p_sne(x0, 0, 2, beta0, divisor0);
		const auto log_u0 = -p_sne01 * std::log2(p_sne01) - p_sne02 * std::log2(p_sne02);
		const auto u0 = std::exp2(log_u0);
		beta(x0, betas, u0);
		EXPECT_NEAR(beta0, betas[0], EPSILON_BETA<Output>);
	}

	TEST_F(SolverRefNaiveTest, Beta2) {
		auto x0 = std::vector<image_t>(3);
		auto betas = std::vector<Output>(3);
		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			x0[0][i] = 0;
			x0[1][i] = 16;
			x0[2][i] = 24;
		}
		constexpr auto beta1 = Output{1} / 10;
		const auto divisor1 = p_sne_divisor(x0, 1, beta1);
		const auto p_sne10 = p_sne(x0, 1, 0, beta1, divisor1);
		const auto p_sne12 = p_sne(x0, 1, 2, beta1, divisor1);
		const auto log_u1 = -p_sne10 * std::log2(p_sne10) - p_sne12 * std::log2(p_sne12);
		const auto u1 = std::exp2(log_u1);
		beta(x0, betas, u1);
		EXPECT_NEAR(beta1, betas[1], EPSILON_BETA<Output>);
	}

	TEST_F(SolverRefNaiveTest, Beta3) {
		auto x0 = std::vector<image_t>(3);
		auto betas = std::vector<Output>(3);
		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			x0[0][i] = 0;
			x0[1][i] = 16;
			x0[2][i] = 24;
		}
		constexpr auto beta2 = Output{1} / 10;
		const auto divisor2 = p_sne_divisor(x0, 2, beta2);
		const auto p_sne20 = p_sne(x0, 2, 0, beta2, divisor2);
		const auto p_sne21 = p_sne(x0, 2, 1, beta2, divisor2);
		const auto log_u2 = -p_sne20 * std::log2(p_sne20) - p_sne21 * std::log2(p_sne21);
		const auto u2 = std::exp2(log_u2);
		beta(x0, betas, u2);
		EXPECT_NEAR(beta2, betas[2], EPSILON_BETA<Output>);
	}

	TEST_F(SolverRefNaiveTest, BisectionP1) {
		constexpr auto N = std::size_t{128};
		constexpr auto u = Output{16};
		EXPECT_NEAR(Output{3} / 4, bisection_p(N, u, P_MIN, P_MAX), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, BisectionP2) {
		constexpr auto N = std::size_t{70000};
		constexpr auto u = Output{50};
		EXPECT_NEAR(static_cast<Output>(0.84099993360670733178L), bisection_p(N, u, P_MIN, P_MAX), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, DistanceX2_00) {
		auto x0 = image_t();

		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			x0[i] = 0;
		}
		EXPECT_NEAR(0, distance_x2(x0, x0), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, DistanceX2_01) {
		auto x0 = image_t();
		auto x1 = image_t();

		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			x0[i] = 0;
			x1[i] = static_cast<typename image_t::value_type>(i % 3);
		}
		EXPECT_NEAR(Output{1305} / 65025, distance_x2(x0, x1), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, DistanceX2_02) {
		auto x0 = image_t();
		auto x2 = image_t();

		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			x0[i] = 0;
			x2[i] = static_cast<typename image_t::value_type>(i % 2);
		}
		EXPECT_NEAR(Output{392} / 65025, distance_x2(x0, x2), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, DistanceX2_03) {
		auto x0 = image_t();
		auto x3 = image_t();

		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			x0[i] = 0;
			x3[i] = static_cast<typename image_t::value_type>((i + 1) % 2);
		}
		EXPECT_NEAR(Output{392} / 65025, distance_x2(x0, x3), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, DistanceX2_12) {
		auto x1 = image_t();
		auto x2 = image_t();

		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			x1[i] = static_cast<typename image_t::value_type>(i % 3);
			x2[i] = static_cast<typename image_t::value_type>(i % 2);
		}
		EXPECT_NEAR(Output{915} / 65025, distance_x2(x1, x2), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, DistanceX2_13) {
		auto x1 = image_t();
		auto x3 = image_t();

		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			x1[i] = static_cast<typename image_t::value_type>(i % 3);
			x3[i] = static_cast<typename image_t::value_type>((i + 1) % 2);
		}
		EXPECT_NEAR(Output{913} / 65025, distance_x2(x1, x3), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, DistanceX2_23) {
		auto x2 = image_t();
		auto x3 = image_t();

		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			x2[i] = static_cast<typename image_t::value_type>(i % 2);
			x3[i] = static_cast<typename image_t::value_type>((i + 1) % 2);
		}

		EXPECT_NEAR(Output{784} / 65025, distance_x2(x2, x3), EPSILON<Output>);
	}

	TEST_F(SolverRefNaiveTest, DistanceY2) {
		EXPECT_NEAR(0, distance_y2({1, 1}, {1, 1}), EPSILON<Output>);
		EXPECT_NEAR(2, distance_y2({0, 1}, {1, 0}), EPSILON<Output>);
		EXPECT_NEAR(8, distance_y2({-1, 1}, {1, -1}), EPSILON<Output>);
	}
} // namespace t_sne
#pragma clang diagnostic pop
