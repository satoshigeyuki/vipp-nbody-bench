#include <limits>
#define ENABLE_TEST
#ifdef ENABLE_TEST
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include <gtest/gtest.h>
#pragma clang diagnostic pop
#endif
#include <problem.hpp>

#ifdef ENABLE_PRINT_IMAGE
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wweak-vtables"

namespace t_sne {
	using Output = double;
	using Interpolate = float;
	using image_t = Problem<Output, Interpolate>::image_t;
	constexpr auto PI = Problem<Output, Interpolate>::PI;
	constexpr auto TRAIN_SET_N = Problem<Output, Interpolate>::TRAIN_SET_N;
	constexpr auto N = Problem<Output, Interpolate>::N;
	constexpr auto IMAGE_SIZE = Problem<Output, Interpolate>::IMAGE_SIZE;
	constexpr auto X_SIZE = Problem<Output, Interpolate>::X_SIZE;
	constexpr auto Y_SIZE = Problem<Output, Interpolate>::Y_SIZE;
	constexpr auto PIXEL_MAX = std::numeric_limits<std::uint8_t>::max();

	template struct Problem<float, float>;
	template struct Problem<float, double>;
	template struct Problem<double, float>;
	template struct Problem<double, double>;

#ifdef ENABLE_PRINT_IMAGE
	struct RGBA {
		std::uint8_t r, g, b, a;
		RGBA() = default;
	};

	void print_image(const image_t &image, const char *path) {
		auto output_image = std::array<RGBA, X_SIZE * Y_SIZE>();
		for (auto i = decltype(X_SIZE * Y_SIZE){0}; i < X_SIZE * Y_SIZE; ++i) {
			output_image[i].r = image[i];
			output_image[i].a = std::numeric_limits<std::uint8_t>::max();
		}
		stbi_write_png(path, X_SIZE, Y_SIZE, sizeof(RGBA), &output_image, 0);
	}
#endif

	struct ProblemTest : ::testing::Test {
	  protected:
		static Problem<Output, Interpolate> getProblem(const std::filesystem::path &data_path,
		                                               const std::size_t n = 1,
		                                               const std::uint_fast64_t seed = 0) {
			return Problem<Output, Interpolate>(data_path, n, seed);
		}

		static std::optional<image_t> augment_image(const image_t &base_image, const Interpolate theta, const Interpolate lambda_x, const Interpolate lambda_y, const Interpolate delta_x, const Interpolate delta_y) {
			return Problem<Output, Interpolate>::augment_image(base_image, theta, lambda_x, lambda_y, delta_x, delta_y);
		}
	};

	TEST_F(ProblemTest, LoadImage) {
		auto problem = ProblemTest::getProblem("data");
		EXPECT_EQ(problem.images().size(), N);
		constexpr std::uint8_t train_first[IMAGE_SIZE] = {
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x12, 0x12, 0x12,
		    0x7e, 0x88, 0xaf, 0x1a, 0xa6, 0xff, 0xf7, 0x7f, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1e, 0x24, 0x5e, 0x9a,
		    0xaa, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xe1, 0xac, 0xfd, 0xf2, 0xc3, 0x40,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x31,
		    0xee, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfb, 0x5d, 0x52,
		    0x52, 0x38, 0x27, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x12, 0xdb, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xc6, 0xb6,
		    0xf7, 0xf1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x9c, 0x6b, 0xfd,
		    0xfd, 0xcd, 0x0b, 0x00, 0x2b, 0x9a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x0e, 0x01, 0x9a, 0xfd, 0x5a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x8b, 0xfd, 0xbe, 0x02, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0b,
		    0xbe, 0xfd, 0x46, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x23, 0xf1, 0xe1, 0xa0, 0x6c, 0x01, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x51, 0xf0, 0xfd,
		    0xfd, 0x77, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x2d, 0xba, 0xfd, 0xfd, 0x96, 0x1b, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x5d, 0xfc, 0xfd, 0xbb,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0xf9, 0xfd, 0xf9, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x2e, 0x82, 0xb7, 0xfd, 0xfd, 0xcf, 0x02, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x27, 0x94, 0xe5, 0xfd, 0xfd, 0xfd, 0xfa, 0xb6,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x72, 0xdd, 0xfd, 0xfd, 0xfd,
		    0xfd, 0xc9, 0x4e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x17, 0x42, 0xd5, 0xfd,
		    0xfd, 0xfd, 0xfd, 0xc6, 0x51, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0xab,
		    0xdb, 0xfd, 0xfd, 0xfd, 0xfd, 0xc3, 0x50, 0x09, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x37, 0xac, 0xe2, 0xfd, 0xfd, 0xfd, 0xfd, 0xf4, 0x85, 0x0b, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x88, 0xfd, 0xfd, 0xfd, 0xd4, 0x87, 0x84, 0x10,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00};
		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			EXPECT_EQ(train_first[i], problem.images().front()[i]);
		}
		constexpr std::uint8_t train_last[IMAGE_SIZE] = {
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x26, 0x30, 0x30, 0x16, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x61, 0xc6, 0xf3, 0xfe, 0xfe, 0xd4,
		    0x1b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x43, 0xac, 0xfe, 0xfe,
		    0xe1, 0xda, 0xda, 0xed, 0xf8, 0x28, 0x00, 0x15, 0xa4, 0xbb, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59,
		    0xdb, 0xfe, 0x61, 0x43, 0x0e, 0x00, 0x00, 0x5c, 0xe7, 0x7a, 0x17, 0xcb,
		    0xec, 0x3b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x19, 0xd9, 0xf2, 0x5c, 0x04, 0x00, 0x00, 0x00, 0x00, 0x04,
		    0x93, 0xfd, 0xf0, 0xe8, 0x5c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x65, 0xff, 0x5c, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x69, 0xfe, 0xfe, 0xb1, 0x0b, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xa7, 0xf4,
		    0x29, 0x00, 0x00, 0x00, 0x07, 0x4c, 0xc7, 0xee, 0xef, 0x5e, 0x0a, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0xc0, 0x79, 0x00, 0x00, 0x02, 0x3f, 0xb4, 0xfe, 0xe9, 0x7e,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xbe, 0xc4, 0x0e, 0x02, 0x61, 0xfe,
		    0xfc, 0x92, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x82, 0xe1,
		    0x47, 0xb4, 0xe8, 0xb5, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x82, 0xfe, 0xfe, 0xe6, 0x2e, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x06, 0x4d, 0xf4, 0xfe, 0xa2, 0x04, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6e, 0xfe, 0xda, 0xfe,
		    0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x83,
		    0xfe, 0x9a, 0x1c, 0xd5, 0x56, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x42, 0xd1, 0x99, 0x13, 0x13, 0xe9, 0x3c, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x8e, 0xfe, 0xa5, 0x00, 0x0e, 0xd8,
		    0xa7, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5a, 0xfe,
		    0xaf, 0x00, 0x12, 0xe5, 0x5c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x1a, 0xe5, 0xf9, 0xb0, 0xde, 0xf4, 0x2c, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x49, 0xc1, 0xc5, 0x86, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00};
		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			EXPECT_EQ(train_last[i], problem.images()[TRAIN_SET_N - 1][i]);
		}
		constexpr std::uint8_t verify_first[IMAGE_SIZE] = {
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54, 0xb9,
		    0x9f, 0x97, 0x3c, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0xde, 0xfe, 0xfe, 0xfe, 0xfe, 0xf1, 0xc6, 0xc6, 0xc6, 0xc6,
		    0xc6, 0xc6, 0xc6, 0xc6, 0xaa, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x43, 0x72, 0x48, 0x72, 0xa3, 0xe3,
		    0xfe, 0xe1, 0xfe, 0xfe, 0xfe, 0xfa, 0xe5, 0xfe, 0xfe, 0x8c, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x11, 0x42, 0x0e, 0x43, 0x43, 0x43, 0x3b, 0x15, 0xec,
		    0xfe, 0x6a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x53, 0xfd, 0xd1, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0xe9, 0xff, 0x53, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x81, 0xfe, 0xee,
		    0x2c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x3b, 0xf9, 0xfe, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x85, 0xfe, 0xbb, 0x05, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0xcd, 0xf8, 0x3a, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7e,
		    0xfe, 0xb6, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x4b, 0xfb, 0xf0, 0x39, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x13, 0xdd, 0xfe, 0xa6, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xcb, 0xfe, 0xdb,
		    0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x26, 0xfe, 0xfe, 0x4d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x1f, 0xe0, 0xfe, 0x73, 0x01, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x85, 0xfe, 0xfe, 0x34, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3d, 0xf2,
		    0xfe, 0xfe, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x79, 0xfe, 0xfe, 0xdb, 0x28, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x79, 0xfe, 0xcf, 0x12, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00};
		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			EXPECT_EQ(verify_first[i], problem.images()[TRAIN_SET_N][i]);
		}
		constexpr std::uint8_t verify_last[IMAGE_SIZE] = {
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x08, 0x75, 0xfe, 0xdc, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x0d, 0x5f, 0xd4, 0xfd, 0xfd, 0xfd, 0x9d, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x5f, 0xd1, 0xfd, 0xfd, 0xfd, 0xf5,
		    0x7d, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x28, 0x60, 0xce, 0xfd, 0xfe,
		    0xfd, 0xfd, 0xc6, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2c, 0xb6, 0xf0,
		    0xfd, 0xfd, 0xfd, 0xfe, 0xfd, 0xc6, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x0f, 0x3c, 0x3c, 0xa8, 0xfd, 0xfd, 0xfe, 0xc8, 0x17, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46, 0xf7, 0xfd, 0xfd, 0xf5,
		    0x15, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4b, 0xcf,
		    0xfd, 0xfd, 0xcf, 0x5c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x4f, 0xdb, 0xfd, 0xfd, 0xfd, 0x8a, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x69, 0xfa, 0xfd, 0xfd, 0xfd, 0x22, 0x01, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5f, 0xfe, 0xfe, 0xfe, 0xfe,
		    0x5e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x0d, 0x0d, 0x0d, 0x08, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6b,
		    0xfd, 0xfd, 0xfd, 0xcc, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x15, 0xa6, 0xfd,
		    0xfd, 0xfd, 0xd4, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x21, 0xd9, 0xfd, 0xfd, 0x84, 0x40, 0x00, 0x00, 0x12, 0x2b,
		    0x9d, 0xab, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xa0, 0x02, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xa6, 0xfd, 0xfd, 0xf2, 0x31, 0x11,
		    0x31, 0x9e, 0xd2, 0xfe, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd,
		    0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0xe3, 0xfd,
		    0xfd, 0xcf, 0x0f, 0xac, 0xfd, 0xfd, 0xfd, 0xfe, 0xf7, 0xc9, 0xfd, 0xd2,
		    0xd2, 0xfd, 0xfd, 0xaf, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x0a, 0xe4, 0xfd, 0xfd, 0xe0, 0x57, 0xf2, 0xfd, 0xfd, 0xb8, 0x3c,
		    0x36, 0x09, 0x3c, 0x23, 0xb6, 0xfd, 0xfd, 0x34, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0xfd, 0xfd, 0xfd, 0xfd, 0xe7, 0xfd,
		    0xfd, 0xfd, 0x5d, 0x56, 0x56, 0x56, 0x6d, 0xd9, 0xfd, 0xfd, 0x86, 0x05,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x73, 0xfd,
		    0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfe, 0xfd, 0xfd, 0xfd, 0xfd,
		    0xfd, 0x86, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x03, 0xa6, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfd, 0xfe,
		    0xfd, 0xfd, 0xfd, 0xaf, 0x34, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x23, 0x84, 0xe1, 0xfd,
		    0xfd, 0xfd, 0xc3, 0x84, 0x84, 0x84, 0x6e, 0x04, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		    0x00, 0x00, 0x00, 0x00};
		for (auto i = decltype(IMAGE_SIZE){0}; i < IMAGE_SIZE; ++i) {
			EXPECT_EQ(verify_last[i], problem.images().back()[i]);
		}
	}

	TEST_F(ProblemTest, LoadLabel) {
		auto problem = ProblemTest::getProblem("data");
		EXPECT_EQ(problem.labels().size(), N);
		EXPECT_EQ(5, problem.labels().front());
		EXPECT_EQ(8, problem.labels()[TRAIN_SET_N - 1]);
		EXPECT_EQ(7, problem.labels()[TRAIN_SET_N]);
		EXPECT_EQ(6, problem.labels().back());
	}

	TEST_F(ProblemTest, Augment) {
		auto problem = ProblemTest::getProblem("data", 2);
		EXPECT_EQ(problem.images().size(), 2 * N);
		EXPECT_EQ(problem.labels().size(), 2 * N);
	}

	TEST_F(ProblemTest, Move) {
		auto image = image_t();
		constexpr auto DELTA_X = Interpolate{23} / 10;
		constexpr auto DELTA_Y = Interpolate{37} / 10;
		constexpr auto MIN_X = std::size_t{5};
		constexpr auto MAX_X = std::size_t{22};
		constexpr auto MIN_Y = std::size_t{5};
		constexpr auto MAX_Y = std::size_t{22};
		for (auto y = MIN_Y; y <= MAX_Y; ++y) {
			for (auto x = MIN_X; x <= MAX_X; ++x) {
				image[y * X_SIZE + x] = PIXEL_MAX;
			}
		}
		const auto transformed_image = ProblemTest::augment_image(image, 0, 1, 1, DELTA_X, DELTA_Y);
		EXPECT_TRUE(transformed_image.has_value());
		for (auto y = decltype(Y_SIZE){0}; y < Y_SIZE; ++y) {
			for (auto x = decltype(X_SIZE){0}; x < X_SIZE; ++x) {
				if (x < static_cast<std::size_t>(MIN_X + std::floor(DELTA_X)) ||
				    x > static_cast<std::size_t>(MAX_X + std::ceil(DELTA_X)) ||
				    y < static_cast<std::size_t>(MIN_Y + std::floor(DELTA_Y)) ||
				    y > static_cast<std::size_t>(MAX_Y + std::ceil(DELTA_Y))) {
					EXPECT_EQ(0, (*transformed_image)[y * X_SIZE + x]);
				}
			}
		}
	}

	TEST_F(ProblemTest, Scale) {
		auto image = image_t();
		constexpr auto LAMBDA_X = Interpolate{13} / 10;
		constexpr auto LAMBDA_Y = Interpolate{27} / 10;
		constexpr auto MIN_X = std::size_t{9};
		constexpr auto MAX_X = std::size_t{16};
		constexpr auto MIN_Y = std::size_t{10};
		constexpr auto MAX_Y = std::size_t{16};
		constexpr auto CENTER_X = (Interpolate{X_SIZE} - 1) / 2;
		constexpr auto CENTER_Y = (Interpolate{Y_SIZE} - 1) / 2;
		for (auto y = MIN_Y; y <= MAX_Y; ++y) {
			for (auto x = MIN_X; x <= MAX_X; ++x) {
				image[y * X_SIZE + x] = PIXEL_MAX;
			}
		}
		const auto transformed_image =
		    ProblemTest::augment_image(image, 0, LAMBDA_X, LAMBDA_Y, 0, 0);
		EXPECT_TRUE(transformed_image.has_value());
		for (auto y = decltype(Y_SIZE){0}; y < Y_SIZE; ++y) {
			for (auto x = decltype(X_SIZE){0}; x < X_SIZE; ++x) {
				if (x < CENTER_X - (CENTER_X - static_cast<Interpolate>(MIN_X - 1)) * LAMBDA_X ||
				    x > CENTER_X - (CENTER_X - static_cast<Interpolate>(MAX_X + 1)) * LAMBDA_X ||
				    y < CENTER_Y - (CENTER_Y - static_cast<Interpolate>(MIN_Y - 1)) * LAMBDA_Y ||
				    y > CENTER_Y - (CENTER_Y - static_cast<Interpolate>(MAX_Y + 1)) * LAMBDA_Y) {
					EXPECT_EQ(0, (*transformed_image)[y * X_SIZE + x]);
				}
			}
		}
	}

	TEST_F(ProblemTest, Rotate) {
		auto image = image_t();
		constexpr auto THETA = PI / 9;
		constexpr auto MIN_X = std::size_t{8};
		constexpr auto MAX_X = std::size_t{19};
		constexpr auto MIN_Y = std::size_t{13};
		constexpr auto MAX_Y = std::size_t{14};
		constexpr auto CENTER_X = (Interpolate{X_SIZE} - 1) / 2;
		constexpr auto CENTER_Y = (Interpolate{Y_SIZE} - 1) / 2;
		for (auto y = MIN_Y; y <= MAX_Y; ++y) {
			for (auto x = MIN_X; x <= MAX_X; ++x) {
				image[y * X_SIZE + x] = PIXEL_MAX;
			}
		}
		const auto transformed_image = ProblemTest::augment_image(image, THETA, 1, 1, 0, 0);
		EXPECT_TRUE(transformed_image.has_value());
		for (auto y = decltype(Y_SIZE){0}; y < Y_SIZE; ++y) {
			for (auto x = decltype(X_SIZE){0}; x < X_SIZE; ++x) {
				if (x < CENTER_X - (CENTER_X - MIN_X) * std::cos(THETA) - std::sin(THETA) / 2 - 1 ||
				    x > CENTER_X - (CENTER_X - MAX_X) * std::cos(THETA) + std::sin(THETA) / 2 + 1 ||
				    y < CENTER_Y - (x + Interpolate{1} / 2) / std::cos(THETA) - 1 ||
				    y > CENTER_Y + (x + Interpolate{1} / 2) / std::cos(THETA) + 1) {
					EXPECT_EQ(0, (*transformed_image)[y * X_SIZE + x]);
				}
			}
		}
	}

	TEST_F(ProblemTest, LerpMove) {
		auto image = image_t();
		constexpr auto DELTA_X = Interpolate{23} / 10;
		constexpr auto DELTA_Y = Interpolate{37} / 10;
		image[13 * X_SIZE + 13] = PIXEL_MAX;
		const auto transformed_image = ProblemTest::augment_image(image, 0, 1, 1, DELTA_X, DELTA_Y);
		EXPECT_TRUE(transformed_image.has_value());
		EXPECT_EQ(54, (*transformed_image)[16 * X_SIZE + 15]);
		EXPECT_EQ(23, (*transformed_image)[16 * X_SIZE + 16]);
		EXPECT_EQ(125, (*transformed_image)[17 * X_SIZE + 15]);
		EXPECT_EQ(54, (*transformed_image)[17 * X_SIZE + 16]);
	}

	TEST_F(ProblemTest, LerpScale) {
		auto image = image_t();
		constexpr auto LAMBDA_X = Interpolate{13} / 10;
		constexpr auto LAMBDA_Y = Interpolate{27} / 10;
		image[13 * X_SIZE + 13] = PIXEL_MAX;
		const auto transformed_image = ProblemTest::augment_image(image, 0, LAMBDA_X, LAMBDA_Y, 0, 0);
		EXPECT_TRUE(transformed_image.has_value());
		EXPECT_EQ(18, (*transformed_image)[10 * X_SIZE + 12]);
		EXPECT_EQ(46, (*transformed_image)[10 * X_SIZE + 13]);
		EXPECT_EQ(6, (*transformed_image)[10 * X_SIZE + 14]);
		EXPECT_EQ(51, (*transformed_image)[11 * X_SIZE + 12]);
		EXPECT_EQ(129, (*transformed_image)[11 * X_SIZE + 13]);
		EXPECT_EQ(17, (*transformed_image)[11 * X_SIZE + 14]);
		EXPECT_EQ(83, (*transformed_image)[12 * X_SIZE + 12]);
		EXPECT_EQ(213, (*transformed_image)[12 * X_SIZE + 13]);
		EXPECT_EQ(28, (*transformed_image)[12 * X_SIZE + 14]);
		EXPECT_EQ(60, (*transformed_image)[13 * X_SIZE + 12]);
		EXPECT_EQ(155, (*transformed_image)[13 * X_SIZE + 13]);
		EXPECT_EQ(20, (*transformed_image)[13 * X_SIZE + 14]);
		EXPECT_EQ(28, (*transformed_image)[14 * X_SIZE + 12]);
		EXPECT_EQ(71, (*transformed_image)[14 * X_SIZE + 13]);
		EXPECT_EQ(9, (*transformed_image)[14 * X_SIZE + 14]);
	}

	TEST_F(ProblemTest, LerpLotate) {
		auto image = image_t();
		image[13 * X_SIZE + 14] = PIXEL_MAX;
		const auto transformed_image = ProblemTest::augment_image(image, PI / 4, 1, 1, 0, 0);
		EXPECT_TRUE(transformed_image.has_value());
		EXPECT_EQ(101, (*transformed_image)[13 * X_SIZE + 14]);
		EXPECT_EQ(101, (*transformed_image)[14 * X_SIZE + 14]);
		EXPECT_EQ(17, (*transformed_image)[13 * X_SIZE + 15]);
		EXPECT_EQ(17, (*transformed_image)[14 * X_SIZE + 15]);
	}

	TEST_F(ProblemTest, MoveFail) {
		auto image = image_t();
		constexpr auto DELTA_X = Interpolate{-1};
		constexpr auto DELTA_Y = Interpolate{-1};
		image[0] = PIXEL_MAX;
		const auto transformed_image = ProblemTest::augment_image(image, 0, 1, 1, DELTA_X, DELTA_Y);
		EXPECT_FALSE(transformed_image.has_value());
	}

	TEST_F(ProblemTest, ScaleFail) {
		auto image = image_t();
		constexpr auto LAMBDA_X = Interpolate{-4} / 5;
		constexpr auto LAMBDA_Y = Interpolate{27} / 10;
		image[27 * X_SIZE] = PIXEL_MAX;
		const auto transformed_image = ProblemTest::augment_image(image, 0, LAMBDA_X, LAMBDA_Y, 0, 0);
		EXPECT_FALSE(transformed_image.has_value());
	}

	TEST_F(ProblemTest, RotateFail) {
		auto image = image_t();
		constexpr auto THETA = PI / 9;
		image[26 * X_SIZE + 1] = PIXEL_MAX;
		const auto transformed_image = ProblemTest::augment_image(image, THETA, 1, 1, 0, 0);
		EXPECT_FALSE(transformed_image.has_value());
	}

	TEST_F(ProblemTest, Move2) {
		auto image = image_t();
		image[13 * X_SIZE + 13] = PIXEL_MAX;
		image[13 * X_SIZE + 14] = PIXEL_MAX;
		image[14 * X_SIZE + 13] = PIXEL_MAX;
		image[14 * X_SIZE + 14] = PIXEL_MAX;
		const auto transformed_image = ProblemTest::augment_image(image, 0, 1, 1, 0.5, 0.5);
		EXPECT_TRUE(transformed_image.has_value());
		EXPECT_EQ(64, (*transformed_image)[13 * X_SIZE + 13]);
		EXPECT_EQ(128, (*transformed_image)[13 * X_SIZE + 14]);
		EXPECT_EQ(64, (*transformed_image)[13 * X_SIZE + 15]);
		EXPECT_EQ(128, (*transformed_image)[14 * X_SIZE + 13]);
		EXPECT_EQ(255, (*transformed_image)[14 * X_SIZE + 14]);
		EXPECT_EQ(128, (*transformed_image)[14 * X_SIZE + 15]);
		EXPECT_EQ(64, (*transformed_image)[15 * X_SIZE + 13]);
		EXPECT_EQ(128, (*transformed_image)[15 * X_SIZE + 14]);
		EXPECT_EQ(64, (*transformed_image)[15 * X_SIZE + 15]);
	}
} // namespace t_sne
#pragma clang diagnostic pop
