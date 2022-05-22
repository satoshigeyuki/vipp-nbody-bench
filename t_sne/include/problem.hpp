#ifndef T_SNE_PROBLEM_HPP
#define T_SNE_PROBLEM_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace t_sne {
	template <class Output>
	struct DefaultAlphaGenerator;

	/// @brief 入力データの生成を行う.
	template <class Output, class Interpolate = float, class AlphaGenerator = DefaultAlphaGenerator<Output>>
	struct Problem final {
		static constexpr auto PI = static_cast<Interpolate>(3.1415926535897932384626433832795028L);
		static constexpr auto LAMBDA_RATE_MAX = Interpolate{6} / 5;
		// MNISTデータセットのサイズ
		static constexpr auto TRAIN_SET_N = std::size_t{60000};
		static constexpr auto TEST_SET_N = std::size_t{10000};
		static constexpr auto N = TRAIN_SET_N + TEST_SET_N;
		static constexpr auto X_SIZE = std::size_t{28};
		static constexpr auto Y_SIZE = std::size_t{28};
		static constexpr auto IMAGE_SIZE = X_SIZE * Y_SIZE;
		static constexpr auto EMBEDDING_DIMENSION = std::size_t{2};
		static constexpr auto DEFAULT_MIN_THETA = -PI / 9;
		static constexpr auto DEFAULT_MAX_THETA = PI / 9;
		static constexpr auto MIN_LAMBDA = Interpolate{4} / 5;
		static constexpr auto DEFAULT_MIN_LAMBDA = Interpolate{4} / 5;
		static constexpr auto DEFAULT_MAX_LAMBDA = Interpolate{6} / 5;
		static constexpr auto DEFAULT_AVERAGE_DELTA = Interpolate{0};
		static constexpr auto DEFAULT_VARIANCE_DELTA = Interpolate{2};
		static constexpr auto TRAINING_SET_IMAGE_FILE_PATH = "train-images-idx3-ubyte";
		static constexpr auto TRAINING_SET_LABEL_FILE_PATH = "train-labels-idx1-ubyte";
		static constexpr auto TEST_SET_IMAGE_FILE_PATH = "t10k-images-idx3-ubyte";
		static constexpr auto TEST_SET_LABEL_FILE_PATH = "t10k-labels-idx1-ubyte";
		static constexpr auto P_BISECTION_ITER = std::size_t{50};
		static constexpr auto BETA_BISECTION_ITER = std::size_t{1000};
		static constexpr auto P_MIN = Output{3} / 4;
		static constexpr auto P_MAX = Output{1};
		static constexpr auto BARNES_HUT_WIDTH_PADDING = Output{1} / 100000;

		using image_t = std::array<std::uint8_t, IMAGE_SIZE>;
		using embedding_t = std::array<Output, EMBEDDING_DIMENSION>;
		using label_t = std::uint8_t;

		static constexpr auto PIXEL_MAX = Output{std::numeric_limits<typename image_t::value_type>::max()};

		static_assert(std::is_invocable_r_v<Output, AlphaGenerator, std::size_t>, "Template Parameter AlphaGenerator of t_sne::Problem must be called as AlphaGenerator(std::size_t) -> Output");
		static_assert(!std::is_const_v<AlphaGenerator> && !std::is_volatile_v<AlphaGenerator> && !std::is_reference_v<AlphaGenerator>, "Template Parameter AlphaGenerator of t_sne::Problem must be non-cvref");

		/// @param data_path MNISTのデータファイルの保存されているディレクトリ
		/// @param n MNISTのデータファイルをn倍に増幅する
		/// @param seed データ増幅の際に用いる乱数のシード
		Problem(const std::filesystem::path &data_path,
		        const std::size_t n = 1,
		        const std::size_t t = 1000,
		        const double a = 2,
		        const std::uint_fast64_t seed = 0,
		        const Output eta = 200,
		        const Output u = 50,
		        const Output theta = Output{1} / 2)
		    : Problem(
		          data_path,
		          std::uniform_real_distribution<Interpolate>(DEFAULT_MIN_THETA, DEFAULT_MAX_THETA),
		          std::uniform_real_distribution<Interpolate>(DEFAULT_MIN_LAMBDA, DEFAULT_MAX_LAMBDA),
		          std::normal_distribution<Interpolate>(DEFAULT_AVERAGE_DELTA, DEFAULT_VARIANCE_DELTA),
		          AlphaGenerator(),
		          n,
		          t,
		          a,
		          seed,
		          eta,
		          u,
		          theta) {}

		template <class ThetaDistribution, class LambdaDistribution, class DeltaDistribution, class AlphaGenerator_>
		Problem(
		    const std::filesystem::path &data_path,
		    ThetaDistribution &&theta_distribution,
		    LambdaDistribution &&lambda_distribution,
		    DeltaDistribution &&delta_distribution,
		    AlphaGenerator_ &&alpha_generator,
		    const std::size_t n = 1,
		    const std::size_t t = 1000,
		    const double a = 2,
		    const std::uint_fast64_t seed = 0,
		    const Output eta = 200,
		    const Output u = 50,
		    const Output theta = Output{1} / 2) : _t(t), _a(a), _u(u), _eta(eta), _theta(theta), _alpha_generator(std::forward<AlphaGenerator_>(alpha_generator)) {
			static_assert(std::is_same_v<AlphaGenerator, std::remove_cv_t<std::remove_reference_t<AlphaGenerator_>>>);
			_images.reserve(n * N);
			_labels.reserve(n * N);

			load_mnist_image(TRAIN_SET_N, data_path / TRAINING_SET_IMAGE_FILE_PATH);
			load_mnist_image(TEST_SET_N, data_path / TEST_SET_IMAGE_FILE_PATH);
			load_mnist_label(TRAIN_SET_N, data_path / TRAINING_SET_LABEL_FILE_PATH);
			load_mnist_label(TEST_SET_N, data_path / TEST_SET_LABEL_FILE_PATH);
			augment_images(
			    std::forward<ThetaDistribution>(theta_distribution),
			    std::forward<LambdaDistribution>(lambda_distribution),
			    std::forward<DeltaDistribution>(delta_distribution), n, seed);

#ifdef TSNE_INPUT_RESIZE
			_images.resize(TSNE_INPUT_RESIZE);
			_labels.resize(TSNE_INPUT_RESIZE);
#endif
		}

		const auto &images() const &noexcept {
			return _images;
		}

		const auto &labels() const &noexcept {
			return _labels;
		}

		auto alpha(const std::size_t t) const {
			return _alpha_generator(t);
		}

		auto t() const noexcept {
			return _t;
		}

		auto a() const noexcept {
			return _a;
		}

		auto u() const noexcept {
			return _u;
		}

		auto eta() const noexcept {
			return _eta;
		}

		auto theta() const noexcept {
			return _theta;
		}

		/// @brief 平均0,標準偏差1E-4の乱数によってyの初期値を与える.
		static inline auto default_y(const std::size_t n, const std::uint_fast64_t seed = 1) {
			auto y = std::vector<embedding_t>();
			y.reserve(n);
			auto engine = std::mt19937_64(seed);
			auto distribution = std::normal_distribution<Output>(0, Output{1} / 10000);
			for (auto i = decltype(n){0}; i < n; ++i) {
				y.push_back({distribution(engine), distribution(engine)});
			}
			return y;
		}

#ifdef ENABLE_TEST
		friend struct ProblemTest;
#endif

	  private:
		/// @brief 非ゼロ値が回転/拡縮/平行移動によって28x28の範囲から飛び出したらstd::nullopt,そうでなければimage
		static std::optional<image_t> augment_image(const image_t &base_image, const Interpolate theta, const Interpolate lambda_x, const Interpolate lambda_y, const Interpolate delta_x, const Interpolate delta_y) {
			constexpr auto MAX_X = (Interpolate{X_SIZE} - 1) / 2;
			constexpr auto MIN_X = -MAX_X;
			constexpr auto MAX_Y = (Interpolate{Y_SIZE} - 1) / 2;
			constexpr auto MIN_Y = -MAX_Y;
			auto transformed_position = std::array<std::pair<Interpolate, Interpolate>, IMAGE_SIZE>();
			auto transformed_image = image_t();
			for (auto j = decltype(Y_SIZE){0}; j < Y_SIZE; ++j) {
				for (auto i = decltype(X_SIZE){0}; i < X_SIZE; ++i) {
					const auto value = base_image[j * X_SIZE + i];
					const auto x = static_cast<Interpolate>(i) + MIN_X;
					const auto y = static_cast<Interpolate>(j) + MIN_Y;
					const auto transformed_x = x * lambda_x * std::cos(theta) - y * lambda_y * std::sin(theta) + delta_x;
					const auto transformed_y = x * lambda_x * std::sin(theta) + y * lambda_y * std::cos(theta) + delta_y;
					if (value != 0 && (transformed_x < MIN_X || transformed_x > MAX_X || transformed_y < MIN_Y || transformed_y > MAX_Y)) {
						return std::nullopt;
					}
					transformed_position[j * X_SIZE + i] = {transformed_x - MIN_X, transformed_y - MIN_Y};
				}
			}
			for (auto j = decltype(Y_SIZE){0}; j < Y_SIZE - 1; ++j) {
				for (auto i = decltype(X_SIZE){0}; i < X_SIZE - 1; ++i) {
					// trans(i, j), trans(i+1, j+1), trans(i, j+1), trans(i+1,
					// j+1)の四角形の中にある格子点の値を線形補完で求める.
					const auto di =
					    (theta < 0) ? std::array<std::size_t, 4>({0, 1, 1, 0})
					                : std::array<std::size_t, 4>({0, 1, 0, 1});
					const auto dj =
					    (theta < 0) ? std::array<std::size_t, 4>({0, 1, 0, 1})
					                : std::array<std::size_t, 4>({1, 0, 0, 1});
					const auto min_x = transformed_position[(j + dj[0]) * X_SIZE + (i + di[0])];
					const auto max_x = transformed_position[(j + dj[1]) * X_SIZE + (i + di[1])];
					const auto min_y = transformed_position[(j + dj[2]) * X_SIZE + (i + di[2])];
					const auto max_y = transformed_position[(j + dj[3]) * X_SIZE + (i + di[3])];
					const auto min_x_value = base_image[(j + dj[0]) * X_SIZE + (i + di[0])];
					const auto max_x_value = base_image[(j + dj[1]) * X_SIZE + (i + di[1])];
					const auto min_y_value = base_image[(j + dj[2]) * X_SIZE + (i + di[2])];
					const auto max_y_value = base_image[(j + dj[3]) * X_SIZE + (i + di[3])];
					if (min_x_value == 0 && max_x_value == 0 && min_y_value == 0 && max_y_value == 0)
						continue;
					for (auto y = std::max(static_cast<std::size_t>(std::floor(min_y.second)), std::size_t{0});
					     y <= std::min(static_cast<std::size_t>(std::ceil(max_y.second)), Y_SIZE - 1);
					     ++y) {
						for (auto x = std::max(static_cast<std::size_t>(std::floor(min_x.first)), std::size_t{0});
						     x <= std::min(static_cast<std::size_t>(std::ceil(max_x.first)), X_SIZE - 1);
						     ++x) {
							// min_x - min_y間を通る直線に(x,y)から垂線を下した時の内分比率を求める.
							const auto minx_miny = subtract(min_y, min_x);
							const auto p = inner_product(subtract({x, y}, min_x), minx_miny) / inner_product(minx_miny, minx_miny);
							// min_x - max_y間を通る直線に(x,y)から垂線を下した時の内分比率を求める.
							const auto minx_maxy = subtract(max_y, min_x);
							const auto q = inner_product(subtract({x, y}, min_x), minx_maxy) / inner_product(minx_maxy, minx_maxy);
							// p,qが[0,1]の範囲に収まらない場合は格子点が四角形の中にないので計算しない
							if (p < 0 || p > 1 || q < 0 || q > 1)
								continue;
							transformed_image[y * X_SIZE + x] = static_cast<typename image_t::value_type>(std::round(
							    lerp(
							        lerp(static_cast<Interpolate>(min_x_value), static_cast<Interpolate>(min_y_value), p),
							        lerp(static_cast<Interpolate>(max_y_value), static_cast<Interpolate>(max_x_value), p),
							        q)));
						}
					}
				}
			}
			return transformed_image;
		}

		static constexpr auto subtract(const std::pair<Interpolate, Interpolate> &a, const std::pair<Interpolate, Interpolate> &b) noexcept {
			return std::make_pair(a.first - b.first, a.second - b.second);
		}

		static constexpr auto inner_product(const std::pair<Interpolate, Interpolate> &a, const std::pair<Interpolate, Interpolate> &b) noexcept {
			return a.first * b.first + a.second * b.second;
		}

		/// @brief 線形補完
		static constexpr auto lerp(Interpolate a, Interpolate b, Interpolate t) noexcept {
			return a + t * (b - a);
		}

		void load_mnist_image(const std::size_t size, const std::filesystem::path &path) {
			using namespace std::literals::string_literals;
			auto image_buffer = std::array<std::byte, IMAGE_SIZE>();
			auto ifs = std::ifstream(path, std::ios_base::in | std::ios_base::binary);
			if (ifs.fail()) {
				throw std::ifstream::failure("at "s + __FILE__ + " line " + std::to_string(__LINE__) + " failed to open " + path.string());
			}
			ifs.ignore(16);
			if (ifs.fail()) {
				throw std::ifstream::failure("at "s + __FILE__ + " line " + std::to_string(__LINE__) + " failed to open " + path.string());
			}
			for (auto i = decltype(size){0}; i < size; ++i) {
				ifs.read(reinterpret_cast<char *>(image_buffer.data()), IMAGE_SIZE);
				if (ifs.fail()) {
					throw std::ifstream::failure("at "s + __FILE__ + " line " + std::to_string(__LINE__) + " failed to open " + path.string());
				}
				auto &image = _images.emplace_back();
				for (auto p = decltype(IMAGE_SIZE){0}; p < IMAGE_SIZE; ++p) {
					image[p] = static_cast<typename image_t::value_type>(image_buffer[p]);
				}
			}
			ifs.close();
		}

		void load_mnist_label(std::size_t size, const std::filesystem::path &path) {
			using namespace std::literals::string_literals;
			auto label_buffer = std::byte{};
			auto ifs = std::ifstream(path, std::ios_base::in | std::ios_base::binary);
			if (ifs.fail()) {
				throw std::ifstream::failure("at "s + __FILE__ + " line " + std::to_string(__LINE__) + " failed to open " + path.string());
			}
			ifs.ignore(8);
			if (ifs.fail()) {
				throw std::ifstream::failure("at "s + __FILE__ + " line " + std::to_string(__LINE__) + " failed to open " + path.string());
			}
			for (auto i = decltype(size){0}; i < size; ++i) {
				ifs.read(reinterpret_cast<char *>(&label_buffer), 1);
				if (ifs.fail()) {
					throw std::ifstream::failure("at "s + __FILE__ + " line " + std::to_string(__LINE__) + " failed to open " + path.string());
				}
				_labels.push_back(std::to_integer<label_t>(label_buffer));
			}
			ifs.close();
		}

		/// @brief 拡縮,回転,平行移動によって画像データを増幅させる.
		template <class ThetaDistribution, class LambdaDistribution, class DeltaDistribution>
		void augment_images(ThetaDistribution &&theta_distribution,
		                    LambdaDistribution &&lambda_distribution,
		                    DeltaDistribution &&delta_distribution,
		                    const std::size_t n,
		                    const std::uint_fast64_t seed) {
			auto engine = std::mt19937_64(seed);
			for (auto i = decltype(n){1}; i < n; ++i) {
				for (auto j = decltype(N){0}; j < N; ++j) {
					const auto &base_image = _images[j];
					while (true) {
						const auto theta = static_cast<Interpolate>(theta_distribution(engine));
						const auto lambda_x = static_cast<Interpolate>(lambda_distribution(engine));
						const auto lambda_y = static_cast<Interpolate>(lambda_distribution(engine));
						const auto delta_x = static_cast<Interpolate>(delta_distribution(engine));
						const auto delta_y = static_cast<Interpolate>(delta_distribution(engine));
						// lambda_xとlambda_yの値が大きく異なる場合は再度パラメータを生成しなおす.
						if (lambda_x < MIN_LAMBDA ||
						    lambda_y < MIN_LAMBDA ||
						    lambda_x > LAMBDA_RATE_MAX * lambda_y ||
						    lambda_y > LAMBDA_RATE_MAX * lambda_x) {
							continue;
						}
						// 各ピクセルごとに拡縮/回転/移動操作を行う.
						// 非ゼロ値のピクセルが28x28の枠の中に収まりきらなかった場合は再度パラメータを生成しなおす.
						// 線形補完で画像を生成.
						if (auto transformed_image = augment_image(base_image, theta, lambda_x, lambda_y, delta_x, delta_y)) {
							_images.emplace_back(std::move(*transformed_image));
							_labels.push_back(_labels[j]);
							// 画像の生成に成功した場合は次へ
							break;
						}
					}
				}
			}
		}

		// input images
		std::vector<image_t> _images;
		std::vector<label_t> _labels;
		std::size_t _t;
		double _a;
		Output _u;
		Output _eta;
		Output _theta;
		AlphaGenerator _alpha_generator;
	};

	/// @brief 現在のiterationを渡すとalphaの値を返す関数オブジェクト
	template <class Output>
	struct DefaultAlphaGenerator {
		constexpr auto operator()(const std::size_t t) const noexcept {
			if (t < 250) {
				return Output{1} / 2;
			} else {
				return Output{4} / 5;
			}
		}
	};
} // namespace t_sne

#endif // T_SNE_PROBLEM_HPP
