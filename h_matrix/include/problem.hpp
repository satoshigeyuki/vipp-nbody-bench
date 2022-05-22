#ifndef H_MATRIX_PROBLEM_HPP
#define H_MATRIX_PROBLEM_HPP

#include <cmath>
#include <matrix.hpp>
#include <random>

namespace h_matrix {
	/// @brief 実装に利用する定数を特殊化によって提供する
	template <class T>
	struct Constant;
	/// @brief 実装に利用する定数を特殊化によって提供する
	template <>
	struct Constant<float> {
		/// @brief 円周率
		static constexpr auto PI = 3.1415926F;
	};
	/// @brief 実装に利用する定数を特殊化によって提供する
	template <>
	struct Constant<double> {
		/// @brief 円周率
		static constexpr auto PI = 3.141592653589793;
	};
	/// @brief 実装に利用する定数を特殊化によって提供する
	template <>
	struct Constant<long double> {
		/// @brief 円周率
		static constexpr auto PI = 3.1415926535897932384626433832795028L;
	};

	/// @brief 入力データを生成する
	template <class T>
	struct Problem {
		/// @brief 分割数,直径,高さから入力データを生成する.円柱の高さは側面メッシュが正方形になるようにHより大きくなりうる.
		Problem(const std::size_t N = 6,
		        const T D = 1,
		        const T H = 5,
		        const std::uint_fast64_t seed = 0) : _d{D}, _n{N}, _seed{seed} {
			_l = D * Constant<T>::PI / N;
			_nh = static_cast<std::size_t>(std::ceil(H / _l));
			_h = _nh * _l;
		}

		/// @brief 円柱の直径
		auto d() const noexcept {
			return _d;
		}

		/// @brief 円柱の高さ
		auto h() const noexcept {
			return _h;
		}

		/// @brief メッシュの1辺の長さ
		auto l() const noexcept {
			return _l;
		}

		/// @brief 分割数
		auto n() const noexcept {
			return _n;
		}

		/// @brief 円柱の側面メッシュの段数
		auto nh() const noexcept {
			return _nh;
		}

		/// @brief Hとxのサイズを取得
		auto matrix_size() const noexcept {
			return _n * (_nh + 2);
		}

		/// @brief 行列Hを初期化する
		/// @details 入力行列の要素の一部は二重積分になるが解析的には解けない.RAWオプションを無効にすると内側の積分を解析的に解いた結果から外側の積分を数値積分する.
		/// RAWオプションを有効化することで二重積分の両方を数値的に計算する.
		void initialize_H(Matrix<T> &matrix) const {
			// 分割数
			const auto n_subdivide = 1000;
			const auto pi = Constant<T>::PI;
			if (matrix.msize() != matrix_size() || matrix.nsize() != matrix_size()) {
				throw std::runtime_error("matrix size not matched");
			}
			// pi/N
			const auto piN = pi / _n;
			// 下底面同士
#pragma omp parallel for
			for (auto i = decltype(_n){0}; i < _n; ++i) {
				for (auto j = decltype(_n){0}; j < _n; ++j) {
					matrix(i, j) = T{0};
				}
			}
			// 上底面同士
#pragma omp parallel for
			for (auto i = matrix_size() - _n; i < matrix_size(); ++i) {
				for (auto j = matrix_size() - _n; j < matrix_size(); ++j) {
					matrix(i, j) = T{0};
				}
			}
			// 下底面-側面
			for (auto j = _n; j < matrix_size() - _n; ++j) {
				for (auto i = decltype(_n){0}; i < _n; ++i) {
					if (j % _n == 0) {
#ifdef RAW
						const auto gx = _n * _d * std::sin(piN) * std::cos(piN * 2 * i) / 3 / pi;
						const auto gy = _n * _d * std::sin(piN) * std::sin(piN * 2 * i) / 3 / pi;
						const auto gz = -_h / 2;
						const auto f = [this, gx, gy, gz](const auto theta, const auto z) {
							const auto x = _d * std::cos(theta) / 2;
							const auto y = _d * std::sin(theta) / 2;
							const auto rx = x - gx;
							const auto ry = y - gy;
							const auto rz = z - gz;
							return -rz / std::pow(rx * rx + ry * ry + rz * rz, T{3} / 2);
						};
						matrix(i, j) = midpoint_rule2(-piN, piN, (static_cast<T>(j / _n) - static_cast<T>(_nh) / 2 - 1) * _l, (static_cast<T>(j / _n) - static_cast<T>(_nh) / 2) * _l, n_subdivide, f) / (-pi * 4);
#else
						const auto f = [this, piN, i, j](const auto x) {
							const auto t = std::sin(piN) / 3 / piN;
							const auto a = _h * _h / 4 + _d * _d * (T{1} / 4 + t * t - t * std::cos(x - piN * 2 * i));
							const auto b = (j / _n - static_cast<T>(_nh) / 2) * _l;
							return T{1} / std::sqrt((b + _h) * b + a) - T{1} / std::sqrt((b - _l + _h) * (b - _l) + a);
						};
						matrix(i, j) = midpoint_rule(-piN, piN, n_subdivide, f) / (-pi * 4);
#endif
					} else {
						matrix(i, j) = matrix((i + _n - 1) % _n, j - 1);
					}
				}
			}
			// 上底面-側面
#pragma omp parallel for
			for (auto i = matrix_size() - _n; i < matrix_size(); ++i) {
				for (auto j = _n; j < matrix_size() - _n; ++j) {
					matrix(i, j) = matrix(matrix_size() - 1 - i, matrix_size() - 1 - j);
				}
			}
			// 下底面-上底面
			for (auto j = matrix_size() - _n; j < matrix_size(); ++j) {
				for (auto i = decltype(_n){0}; i < _n; ++i) {
					if (j % _n == 0) {
#ifdef RAW
						const auto gx = _n * _d * std::sin(piN) * std::cos(piN * 2 * i) / 3 / pi;
						const auto gy = _n * _d * std::sin(piN) * std::sin(piN * 2 * i) / 3 / pi;
						const auto f = [this, gx, gy](const auto theta, const auto r) {
							const auto x = r * std::cos(theta);
							const auto y = r * std::sin(theta);
							const auto rx = x - gx;
							const auto ry = y - gy;
							const auto rz = _h;
							return -rz * r / std::pow(rx * rx + ry * ry + rz * rz, T{3} / 2);
						};
						matrix(i, j) = midpoint_rule2(-piN, piN, 0, _d / 2, n_subdivide, f) / (-pi * 4);
#else
						const auto f = [this, piN, i](const auto x) {
							const auto t = _d * std::sin(piN) / 3 / piN;
							const auto a = t * 2 * std::cos(x - piN * 2 * i);
							const auto b = t * t + _h * _h;
							const auto c = _h * _h * 4 + std::pow(t * 2 * std::sin(x - piN * 2 * i), 2);
							return static_cast<T>((a * _d - 4 * b) * 2 / c / std::sqrt(_d * (_d - 2 * a) + 4 * b) + std::sqrt(b) * 4 / c);
						};
						matrix(i, j) = _h / pi / 4 * midpoint_rule(-piN, piN, n_subdivide, f);
#endif
					} else {
						matrix(i, j) = matrix((i + _n - 1) % _n, j - 1);
					}
				}
			}
			// 上底面-下底面
#pragma omp parallel for
			for (auto i = matrix_size() - _n; i < matrix_size(); ++i) {
				for (auto j = decltype(_n){0}; j < _n; ++j) {
					matrix(i, j) = matrix(matrix_size() - 1 - i, matrix_size() - 1 - j);
				}
			}
			// 側面-下底面
			for (auto j = decltype(_n){0}; j < _n; ++j) {
				for (auto i = _n; i < matrix_size() - _n; ++i) {
					if (j % _n == 0) {
#ifdef RAW
						const auto gx = _n * _d * std::sin(piN) * std::cos(piN * 2 * i) / 2 / pi;
						const auto gy = _n * _d * std::sin(piN) * std::sin(piN * 2 * i) / 2 / pi;
						const auto gz = (static_cast<T>(i / _n) - static_cast<T>(_nh + 1) / 2) * _l;
						const auto f = [this, gx, gy, gz, i, piN](const auto theta, const auto r) {
							const auto x = r * std::cos(theta);
							const auto y = r * std::sin(theta);
							const auto z = -_h / 2;
							const auto rx = x - gx;
							const auto ry = y - gy;
							const auto rz = z - gz;
							return (rx * std::cos(piN * 2 * i) + ry * std::sin(piN * 2 * i)) * r / std::pow(rx * rx + ry * ry + rz * rz, T{3} / 2);
						};
						matrix(i, j) = midpoint_rule2(-piN, piN, 0, _d / 2, n_subdivide, f) / (-pi * 4);
#else
						const auto f = [this, piN, i](const auto x) {
							const auto a = std::cos(x - piN * 2 * i);
							const auto b = _d * std::sin(piN) / 2 / piN;
							const auto c = b * b + std::pow(_h / 2 + (i / _n - static_cast<T>(_nh + 1) / 2) * _l, 2);
							const auto d = std::pow(b * std::sin(x - piN * 2 * i), 2) + std::pow(_h / 2 + (i / _n - static_cast<T>(_nh + 1) / 2) * _l, 2);
							return static_cast<T>(a * (std::asinh((_d - 2 * a * b) / 2 / std::sqrt(d)) - std::asinh(-a * b / std::sqrt(d))) + (a * _d * (-c + b * b * (2 * a * a - 1)) + 2 * b * c * (1 - a * a)) / d / std::sqrt(_d * (_d - 4 * a * b) + 4 * c) - b * std::sqrt(c) * (1 - a * a) / d);
						};
						matrix(i, j) = midpoint_rule(-piN, piN, n_subdivide, f) / (-pi * 4);
#endif
					} else {
						matrix(i, j) = matrix(i % _n == 0 ? i + _n - 1 : i - 1, j - 1);
					}
				}
			}
			// 側面-上底面
#pragma omp parallel for
			for (auto i = _n; i < matrix_size() - _n; ++i) {
				for (auto j = matrix_size() - _n; j < matrix_size(); ++j) {
					matrix(i, j) = matrix(matrix_size() - 1 - i, matrix_size() - 1 - j);
				}
			}
			// 側面-側面
			for (auto j = _n; j < matrix_size() - _n; ++j) {
				for (auto i = _n; i < matrix_size() - _n; ++i) {
					if (i == j) {
						matrix(i, j) = T{0};
					} else if (j % _n == 0) {
#ifdef RAW
						const auto gx = _n * _d * std::sin(piN) * std::cos(piN * 2 * i) / 2 / pi;
						const auto gy = _n * _d * std::sin(piN) * std::sin(piN * 2 * i) / 2 / pi;
						const auto gz = (static_cast<T>(i / _n) - static_cast<T>(_nh + 1) / 2) * _l;
						const auto f = [this, gx, gy, gz, i, piN](const auto theta, const auto z) {
							const auto x = _d * std::cos(theta) / 2;
							const auto y = _d * std::sin(theta) / 2;
							const auto rx = x - gx;
							const auto ry = y - gy;
							const auto rz = z - gz;
							return (rx * std::cos(piN * 2 * i) + ry * std::sin(piN * 2 * i)) / std::pow(rx * rx + ry * ry + rz * rz, T{3} / 2);
						};
						matrix(i, j) = midpoint_rule2(-piN, piN, (static_cast<T>(j / _n) - static_cast<T>(_nh) / 2 - 1) * _l, (static_cast<T>(j / _n) - static_cast<T>(_nh) / 2) * _l, n_subdivide, f) / (-pi * 4);
#else
						const auto f = [this, piN, i, j](const auto x) {
							const auto t = _d * std::sin(piN) / 2 / piN;
							const auto u = std::cos(x - piN * 2 * i);
							const auto a = _d / 2 * u - t;
							const auto b = (i / _n - static_cast<T>(_nh + 1) / 2) * _l;
							const auto c = b * b + _d * _d / 4 + t * (t - _d * u);
							const auto d = (j / _n - static_cast<T>(_nh) / 2) * _l;
							// c-b^2
							const auto cb2 = _d * _d / 4 + t * (t - _d * u);
							return a * ((d - b) / cb2 / std::sqrt(d * (d - 2 * b) + c) - (d - _l - b) / cb2 / std::sqrt((d - _l) * (d - _l - 2 * b) + c));
						};
						matrix(i, j) = midpoint_rule(-piN, piN, n_subdivide, f) / (-pi * 4);
#endif
					} else {
						matrix(i, j) = matrix(i % _n == 0 ? i + _n - 1 : i - 1, j - 1);
					}
				}
			}
		}

		/// @brief ベクトルxを初期化する
		void initialize_x(Vector<T> &vector) const {
			if (vector.size() != matrix_size()) {
				throw std::runtime_error("vector size not matched");
			}
			auto engine = std::mt19937_64(_seed);
#ifdef H_MATRIX_DISTRIBUTION_A
			const auto a = T{H_MATRIX_DISTRIBUTION_A};
#else
			const auto a = T{0};
#endif
#ifdef H_MATRIX_DISTRIBUTION_B
			const auto b = T{H_MATRIX_DISTRIBUTION_B};
#else
			const auto b = T{1};
#endif
#ifdef H_MATRIX_UNIFORM_DISTRIBUTION
			// 最小値 a, 最大値 b の一様分布
			auto distribution_x = std::uniform_real_distribution<T>(a, b);
#else
			// 平均 a, 分散 b の正規分布
			auto distribution_x = std::normal_distribution<T>(a, b);
#endif
			for (auto i = decltype(matrix_size()){0}; i < matrix_size(); ++i) {
				vector(i) = distribution_x(engine);
			}
		}

#ifdef ENABLE_TEST
		friend struct ProblemTest;
#endif
	  private:
#ifdef RAW
		/// @brief 関数f(x,y)をa < x < b, c < y < dの間でそれぞれn分割して中点則で数値積分する.
		/// @details n != 0, fはTを引数とする関数オブジェクトであることを要求する.
		template <class F>
		static auto midpoint_rule2(const T a, const T b, const T c, const T d, const std::size_t n, const F &f) {
			const auto hx = (b - a) / n;
			const auto hy = (d - c) / n;
			auto result = decltype(f(a, b)){0};
#pragma omp parallel for reduction(+ \
                                   : result)
			for (auto i = decltype(n){0}; i < n; ++i) {
				const auto x = a + hx / 2 * (i * 2 + 1);
				for (auto j = decltype(n){0}; j < n; ++j) {
					const auto y = c + hy / 2 * (j * 2 + 1);
					result += hx * hy * f(x, y);
				}
			}
			return result;
		}
#else
		/// @brief 関数f(x)をa < x < bの間でn分割して中点則で数値積分する.
		/// @details a < b,n != 0,fはTを引数とする関数オブジェクトであることを要求する.
		template <class F>
		static auto midpoint_rule(const T a, const T b, const std::size_t n, const F &f) {
			const auto h = (b - a) / n;
			auto result = decltype(f(a)){0};
#pragma omp parallel for reduction(+ \
                                   : result)
			for (auto i = decltype(n){0}; i < n; ++i) {
				const auto x = a + h / 2 * (i * 2 + 1);
				result += h * f(x);
			}
			return result;
		}
#endif

		T _d;
		T _h;
		T _l;
		std::size_t _n;
		std::size_t _nh;
		std::uint_fast64_t _seed;
	};
} // namespace h_matrix

#endif // H_MATRIX_PROBLEM_HPP
