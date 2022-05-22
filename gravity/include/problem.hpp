#ifndef GRAVITY_PROBLEM_HPP
#define GRAVITY_PROBLEM_HPP

#include <random>
#include <stdexcept>
#include <vector>

namespace gravity {
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

	/// @brief 3次元の直交座標ベクトルを表現する
	template <class T>
	struct Vector {
		/// @brief 型Tのデフォルト値でベクトルを構築する
		explicit constexpr Vector() noexcept : _x{}, _y{}, _z{} {}
		/// @brief x,y,z座標の値からベクトルを構築する
		constexpr Vector(T x, T y, T z) noexcept : _x{x}, _y{y}, _z{z} {}

		/// @brief x座標の値を返す.
		constexpr auto x() const noexcept {
			return _x;
		}

		/// @brief y座標の値を返す.
		constexpr auto y() const noexcept {
			return _y;
		}

		/// @brief z座標の値を返す.
		constexpr auto z() const noexcept {
			return _z;
		}

		/// @brief x座標の値を返す.
		constexpr auto &x() noexcept {
			return _x;
		}

		/// @brief y座標の値を返す.
		constexpr auto &y() noexcept {
			return _y;
		}

		/// @brief z座標の値を返す.
		constexpr auto &z() noexcept {
			return _z;
		}

		/// @brief Vector<U>への明示的キャスト
		template <class U>
		explicit operator Vector<U>() const {
			return Vector<U>{static_cast<U>(_x), static_cast<U>(_y), static_cast<U>(_z)};
		}

		/// @brief 加算
		template <class U>
		constexpr auto operator+(const Vector<U> &rhs) const noexcept {
			return Vector<decltype(_x + rhs.x())>{_x + rhs.x(), _y + rhs.y(), _z + rhs.z()};
		}

		/// @brief 減算
		template <class U>
		constexpr auto operator-(const Vector<U> &rhs) const noexcept {
			return Vector<decltype(_x - rhs.x())>{_x - rhs.x(), _y - rhs.y(), _z - rhs.z()};
		}

		/// @brief スカラ乗算
		template <class U>
		constexpr auto operator*(const U c) const noexcept {
			return Vector<decltype(_x * c)>{_x * c, _y * c, _z * c};
		}

		/// @brief スカラ除算
		template <class U>
		constexpr auto operator/(const U c) const {
			return Vector<decltype(_x / c)>{_x / c, _y / c, _z / c};
		}

		/// @brief 加算代入
		constexpr auto &operator+=(const Vector &rhs) noexcept {
			_x += rhs.x();
			_y += rhs.y();
			_z += rhs.z();
			return *this;
		}

		/// @brief 二乗ノルムの二乗
		constexpr auto norm2() const noexcept {
			return _x * _x + _y * _y + _z * _z;
		}

		/// @brief 球面座標への変換
		auto to_polar() const {
			const auto r = std::sqrt(norm2());
			const auto theta = std::abs(r) < std::numeric_limits<T>::min() ? T{0} : std::acos(_z / r);
			const auto r__x_y = std::sqrt(_x * _x + _y * _y);
			const auto phi = (_y < 0 ? -1 : 1) * (r__x_y < std::numeric_limits<T>::min() ? T{0} : std::acos(_x / r__x_y));
			return std::make_tuple(r, theta, phi);
		}

		/// @brief x,y,zごとの最小値
		static auto direction_wise_min(const std::vector<Vector> &vectors) {
			auto min = Vector{std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), std::numeric_limits<T>::max()};
			for (const auto &vector : vectors) {
				min._x = std::min(min._x, vector._x);
				min._y = std::min(min._y, vector._y);
				min._z = std::min(min._z, vector._z);
			}
			return min;
		}

		/// @brief x,y,zごとの最大値
		static auto direction_wise_max(const std::vector<Vector> &vectors) {
			auto max = Vector{std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest()};
			for (const auto &vector : vectors) {
				max._x = std::max(max._x, vector._x);
				max._y = std::max(max._y, vector._y);
				max._z = std::max(max._z, vector._z);
			}
			return max;
		}

	  private:
		T _x;
		T _y;
		T _z;
	};

	/// @brief スカラ乗算
	template <class T, class U>
	auto operator*(const T c, const Vector<U> &v) {
		using R = decltype(c * v.x());
		return Vector<R>{c * v.x(), c * v.y(), c * v.z()};
	}

	/// @brief 入力データを生成する.
	/// @details 各型パラメータは以下の物理量に対応する.
	/// @tparam M 質量
	/// @tparam T 時間
	/// @tparam X 位置
	/// @tparam V 速度
	/// @tparam A 加速度
	/// @tparam G 万有引力定数
	template <class M, class T, class X, class V, class A, class G>
	struct Problem final {
		/// @brief Octreeで管理する質点の外側のPADDING
		static constexpr auto OCTREE_WIDTH_PADDING = static_cast<X>(1E-5L);

		/// @brief 質点数,セルに含まれる質点数の閾値,ソフトニングパラメータ等からProblemを構築する.
		/// @exception 質点数N,閾値sが0だった場合はstd::runtime_error例外が送出される
		explicit Problem(const std::size_t N,
		                 const std::size_t s,
		                 const std::optional<X> epsilon = std::nullopt,
		                 const G G_ = 1,
		                 const M M_ = 1,
		                 const X R = 1,
		                 const unsigned p = 4,
		                 const std::uint_fast64_t seed = 0) : _N(N), _epsilon(epsilon.value_or(calc_epsilon(N))), _G(G_), _M(M_), _R(R), _p(p), _s(s), _seed(seed) {
			if (N == 0) {
				throw std::runtime_error("N must not be 0");
			}
			if (s == 0) {
				throw std::runtime_error("s must not be 0");
			}
		}

		/// @brief 質点の位置と速度を初期化する
		auto initialize(std::vector<Vector<X>> &x, std::vector<Vector<V>> &v) const {
			auto engine = std::mt19937_64(_seed);
			auto distribution_position = std::uniform_real_distribution<X>(0, X{1} - _epsilon);
			auto standard_uniform_distribution_position = std::uniform_real_distribution<X>(0, 1);
			auto standard_uniform_distribution_velocity = std::uniform_real_distribution<V>(0, 1);
			x.clear();
			v.clear();
			x.reserve(_N);
			v.reserve(_N);
			const auto p_max = P_MAX();
			for (auto i = decltype(_N){0}; i < _N; ++i) {
				const auto x_0 = distribution_position(engine);
				const auto r_i = r(x_0);
				const auto v_e_i = v_e(r_i);
				const auto q_i = q(p_max, standard_uniform_distribution_velocity, engine);
				const auto v_i = q_i * v_e_i;
				const auto x_3 = standard_uniform_distribution_position(engine);
				const auto x_4 = standard_uniform_distribution_position(engine);
				x.emplace_back(calc_xyz<X>(r_i, x_3, x_4));
				const auto x_5 = standard_uniform_distribution_velocity(engine);
				const auto x_6 = standard_uniform_distribution_velocity(engine);
				v.emplace_back(calc_xyz<V>(v_i, x_5, x_6));
			}
		}

		/// @brief 質点数を返す
		auto N() const noexcept { return _N; }

		/// @brief ソフトニングパラメータを返す
		auto epsilon() const noexcept { return _epsilon; }

		/// @brief 万有引力定数を返す
		auto get_G() const noexcept { return _G; }

		/// @brief 各質点の質量を返す
		auto m() const noexcept { return _M / _N; }

		/// @brief プラマー半径を返す
		auto R() const noexcept { return _R; }

		/// @brief 展開次数を返す
		auto p() const noexcept { return _p; }

		/// @brief 一つのセルで管理する質点の数の閾値を返す
		auto s() const noexcept { return _s; }

		/// @brief delta_tを計算する.
		auto delta_t() const {
#ifdef GRAVITY_SEARCH_DELTA_T
			// benchmark.make_t_set() から指定しやすいように1を返す
			return T{1};
#else
			const auto max_v = static_cast<V>(std::sqrt(2));
			return static_cast<T>(_epsilon / max_v);
#endif
		}

#ifdef ENABLE_TEST
		friend struct ProblemTest;
#endif

	  private:
		static auto r(const X x_0) {
			return std::pow(x_0, X{1} / 3) / std::pow(X{1} - std::pow(x_0, X{2} / 3), X{1} / 2);
		}

		static auto v_e(const X r) {
			return static_cast<V>(std::sqrt(X{2}) * std::pow(r * r + 1, X{-1} / 4));
		}

		static auto q(const V P_MAX, std::uniform_real_distribution<V> &distribution, std::mt19937_64 &engine) {
			while (true) {
				const auto x_1 = distribution(engine);
				const auto x_2 = distribution(engine);
				if (P_MAX * x_1 < P(x_2)) {
					return x_2;
				}
			}
		}

		static auto P(const V q) {
			return q * q * std::pow(V{1} - q * q, V{7} / 2);
		}

		// normをx,y,z成分に分解する.座標,速度の分解に用いる.
		// x = (1 - 2r)*norm
		// y = sqrt(norm^2 - x^2)cos(2\pi X_)
		// z = sqrt(norm^2 - x^2)sin(2\pi X_)
		template <class U>
		static auto calc_xyz(const U norm, const U r, const U X_) {
			const auto x = (U{1} - r * 2) * norm;
			// sqrt(y^2 + z^2) = sqrt(r^2 - x^2)
			const auto yz2 = std::sqrt(norm * norm - x * x);
			const auto y = yz2 * std::cos(Constant<U>::PI * 2 * X_);
			const auto z = yz2 * std::sin(Constant<U>::PI * 2 * X_);
			return Vector<U>{x, y, z};
		}

		static auto P_MAX() {
			return std::sqrt(V{7}) * 686 / 19683;
		}

		static auto calc_epsilon(const std::size_t N) {
			return X{63} / 100 * std::pow(static_cast<X>(N), X{-22} / 100);
		}

		std::size_t _N;
		X _epsilon;
		G _G;
		M _M;
		X _R;
		unsigned _p;
		std::size_t _s;
		std::uint_fast64_t _seed;
	};

} // namespace gravity

#endif // GRAVITY_PROBLEM_HPP
