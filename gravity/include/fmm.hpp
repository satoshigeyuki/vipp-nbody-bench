#ifndef GRAVITY_FMM_HPP
#define GRAVITY_FMM_HPP

#include <octree.hpp>
#include <problem.hpp>

namespace gravity {
	/// @brief 階乗を計算する
	template <class T>
	static constexpr auto factorial(const unsigned n) noexcept {
		auto result = T{1};
		for (auto i = decltype(n){2}; i <= n; ++i) {
			result *= i;
		}
		return result;
	}

	/// @brief ルジャンドルの陪関数P_{n,m}(x)を計算する.
	/// @details 0 <= m <= m_max, m <= n <= n_max, -1 <= x <= 1について計算できる.
	template <class X>
	struct P {
		/// @brief ルジャンドルの陪関数P(x)をP_{n_max,m_max}まで計算する
		explicit P(const unsigned n_max, const unsigned m_max, const X x) : _p(n_max + 1, std::vector<X>(m_max + 1, 0)), _x{x} {
			// (1-x^2)^{m/2}
			auto x1m2 = X{1};
			for (auto m = decltype(m_max){0}; m <= m_max; ++m) {
				// 2^n
				auto n2 = std::pow(2, m);
				for (auto n = m; n <= n_max; ++n) {
					const auto k_max = (n - m) / 2;
					// (-1)^k
					auto sign = X{1};
					// (2n-2k)!
					auto fact_2nk = factorial<X>(2 * n);
					// k!
					auto fact_k = X{1};
					// (n-k)!
					auto fact_nk = factorial<X>(n);
					// (n-2k-m)!
					auto fact_n2km = factorial<X>(n - m);

					for (auto k = decltype(k_max){0}; k <= k_max; ++k) {
						// x^{n-2k-m}
						const auto xn2km = std::pow(x, n - 2 * k - m);
						_p[n][m] += static_cast<X>(sign * fact_2nk * xn2km / (fact_k * fact_nk * fact_n2km));

						sign *= -1;
						fact_2nk /= 2 * (n - k) * (2 * (n - k) - 1);
						fact_k *= k + 1;
						fact_nk /= n - k;
						fact_n2km /= (n - m - 2 * k) * (n - m - 2 * k - 1);
					}

					_p[n][m] *= static_cast<X>(x1m2 / n2 * (m % 2 ? -1 : 1));

					n2 *= 2;
				}
				x1m2 *= std::sqrt(1 - x * x);
			}
		}

		/// @brief P_{n,m}(x)を返す
		auto operator()(const unsigned n, const unsigned m) const {
			if (n < m) {
				throw std::runtime_error("n must greater than or equal to m");
			}
			return _p.at(n).at(m);
		}

	  private:
		std::vector<std::vector<X>> _p;
		X _x;
	};

	/// @brief ルジャンドルの陪関数の導関数の-sin theta倍を計算する(dP_{n,m}(x)/dx *-sin theta)を計算する. ただしx=cos theta
	/// @details 0 <= m <= m_max, m <= n <= n_max, -1 <= x <= 1について計算できる.
	template <class X>
	struct DP {
		/// @brief ルジャンドルの陪関数の導関数dP(x)/dxの-sin theta倍を計算する.
		explicit DP(const unsigned n_max, const unsigned m_max, const X theta) : _dp(n_max + 1, std::vector<X>(m_max + 1, 0)), _theta{theta} {
			const auto x = std::cos(theta);
			const auto y = std::sin(theta);
			{
				// m = 0の場合
				// 2^n
				auto n2 = X{1};
				for (auto n = decltype(n_max){0}; n <= n_max; ++n) {
					const auto k_max = n / 2;
					// (2n-2k)!
					auto fact_2nk = factorial<X>(2 * n);
					// k!
					auto fact_k = X{1};
					// (n-k)!
					auto fact_nk = factorial<X>(n);
					// (n-2k)!
					auto fact_n2km = factorial<X>(n);
					for (auto k = decltype(k_max){0}; k <= k_max; ++k) {
						if (n != k * 2) {
							// x^{n-2k-1}
							const auto xn2k1 = std::pow(x, n - 2 * k - 1);
							_dp[n][0] -= static_cast<X>(y / n2 * (k % 2 ? -1 : 1) * fact_2nk * (n - 2 * k) / (fact_k * fact_nk * fact_n2km) * xn2k1);
						}
						fact_2nk /= 2 * (n - k) * (2 * (n - k) - 1);
						fact_k *= k + 1;
						fact_nk /= n - k;
						fact_n2km /= (n - 2 * k) * (n - 2 * k - 1);
					}

					n2 *= 2;
				}
			}

			// y^{m-1}
			auto ym1 = X{1};
			for (auto m = decltype(m_max){1}; m <= m_max; ++m) {
				// 2^n
				auto n2 = std::pow(2, m);
				for (auto n = m; n <= n_max; ++n) {
					const auto k_max = (n - m) / 2;
					// (2n-2k)!
					auto fact_2nk = factorial<X>(2 * n);
					// k!
					auto fact_k = X{1};
					// (n-k)!
					auto fact_nk = factorial<X>(n);
					// (n-2k-m)!
					auto fact_n2km = factorial<X>(n - m);

					for (auto k = decltype(k_max){0}; k <= k_max; ++k) {
						if (n - m != k * 2) {
							// x^{n-2k-m-1}
							const auto xn2km1 = std::pow(x, n - 2 * k - m - 1);
							_dp[n][m] -= static_cast<X>(ym1 / n2 * (k % 2 ? -1 : 1) * fact_2nk / (fact_k * fact_nk * fact_n2km) * xn2km1 * (-static_cast<X>(m) * x * x + y * y * (n - 2 * k - m)));
						} else {
							// x^{n-2k-m}
							const auto xn2km = std::pow(x, n - 2 * k - m);
							_dp[n][m] -= static_cast<X>((-static_cast<X>(m) * x * ym1) / n2 * (k % 2 ? -1 : 1) * fact_2nk / (fact_k * fact_nk * fact_n2km) * xn2km);
						}
						fact_2nk /= 2 * (n - k) * (2 * (n - k) - 1);
						fact_k *= k + 1;
						fact_nk /= n - k;
						fact_n2km /= (n - m - 2 * k) * (n - m - 2 * k - 1);
					}

					_dp[n][m] *= (m % 2 ? -1 : 1);

					n2 *= 2;
				}

				ym1 *= y;
			}
		}

		/// @brief dP_{n,m}(x)/dx* -sin thetaを返す
		auto operator()(const unsigned n, const unsigned m) const {
			if (n < m) {
				throw std::runtime_error("n must greater than or equal to m");
			}
			return _dp.at(n).at(m);
		}

	  private:
		std::vector<std::vector<X>> _dp;
		X _theta;
	};

	/// @brief 球面調和関数Y_{k,m}(theta,phi)を計算する
	/// @details 0 <= k <= k_max, -k <= m <= kについて計算できる
	template <class X>
	struct Y {
		/// @brief 球面調和関数Y(theta,phi)をY_{k_max, m_max}まで計算する
		explicit Y(const unsigned k_max, const unsigned m_max, const X theta, const X phi) : _p{k_max, m_max, std::cos(theta)}, _phi{phi} {}

		/// @brief Y_{k,m}(theta,phi)を返す.
		auto operator()(const unsigned k, const signed m) const {
			// |m|
			const auto absm = static_cast<unsigned>(m < 0 ? -m : m);
			// P_{k,|m|}(cos\theta)
			const auto p = _p(k, absm);
			// e^{im\phi}
			const auto e = std::complex<X>{std::cos(m * _phi), std::sin(m * _phi)};
			// \sqrt((k-|m|)!/(k+|m|)!)
			auto c = X{1};
			for (auto i = k - absm + 1; i <= k + absm; ++i) {
				c /= i;
			}
			c = std::sqrt(c);
			return c * p * e;
		}

	  private:
		P<X> _p;
		X _phi;
	};

	template <class M, class T, class X, class V, class A, class G>
	struct Cell;

	/// @brief M2MとL2L(モーメントのシフト)で用いる係数を計算する.
	/// @details n - m >= 0 かつ n + m >= 0でなければならない.
	template <class M, class T, class X, class V, class A, class G>
	static auto A_(const unsigned n, const signed m) {
		const auto absm = static_cast<unsigned>(m < 0 ? -m : m);
		if (n < absm) {
			throw std::runtime_error("n must greater than or equal to |m|");
		}
		auto divisor = typename Cell<M, T, X, V, A, G>::momentum_t{1};
		for (auto i = n - absm + 1; i <= n + absm; ++i) {
			divisor *= i;
		}
		divisor = std::sqrt(divisor);
		for (auto i = decltype(n){1}; i <= n - absm; ++i) {
			divisor *= i;
		}

		if (n % 2 == 0) {
			return decltype(divisor){1} / divisor;
		} else {
			return -decltype(divisor){1} / divisor;
		}
	}

	/// @brief P2M:ノードに含まれる質点が作るポテンシャル場を近似する.
	template <class M, class T, class X, class V, class A, class G>
	void p2m(Cell<M, T, X, V, A, G> &cell, const unsigned p, const std::vector<Vector<X>> &x, const M m_i) {
		cell._multipole.resize(p + 1, std::vector<std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>>(2 * p + 1));
		for (const auto &i : cell.points()) {
			const auto point = x[i] - cell.center();
			const auto [rho_i, alpha_i, beta_i] = point.to_polar();
			const auto y = Y<X>{p, p, alpha_i, beta_i};
			auto rho_i_k = decltype(rho_i){1};
			for (auto k = decltype(p){0}; k <= p; ++k) {
				for (auto m = decltype(k){0}; m <= k; ++m) {
					cell._multipole[k][p + m] += static_cast<typename Cell<M, T, X, V, A, G>::momentum_t>(m_i * rho_i_k) * static_cast<std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>>(y(k, -static_cast<signed>(m)));
				}
				rho_i_k *= rho_i;
			}
			for (auto k = decltype(p){1}; k <= p; ++k) {
				for (auto m = decltype(k){1}; m <= k; ++m) {
					cell._multipole[k][p - m] = std::conj(cell._multipole[k][p + m]);
				}
			}
		}
	}

	/// @brief M2M:子ノードの多重極モーメントから親ノードの多重極モーメントを計算する.
	template <class M, class T, class X, class V, class A, class G>
	void m2m(Cell<M, T, X, V, A, G> &cell, const unsigned p, const std::vector<Cell<M, T, X, V, A, G>> &cells) {
		cell._multipole.resize(p + 1, std::vector<std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>>(2 * p + 1));
		for (const auto index : cell.cells()) {
			const auto &child = cells[index];
			// 子ノードの中心座標
			const auto point = child.center() - cell.center();
			const auto [rho_i, alpha_i, beta_i] = point.to_polar();
			const auto y = Y<X>{p, p, alpha_i, beta_i};
			for (auto j = decltype(p){0}; j <= p; ++j) {
				for (auto k = -static_cast<signed>(j); k <= static_cast<signed>(j); ++k) {
					const auto a_j_k = A_<M, T, X, V, A, G>(j, k);
					// rho^n
					auto rho_n = decltype(rho_i){1};
					for (auto n = decltype(j){0}; n <= j; ++n) {
						for (auto m = std::max(-static_cast<signed>(n), static_cast<signed>(n) + k - static_cast<signed>(j)); m <= std::min(static_cast<signed>(j) + k - static_cast<signed>(n), static_cast<signed>(n)); ++m) {
							// O
							const auto child_moment = child.multipole(p, j - n, k - m);
							// i^{|k| - |m| - |k - m|} (虚数にはならない)
							const auto imaginary = static_cast<std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>>(std::pow(std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>{0, 1}, std::abs(k) - std::abs(m) - std::abs(k - m)));
							cell._multipole[j][static_cast<unsigned>(static_cast<signed>(p) + k)] += (child_moment * imaginary * A_<M, T, X, V, A, G>(n, m) * A_<M, T, X, V, A, G>(j - n, k - m) * static_cast<typename Cell<M, T, X, V, A, G>::momentum_t>(rho_n) * static_cast<std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>>(y(n, -m))) / a_j_k;
						}
						rho_n *= rho_i;
					}
				}
			}
		}
	}

	/// @brief M2L:遠方ノードの多重極モーメントから局所モーメントを計算する.
	template <class M, class T, class X, class V, class A, class G>
	void m2l(Cell<M, T, X, V, A, G> &cell, const unsigned p, const std::vector<Cell<M, T, X, V, A, G>> &cells, const std::vector<std::size_t> &approx_interact_indices) {
		cell._local.resize(p + 1, std::vector<std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>>(2 * p + 1));
		for (const auto index : approx_interact_indices) {
			const auto m_cell = cells[index];
			const auto point = m_cell.center() - cell.center();
			const auto [rho_i, alpha_i, beta_i] = point.to_polar();
			const auto y = Y<X>{p * 2, p * 2, alpha_i, beta_i};
			// rho_i^{j+1}
			auto rho_j1 = rho_i;
			for (auto j = decltype(p){0}; j <= p; ++j) {
				for (auto k = -static_cast<signed>(j); k <= static_cast<signed>(j); ++k) {
					const auto a_j_k = A_<M, T, X, V, A, G>(j, k);
					auto rho_n = decltype(rho_i){1};
					for (auto n = decltype(p){0}; n <= p; ++n) {
						for (auto m = -static_cast<signed>(n); m <= static_cast<signed>(n); ++m) {
							const auto mmoment = m_cell.multipole(p, n, m);
							// i^{|k-m|-|k|-|m|}
							const auto imaginary = static_cast<std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>>(std::pow(std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>{0, 1}, std::abs(k - m) - std::abs(k) - std::abs(m)));
							cell._local[j][static_cast<unsigned>(static_cast<signed>(p) + k)] += (mmoment * imaginary * A_<M, T, X, V, A, G>(n, m) * a_j_k * static_cast<std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>>(y(j + n, m - k))) / ((n % 2 ? -1 : 1) * A_<M, T, X, V, A, G>(j + n, m - k) * rho_j1 * rho_n);
						}
						rho_n *= rho_i;
					}
				}
				rho_j1 *= rho_i;
			}
		}
	}

	/// @brief L2L:親ノードの局所モーメントから子ノードの局所モーメントを計算する.
	template <class M, class T, class X, class V, class A, class G>
	void l2l(Cell<M, T, X, V, A, G> &cell, const unsigned p, const std::vector<Cell<M, T, X, V, A, G>> &cells) {
		const auto &parent = cells[cell.parent().value()];
		const auto point = parent.center() - cell.center();
		const auto [rho_i, alpha_i, beta_i] = point.to_polar();
		const auto y = Y<X>{p, p, alpha_i, beta_i};
		for (auto j = decltype(p){0}; j <= p; ++j) {
			for (auto k = -static_cast<signed>(j); k <= static_cast<signed>(j); ++k) {
				const auto a_j_k = A_<M, T, X, V, A, G>(j, k);
				// rho_i^{n-j}
				auto rho_nj = typename Cell<M, T, X, V, A, G>::momentum_t{1};
				for (auto n = j; n <= p; ++n) {
					for (auto m = std::max(-static_cast<signed>(n), k - static_cast<signed>(n - j)); m <= std::min(static_cast<signed>(n), static_cast<signed>(n - j) + k); ++m) {
						const auto parent_moment = parent.local(p, n, m);
						// i^{|m|-|m-k|-|k|}
						const auto imaginary = static_cast<std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>>(std::pow(std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>{0, 1}, std::abs(m) - std::abs(m - k) - std::abs(k)));
						cell._local[j][static_cast<unsigned>(static_cast<signed>(p) + k)] += (parent_moment * imaginary * A_<M, T, X, V, A, G>(n - j, m - k) * a_j_k * static_cast<std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>>(y(n - j, m - k)) * rho_nj) / (((n + j) % 2 ? -1 : 1) * A_<M, T, X, V, A, G>(n, m));
					}
					rho_nj *= rho_i;
				}
			}
		}
	}

	/// @brief L2P:遠方の点から受ける力を局所モーメントを使って計算する
	template <class M, class T, class X, class V, class A, class G>
	void l2p(const Cell<M, T, X, V, A, G> &cell, const unsigned p, const std::vector<Vector<X>> &x, std::vector<Vector<A>> &a) {
		for (const auto i : cell.points()) {
			auto &acceleration = a[i];
			const auto point = x[i] - cell.center();
			const auto [r, theta, phi] = point.to_polar();
			const auto y = Y<X>{p, p, theta, phi};
			const auto p_ = P<X>{p, p, std::cos(theta)};
			const auto dp = DP<X>{p, p, theta};

			auto r_l = typename Cell<M, T, X, V, A, G>::momentum_t{1};
			auto dr = typename Cell<M, T, X, V, A, G>::momentum_t{0};
			auto dtheta = typename Cell<M, T, X, V, A, G>::momentum_t{0};
			auto dphi = typename Cell<M, T, X, V, A, G>::momentum_t{0};
			for (auto l = decltype(p){0}; l <= p; ++l) {
				for (auto m = -static_cast<signed>(l); m <= static_cast<signed>(l); ++m) {
					const auto L_l_m = cell.local(p, l, m);
					const auto y_l_m = static_cast<std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>>(y(l, m));
					const auto absm = static_cast<unsigned>(std::abs(m));
					// \sqrt((l-|m|)!/(l+|m|)!)
					auto c = typename Cell<M, T, X, V, A, G>::momentum_t{1};
					for (auto j = l - absm + 1; j <= l + absm; ++j) {
						c /= j;
					}
					c = std::sqrt(c);
					// e^{im\phi}
					const auto e = std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>{std::cos(m * phi), std::sin(m * phi)};

					dr += (L_l_m * r_l * y_l_m * static_cast<typename Cell<M, T, X, V, A, G>::momentum_t>(l / r)).real();
					dtheta += (L_l_m * r_l * c * static_cast<typename Cell<M, T, X, V, A, G>::momentum_t>(dp(static_cast<unsigned>(l), static_cast<unsigned>(absm))) * e).real();
					dphi += (L_l_m * r_l * y_l_m * std::complex<typename Cell<M, T, X, V, A, G>::momentum_t>{0, static_cast<typename Cell<M, T, X, V, A, G>::momentum_t>(m)}).real();
				}
				r_l *= r;
			}
			const auto cot = static_cast<typename Cell<M, T, X, V, A, G>::momentum_t>(std::cos(theta));
			const auto sit = static_cast<typename Cell<M, T, X, V, A, G>::momentum_t>(std::sin(theta));
			const auto cop = static_cast<typename Cell<M, T, X, V, A, G>::momentum_t>(std::cos(phi));
			const auto sip = static_cast<typename Cell<M, T, X, V, A, G>::momentum_t>(std::sin(phi));
			acceleration.x() += static_cast<A>(sit * cop * dr + cot * cop / r * dtheta - sip / (r * sit) * dphi);
			acceleration.y() += static_cast<A>(sit * sip * dr + cot * sip / r * dtheta + cop / (r * sit) * dphi);
			acceleration.z() += static_cast<A>(cot * dr - sit / r * dtheta);
		}
	}

	/// @brief P2P:近傍の点から受ける力を直接法で計算する
	template <class M, class T, class X, class V, class A, class G>
	void p2p(const Cell<M, T, X, V, A, G> &cell, const std::vector<Vector<X>> &x, std::vector<Vector<A>> &a, const std::vector<Cell<M, T, X, V, A, G>> &cells, const std::vector<std::size_t> &neighbor_indices, G G_, M m_i, X epsilon) {
		for (const auto i : cell.points()) {
			auto &acceleration = a[i];
			for (const auto neighbor_index : neighbor_indices) {
				acceleration += cells[neighbor_index].acceleration_i(G_, m_i, epsilon, x, i, cells);
			}
		}
	}

	/// @brief FMMの際のOctreeのCell
	template <class M, class T, class X, class V, class A, class G>
	struct Cell {
		using point_t = Vector<X>;
		/// 多重極モーメント,局所モーメントの型
		using momentum_t = decltype(std::declval<M>() * std::declval<X>());

		/// @brief Octreeで管理する質点の外側のPADDING
		static constexpr auto WIDTH_PADDING = Problem<M, T, X, V, A, G>::OCTREE_WIDTH_PADDING;

		/// @brief セルが葉ノードである場合のCell::content()が返すvariantのindex
		static constexpr auto INDEX_POINTS = Octree<Cell, point_t>::INDEX_POINTS;
		/// @brief セルが中間ノードである場合のCell::content()が返すvariantのindex
		static constexpr auto INDEX_CELLS = Octree<Cell, point_t>::INDEX_CELLS;

		/// @brief ルートノードの構築
		explicit Cell(const std::vector<point_t> &points) : _index{0},
		                                                    _min{points.size() != 0 ? point_t::direction_wise_min(points) - point_t{WIDTH_PADDING, WIDTH_PADDING, WIDTH_PADDING} : point_t{}},
		                                                    _max{points.size() != 0 ? point_t::direction_wise_max(points) + point_t{WIDTH_PADDING, WIDTH_PADDING, WIDTH_PADDING} : point_t{}} {}

		/// @brief 中間ノードの構築
		Cell(const std::size_t index, const std::size_t parent, const point_t min, const point_t max) : _parent{parent}, _index{index}, _min{min}, _max{max} {}

		/// @brief 親ノードのindexを返す
		const auto &parent() const noexcept {
			return _parent;
		}

		/// @brief ノードのindexを返す
		auto index() const noexcept {
			return _index;
		}

		/// @brief セルの領域の下限を返す
		const auto &min() const noexcept {
			return _min;
		}

		/// @brief セルの領域の上限を返す
		const auto &max() const noexcept {
			return _max;
		}

		/// @brief セルの領域の中心を返す
		auto center() const noexcept {
			return (_min + _max) / 2;
		}

		/// @brief セルの子要素を返す
		const auto &content() const noexcept {
			return _content;
		}

		/// @brief セルの子要素を返す
		auto &content() noexcept {
			return _content;
		}

		/// @brief 葉ノードであるかどうかを返す
		auto is_leaf() const noexcept {
			return _content.index() == INDEX_POINTS;
		}

		/// @return 葉ノードである場合はセルが含む点のindexの集合を返す
		/// @exception 中間ノードである場合はstd::bad_variant_access例外が送出される
		const auto &points() const {
			return std::get<INDEX_POINTS>(_content);
		}

		/// @return 中間ノードである場合はセルが含む子ノードのindexをiterateできるOctreeCellsIndexRangeオブジェクトを返す
		/// @exception 葉ノードである場合はstd::bad_variant_access例外が送出される
		auto cells() const {
			return OctreeCellsIndexRange{std::get<INDEX_CELLS>(_content)};
		}

		/// @brief 同じ親を持つ8個のノードの中でのx,y,z座標上での位置を求める.
		/// @details レベル1より深い階層のノードであることを要求する.
		auto get_children_position(const std::vector<Cell> &cells) const {
			const auto &parent = cells[this->parent().value()];
			// 親の何番目の子供か
			const auto children_index = index() - std::get<INDEX_CELLS>(parent.content());
			return Vector<std::size_t>{children_index & 1,
			                           (children_index & 2) >> 1,
			                           (children_index & 4) >> 2};
		}

		/// @brief 二つのセルが同一か,隣接しているかどうかを判定する.
		/// @details thisとotherは同一levelのノードであることを要求する.
		auto is_adjacent_or_equal_to(const std::vector<Cell> &cells, const Cell &other) const {
			if (parent() == other.parent()) {
				return true;
			}
			auto index_self = get_children_position(cells);
			auto index_other = other.get_children_position(cells);
			auto ancestor_self = std::reference_wrapper<const Cell>(cells[parent().value()]);
			auto ancestor_other = std::reference_wrapper<const Cell>(cells[other.parent().value()]);
			for (auto level = std::size_t{1};; ++level) {
				index_self = (std::size_t{1} << level) * ancestor_self.get().get_children_position(cells) + index_self;
				index_other = (std::size_t{1} << level) * ancestor_other.get().get_children_position(cells) + index_other;
				if (ancestor_self.get().parent() == ancestor_other.get().parent()) {
					return (index_self - index_other).norm2() <= 3;
				}
				ancestor_self = cells[ancestor_self.get().parent().value()];
				ancestor_other = cells[ancestor_other.get().parent().value()];
			}
			return false;
		}

#ifdef ENABLE_TEST
		friend struct FMMTest;
#endif
		/// @brief P2M:ノードに含まれる質点が作るポテンシャル場を近似する.
		friend void p2m<M, T, X, V, A, G>(Cell &cell, const unsigned p, const std::vector<point_t> &x, const M m_i);
		/// @brief M2M:子ノードの多重極モーメントから親ノードの多重極モーメントを計算する.
		friend void m2m<M, T, X, V, A, G>(Cell &cell, const unsigned p, const std::vector<Cell> &cells);
		/// @brief M2L:遠方ノードの多重極モーメントから局所モーメントを計算する.
		friend void m2l<M, T, X, V, A, G>(Cell &cell, const unsigned p, const std::vector<Cell> &cells, const std::vector<std::size_t> &approx_interact_indices);
		/// @brief L2L:親ノードの局所モーメントから子ノードの局所モーメントを計算する.
		friend void l2l<M, T, X, V, A, G>(Cell &cell, const unsigned p, const std::vector<Cell> &cells);
		/// @brief L2P:遠方の点から受ける力を局所モーメントを使って計算する
		friend void l2p<M, T, X, V, A, G>(const Cell &cell, const unsigned p, const std::vector<point_t> &x, std::vector<Vector<A>> &a);
		/// @brief P2P:近傍の点から受ける力を直接法で計算する
		friend void p2p<M, T, X, V, A, G>(const Cell &cell, const std::vector<point_t> &x, std::vector<Vector<A>> &a, const std::vector<Cell> &cells, const std::vector<std::size_t> &neighbor_indices, G G_, M m_i, X epsilon);

	  private:
		/// @brief ノード内の点から点iが受ける加速度を計算する.
		Vector<A> acceleration_i(const G G_, const M m, const X epsilon, const std::vector<point_t> &x, const std::size_t i, const std::vector<Cell> &cells) const {
			auto acceleration = Vector<A>{};
			if (is_leaf()) {
				for (const auto j : points()) {
					if (i == j) {
						continue;
					}
					acceleration += acceleration_ij(x, G_, m, epsilon, i, j);
				}
			} else {
				for (const auto child_index : this->cells()) {
					acceleration += cells[child_index].acceleration_i(G_, m, epsilon, x, i, cells);
				}
			}
			return acceleration;
		}

		/// @brief 点jから点iが受ける加速度を計算する.
		static auto acceleration_ij(const std::vector<point_t> &x,
		                            const G G_,
		                            const M m,
		                            const X epsilon,
		                            const std::size_t i,
		                            const std::size_t j) {
			const auto &xi = x[i];
			const auto &xj = x[j];
			return static_cast<Vector<A>>(G_ * m * (xj - xi) / std::pow((xj - xi).norm2() + epsilon * epsilon, X{3} / 2));
		}

		/// @brief 展開次数pで計算した多重極モーメントM_{k,m}を取得する.
		auto multipole(const unsigned p, const unsigned k, const signed m) const {
			return _multipole.at(k).at(static_cast<unsigned>(static_cast<signed>(p) + m));
		}

		/// @brief 展開次数pで計算した局所モーメントL_{k,m}を取得する.
		auto local(const unsigned p, const unsigned k, const signed m) const {
			return _local.at(k).at(static_cast<unsigned>(static_cast<signed>(p) + m));
		}

		// 親ノードのcellsにおけるindex
		std::optional<std::size_t> _parent;
		// 自分のcellsにおけるindex
		std::size_t _index;
		// ノードの領域の下限
		point_t _min;
		// ノードの領域の上限
		point_t _max;
		// 葉ノードである場合はstd::vector<std::size_t>
		// 子ノードを持つ場合はstd::size_t, std::size_tは子ノードのcellsにおけるindexの開始点
		std::variant<std::vector<std::size_t>, std::size_t> _content;
		// M_{k,m}
		std::vector<std::vector<std::complex<momentum_t>>> _multipole;
		// L_{j,k}
		std::vector<std::vector<std::complex<momentum_t>>> _local;
	};
} // namespace gravity

#endif // GRAVITY_FMM_HPP
