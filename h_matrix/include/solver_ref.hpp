#ifndef H_MATRIX_SOLVER_REF_HPP
#define H_MATRIX_SOLVER_REF_HPP

#include <complex>
#include <lapacke.h>
#include <matrix.hpp>
#include <numeric>
#include <omp.h>
#include <optional>
#include <problem.hpp>

namespace h_matrix {
	/// @brief 特異値分解によって行列を近似する.
	/// @details matrixのmin_i行min_j列目から始まるM行N列の行列を特異値分解する.
	template <class T>
	auto singular_value_decomposition(const Matrix<T> &matrix, const std::size_t min_i, const std::size_t min_j, const std::size_t M, const std::size_t N);

	template <>
	auto singular_value_decomposition(const Matrix<float> &matrix, const std::size_t min_i, const std::size_t min_j, const std::size_t M, const std::size_t N) {
		const auto leading_N = matrix.nsize();
		const auto m = static_cast<int>(M);
		const auto n = static_cast<int>(N);
		auto A = std::vector<float>(M * N);
#pragma omp parallel for
		for (auto i = decltype(M){0}; i < M; ++i)
			std::copy(matrix.data() + (min_i + i) * leading_N + min_j, matrix.data() + (min_i + i) * leading_N + min_j + N, A.data() + i * N);
		auto u = Matrix<float>{M, M};
		auto s = DiagonalMatrix<float>{M, N};
		auto vt = Matrix<float>{N, N};
		auto lwork = std::max(3 * std::min(M, N) + std::max(M, N), 5 * std::min(M, N));
		auto work = std::vector<float>(lwork);
		LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A.data(), n, s.data(), u.data(), m, vt.data(), n, work.data());
		return std::make_tuple(u, s, vt);
	}

	template <>
	auto singular_value_decomposition(const Matrix<double> &matrix, const std::size_t min_i, const std::size_t min_j, const std::size_t M, const std::size_t N) {
		const auto leading_N = matrix.nsize();
		const auto m = static_cast<int>(M);
		const auto n = static_cast<int>(N);
		auto A = std::vector<double>(M * N);
#pragma omp parallel for
		for (auto i = decltype(M){0}; i < M; ++i)
			std::copy(matrix.data() + (min_i + i) * leading_N + min_j, matrix.data() + (min_i + i) * leading_N + min_j + N, A.data() + i * N);
		auto u = Matrix<double>{M, M};
		auto s = DiagonalMatrix<double>{M, N};
		auto vt = Matrix<double>{N, N};
		auto lwork = std::max(3 * std::min(M, N) + std::max(M, N), 5 * std::min(M, N));
		auto work = std::vector<double>(lwork);
		LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A.data(), n, s.data(), u.data(), m, vt.data(), n, work.data());
		return std::make_tuple(u, s, vt);
	}

	/// @brief 階層型低ランク近似の参照実装
	template <class T>
	struct SolverRef {
		const T tolerance;

		/// 特異値の総和が全体の 1 - tolerance 倍に達したところで近似する
		SolverRef(T tolerance) : tolerance(tolerance) {}
		SolverRef() : tolerance(1e-5) {}

		/// @brief xを超えない最大のaの倍数を返す
		static auto rounddown(const std::size_t x, const std::size_t a) {
			return (x / a) * a;
		}

		/// @brief 行列を分割するかを決定し,分割する場合はそのサイズを返す
		static std::optional<std::pair<std::size_t, std::size_t>> get_split_size(const Problem<T> &problem, const std::size_t M, const std::size_t N) {

			// 分割するかどうかの閾値を設定する
#ifdef H_MATRIX_SPLIT_THRESHOLD
			// 分割後の行列サイズが与えられた定数 H_MATRIX_SPLIT_THRESHOLD 以上になるように分割する
			const auto split_threshold = static_cast<std::size_t>(H_MATRIX_SPLIT_THRESHOLD);
#else
			// 行列を分割する閾値を入力された円柱の分割数 n で決める（デフォルト）
			const auto split_threshold = problem.n();
#endif

			if (std::min(M, N) >= split_threshold * 2) {

				// 分割後の行列サイズについて制約を与える
#ifdef H_MATRIX_ROUNDDOWN_BASE
				// 分割後の行列サイズが与えられた定数 H_MATRIX_ROUNDDOWN_BASE になるように分割する
				const auto rounddown_base = static_cast<std::size_t>(H_MATRIX_ROUNDDOWN_BASE);
#else
				// 分割後の行列サイズが入力された円柱の分割数 n の倍数になるように分割する（デフォルト）
				const auto rounddown_base = problem.n();
#endif

				const auto m_split = rounddown(M / 2, rounddown_base);
				const auto n_split = rounddown(N / 2, rounddown_base);
				if (m_split == 0 || n_split == 0) {
					// 分割の制約がサイズを下回るので分割しない
					return std::nullopt;
				}

				return std::make_pair(m_split, n_split);
			}
			return std::nullopt;
		}

		/// @brief 行列のランクを決定する
		auto get_rank(const Problem<T> &, const Matrix<T> &, const DiagonalMatrix<T> &s, const Matrix<T> &) const {
			const auto rank = std::min(s.msize(), s.nsize());
			const auto sum = std::accumulate(s.data(), s.data() + rank, T{0});
			auto part_sum = T{0};
			for (auto i = decltype(rank){0}; i < rank; ++i) {
				part_sum += s(i);
				if (part_sum / sum >= 1 - tolerance) {
					return i + 1;
				}
			}
			return rank;
		}

		/// @brief 対角行列とベクトルの積
		static auto matrix_vector_multiply(const DiagonalMatrix<T> &matrix, const Vector<T> &vector) {
			const auto M = matrix.msize();
			const auto N = matrix.nsize();
			if (N != vector.size()) {
				throw std::runtime_error("Vector size not matched.");
			}
			auto result = Vector<T>{M};
			const auto min = std::min(M, N);
			for (auto i = decltype(min){0}; i < min; ++i) {
				result(i) = matrix(i) * vector(i);
			}
			return result;
		}

		/// @brief 近似行列とベクトルの積
		static auto matrix_vector_multiply(const ApproximateMatrix<T> &matrix, const Vector<T> &vector) {
			const auto M = matrix.msize();
			const auto N = matrix.nsize();
			if (N != vector.size()) {
				throw std::runtime_error("Vector size not matched.");
			}
			auto result = Vector<T>{M};
			for (const auto idx : matrix.leaves()) {
				const auto &node = matrix.node(idx);
				// 部分行列とベクトルの積をベクトルに加算する
				switch (node.index()) {
				case ApproximateMatrix<T>::INDEX_DENSE: {
					const auto &dense = std::get<ApproximateMatrix<T>::INDEX_DENSE>(node);
					matrix_vector_multiply(dense.matrix(), vector, result, dense.min_i(), dense.min_j());
					break;
				}
				case ApproximateMatrix<T>::INDEX_LOW_RANK: {
					const auto &low_rank = std::get<ApproximateMatrix<T>::INDEX_LOW_RANK>(node);
					// vt * vector
					auto vv = Vector<T>{low_rank.rank()};
					matrix_vector_multiply(low_rank.vt(), vector, vv, 0, low_rank.min_j());
					// vv = s * vv
					vv = matrix_vector_multiply(low_rank.s(), vv);
					// u * vv
					matrix_vector_multiply(low_rank.u(), vv, result, low_rank.min_i(), 0);
					break;
				}
				default:
					throw std::runtime_error("unexpected node index");
				}
			}
			return result;
		}

		/// @brief 前処理を行う
		void preprocess(const Problem<T> &, const Matrix<T> &, const Vector<T> &) {}
		/// @brief 後処理を行う
		void postprocess(const Problem<T> &, Matrix<T> &, Vector<T> &, Vector<T> &) {}

		/// @brief matrixを階層型低ランク近似してapproxに書き込む
		void low_rank_approximate(const Problem<T> &problem, const Matrix<T> &matrix, ApproximateMatrix<T> &approx, const std::optional<ApproximateMatrix<double>> &approx_ref = std::nullopt) const {
			approximate_matrix(problem, matrix, approx, 0, 0, 0, matrix.msize(), matrix.nsize(), approx_ref);
		}

	  private:
		/// @brief matrixのmin_i行min_j列目から始まるM行N列の行列を階層型低ランク近似行列に変換する.結果はapproxのidx番目のノードに書き込む.
		void approximate_matrix(const Problem<T> &problem, const Matrix<T> &matrix, ApproximateMatrix<T> &approx, const std::size_t idx, const std::size_t min_i, const std::size_t min_j, const std::size_t M, const std::size_t N, const std::optional<ApproximateMatrix<double>> &approx_ref) const {
			if (const auto split = get_split_size(problem, M, N)) {
				const auto [m_split, n_split] = split.value();
				// 分割
				auto &node = approx.split(idx, m_split, n_split);
				// 右上
				low_rank_matrix(problem, matrix, approx, node.upper_right(), min_i, min_j + n_split, m_split, N - n_split, approx_ref);
				// 左下
				low_rank_matrix(problem, matrix, approx, node.lower_left(), min_i + m_split, min_j, M - m_split, n_split, approx_ref);
				const auto ul_idx = node.upper_left();
				const auto lr_idx = node.lower_right();
				// 左上
				approximate_matrix(problem, matrix, approx, ul_idx, min_i, min_j, m_split, n_split, approx_ref);
				// 右下
				approximate_matrix(problem, matrix, approx, lr_idx, min_i + m_split, min_j + n_split, M - m_split, N - n_split, approx_ref);
			} else {
				// 行列を分割しない場合は密行列として計算する.
				auto &dense = approx.set_dense(idx).matrix();
#pragma omp parallel for
				for (auto i = decltype(M){0}; i < M; ++i) {
					std::copy(matrix.data() + (min_i + i) * matrix.nsize() + min_j, matrix.data() + (min_i + i) * matrix.nsize() + min_j + N, dense.data() + i * N);
				}
			}
		}

		/// @brief 特異値分解された行列を低ランク近似する.結果はapproxのidx番目のノードに書き込む.
		void low_rank_matrix(const Problem<T> &problem, const Matrix<T> &matrix, ApproximateMatrix<T> &approx, const std::size_t idx, const std::size_t min_i, const std::size_t min_j, const std::size_t M, const std::size_t N, const std::optional<ApproximateMatrix<double>> &approx_ref) const {
			const auto [u_entity, s_entity, vt_entity] = singular_value_decomposition<T>(matrix, min_i, min_j, M, N);
			const auto &u = u_entity;
			const auto &s = s_entity;
			const auto &vt = vt_entity;
			const auto approx_ref_ptr = approx_ref ? std::get_if<ApproximateMatrix<T>::INDEX_LOW_RANK>(&approx_ref.value().node(idx)) : nullptr;
			const auto rank = approx_ref_ptr ? approx_ref_ptr->rank() : get_rank(problem, u, s, vt);
#ifdef H_MATRIX_APPROX_ALPHA
			const auto approx_alpha = static_cast<T>(H_MATRIX_APPROX_ALPHA);
#else
			const auto approx_alpha = T{};
#endif
			// 計算量が閾値を超える場合は近似を行わない
			if (M * N < approx_alpha * (M + N + 1) * rank) {
				auto &dense = approx.set_dense(idx).matrix();
#pragma omp parallel for
				for (auto i = decltype(M){0}; i < M; ++i) {
					std::copy(matrix.data() + (min_i + i) * matrix.nsize() + min_j, matrix.data() + (min_i + i) * matrix.nsize() + min_j + N, dense.data() + i * N);
				}
			} else {
				auto &low_rank = approx.set_low_rank(idx, rank);
				auto &lr_u = low_rank.u();
#pragma omp parallel for
				for (auto i = decltype(M){0}; i < M; ++i) {
					for (auto j = decltype(rank){0}; j < rank; ++j) {
						lr_u(i, j) = u(i, j);
					}
				}
				auto &lr_s = low_rank.s();
				for (auto i = decltype(rank){0}; i < rank; ++i) {
					lr_s(i) = s(i);
				}
				auto &lr_vt = low_rank.vt();
#pragma omp parallel for
				for (auto i = decltype(rank){0}; i < rank; ++i) {
					for (auto j = decltype(N){0}; j < N; ++j) {
						lr_vt(i, j) = vt(i, j);
					}
				}
			}
		}

		/// @brief 部分行列とベクトルの積を結果ベクトルに加算する
		static auto matrix_vector_multiply(const Matrix<T> &matrix, const Vector<T> &vector, Vector<T> &result, const std::size_t offset_i, const std::size_t offset_j) {
			const auto M = matrix.msize();
			const auto N = matrix.nsize();
#pragma omp parallel for
			for (auto i = decltype(M){0}; i < M; ++i) {
				for (auto j = decltype(N){0}; j < N; ++j) {
					result(offset_i + i) += matrix(i, j) * vector(offset_j + j);
				}
			}
		}
	};
} // namespace h_matrix

#endif // H_MATRIX_SOLVER_REF_HPP
