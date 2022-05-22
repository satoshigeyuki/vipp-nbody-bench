#ifndef H_MATRIX_MATRIX_HPP
#define H_MATRIX_MATRIX_HPP

#include <stdexcept>
#include <variant>
#include <vector>

namespace h_matrix {
	template <class T>
	struct Vector;
	template <class T>
	struct DiagonalMatrix;

	/// @brief 型Tを要素として持つ行列
	template <class T>
	struct Matrix {
		/// @brief M行N列の零行列を生成
		Matrix(const std::size_t m, const std::size_t n) : _m{m}, _n{n}, _entity(m * n) {}

		/// @brief 行数を取得
		auto msize() const noexcept {
			return _m;
		}

		/// @brief 列数を取得
		auto nsize() const noexcept {
			return _n;
		}

		/// @brief Matrixの実体へのポインタを返す
		auto data() const noexcept {
			return _entity.data();
		}

		/// @brief Matrixの実体へのポインタを返す
		auto data() noexcept {
			return _entity.data();
		}

		/// @brief 行列の(i,j)要素の参照を取得
		const auto &operator()(const std::size_t i, const std::size_t j) const {
			return _entity.at(i * _n + j);
		}

		/// @brief 行列の(i,j)要素の参照を取得
		auto &operator()(const std::size_t i, const std::size_t j) {
			return _entity.at(i * _n + j);
		}

#ifdef ENABLE_TEST
		friend struct MatrixTest;
#endif
	  private:
		std::size_t _m;
		std::size_t _n;
		std::vector<T> _entity;
	};

	/// @brief 型Tを要素として持つ対角行列
	template <class T>
	struct DiagonalMatrix {
		/// @brief M行N列の対角行列を生成
		DiagonalMatrix(const std::size_t m, const std::size_t n) : _m{m}, _n{n}, _entity(std::min(m, n)) {}

		/// @brief 行数を取得
		auto msize() const noexcept {
			return _m;
		}

		/// @brief 列数を取得
		auto nsize() const noexcept {
			return _n;
		}

		/// @brief DiagonalMatrixの実体へのポインタを返す
		auto data() const noexcept {
			return _entity.data();
		}

		/// @brief DiagonalMatrixの実体へのポインタを返す
		auto data() noexcept {
			return _entity.data();
		}

		/// @brief 行列(i,i)要素の参照を取得
		const auto &operator()(const std::size_t i) const {
			return _entity.at(i);
		}

		/// @brief 行列(i,i)要素の参照を取得
		auto &operator()(const std::size_t i) {
			return _entity.at(i);
		}

	  private:
		std::size_t _m;
		std::size_t _n;
		std::vector<T> _entity;
	};

	/// @brief 低ランク近似された行列
	template <class T>
	struct ApproximateMatrix {
		static constexpr auto INDEX_EMPTY = std::size_t{0};
		static constexpr auto INDEX_DENSE = std::size_t{1};
		static constexpr auto INDEX_LOW_RANK = std::size_t{2};
		static constexpr auto INDEX_NODE = std::size_t{3};

		/// @brief M行N列の空の近似行列を生成
		ApproximateMatrix(const std::size_t m, const std::size_t n) : _m{m}, _n{n} {
			_nodes.emplace_back(Empty{0, 0, 0, m, n});
		}

		/// @brief 行数を取得
		auto msize() const noexcept {
			return _m;
		}

		/// @brief 列数を取得
		auto nsize() const noexcept {
			return _n;
		}

		/// @brief 葉ノードのidxのリストを取得
		const auto &leaves() const noexcept {
			return _leaves;
		}

		/// @brief ルートノードへのアクセス
		const auto &root() const noexcept {
			return _nodes[0];
		}
		/// @brief ルートノードへのアクセス
		auto &root() noexcept {
			return _nodes[0];
		}

		/// @brief ノードへのアクセス
		const auto &node(const std::size_t idx) const {
			return _nodes.at(idx);
		}
		/// @brief ノードへのアクセス
		auto &node(const std::size_t idx) {
			return _nodes.at(idx);
		}

		/// @brief idx番目のノードをm_split,n_splitで分割するようなNodeに変更
		/// @details node(idx)はEmptyであることを要求する.
		auto &split(const std::size_t idx, const std::size_t m_split, const std::size_t n_split) {
			// idx番目がEmptyであることを確認
			const auto &node = _nodes.at(idx);
			if (node.index() != INDEX_EMPTY) {
				throw std::runtime_error("node must be Empty");
			}
			// m_split,n_splitがm,nより小さいことを確認する
			const auto &empty = std::get<INDEX_EMPTY>(node);
			const auto min_i = empty._min_i;
			const auto min_j = empty._min_j;
			const auto m = empty._m;
			const auto n = empty._n;
			if (m < m_split) {
				throw std::runtime_error("m_split must be less than or equal to m");
			}
			if (n < n_split) {
				throw std::runtime_error("n_split must be less than or equal to n");
			}
			// 空の子ノードを追加する
			const auto idx_base = _nodes.size();
			// 左上
			_nodes.emplace_back(Empty{idx_base, min_i, min_j, m_split, n_split});
			// 右上
			_nodes.emplace_back(Empty{idx_base + 1, min_i, min_j + n_split, m_split, n - n_split});
			// 左下
			_nodes.emplace_back(Empty{idx_base + 2, min_i + m_split, min_j, m - m_split, n_split});
			// 右下
			_nodes.emplace_back(Empty{idx_base + 3, min_i + m_split, min_j + n_split, m - m_split, n - n_split});
			// 親ノードをNodeに変更する
			_nodes[idx] = Node{idx, min_i, min_j, m, n, m_split, n_split, idx_base};
			return std::get<INDEX_NODE>(_nodes[idx]);
		}

		/// @brief idx番目のノードを密行列に変更
		/// @details node(idx)はEmptyであることを要求する.
		auto &set_dense(const std::size_t idx) {
			// idx番目がEmptyであることを確認
			const auto &node = _nodes.at(idx);
			if (node.index() != INDEX_EMPTY) {
				throw std::runtime_error("node must be Empty");
			}
			// m_split,n_splitがm,nより小さいことを確認する
			const auto &empty = std::get<INDEX_EMPTY>(node);
			const auto min_i = empty._min_i;
			const auto min_j = empty._min_j;
			const auto m = empty._m;
			const auto n = empty._n;
			// ノードを密行列に変更する.
			_nodes[idx] = Dense{idx, min_i, min_j, m, n};
			// leavesに追加
			_leaves.push_back(idx);
			return std::get<INDEX_DENSE>(_nodes[idx]);
		}

		/// @brief idx番目のノードを低ランク近似行列に変更
		/// @details node(idx)はEmptyであることを要求する.
		auto &set_low_rank(const std::size_t idx, const std::size_t rank) {
			// idx番目がEmptyであることを確認
			const auto &node = _nodes.at(idx);
			if (node.index() != INDEX_EMPTY) {
				throw std::runtime_error("node must be Empty");
			}
			// m_split,n_splitがm,nより小さいことを確認する
			const auto &empty = std::get<INDEX_EMPTY>(node);
			const auto min_i = empty._min_i;
			const auto min_j = empty._min_j;
			const auto m = empty._m;
			const auto n = empty._n;
			// ノードを低ランク近似行列に変更する.
			_nodes[idx] = LowRank{idx, min_i, min_j, m, n, rank};
			// leavesに追加
			_leaves.push_back(idx);
			return std::get<INDEX_LOW_RANK>(_nodes[idx]);
		}

		/// @brief 空の行列,splitかset_dense, set_low_rankでNode,Dense,LowRankに変更する
		struct Empty {
			/// @brief idxを取得
			auto idx() const noexcept {
				return _idx;
			}

			/// @brief 元の行列の何行目からを表すか
			auto min_i() const noexcept {
				return _min_i;
			}

			/// @brief 元の行列の何列目からを表すか
			auto min_j() const noexcept {
				return _min_j;
			}

			/// @brief 行数を取得
			auto msize() const noexcept {
				return _m;
			}

			/// @brief 列数を取得
			auto nsize() const noexcept {
				return _n;
			}

			friend struct ApproximateMatrix;

		  private:
			Empty(const std::size_t idx, const std::size_t min_i, const std::size_t min_j, const std::size_t m, const std::size_t n) : _idx{idx}, _min_i{min_i}, _min_j{min_j}, _m{m}, _n{n} {}

			std::size_t _idx;
			std::size_t _min_i, _min_j;
			std::size_t _m, _n;
		};

		/// @brief 密行列
		struct Dense {
			/// @brief idxを取得
			auto idx() const noexcept {
				return _idx;
			}

			/// @brief 元の行列の何行目からを表すか
			auto min_i() const noexcept {
				return _min_i;
			}

			/// @brief 元の行列の何列目からを表すか
			auto min_j() const noexcept {
				return _min_j;
			}

			/// @brief 行数を取得
			auto msize() const noexcept {
				return _m;
			}

			/// @brief 列数を取得
			auto nsize() const noexcept {
				return _n;
			}

			/// @brief 密行列の本体
			auto &matrix() noexcept {
				return _matrix;
			}
			/// @brief 密行列の本体
			const auto &matrix() const noexcept {
				return _matrix;
			}

			friend struct ApproximateMatrix;

		  private:
			Dense(const std::size_t idx, const std::size_t min_i, const std::size_t min_j, const std::size_t m, const std::size_t n) : _idx{idx}, _min_i{min_i}, _min_j{min_j}, _m{m}, _n{n}, _matrix{m, n} {}

			std::size_t _idx;
			std::size_t _min_i, _min_j;
			std::size_t _m, _n;
			Matrix<T> _matrix;
		};

		/// @brief 低ランク近似した行列
		struct LowRank {
			/// @brief idxを取得
			auto idx() const noexcept {
				return _idx;
			}

			/// @brief 元の行列の何行目からを表すか
			auto min_i() const noexcept {
				return _min_i;
			}

			/// @brief 元の行列の何列目からを表すか
			auto min_j() const noexcept {
				return _min_j;
			}

			/// @brief 行数を取得
			auto msize() const noexcept {
				return _m;
			}

			/// @brief 列数を取得
			auto nsize() const noexcept {
				return _n;
			}

			/// @brief ランクを取得
			auto rank() const noexcept {
				return _rank;
			}

			/// @brief 左特異行列の本体
			auto &u() noexcept {
				return _u;
			}
			/// @brief 左特異行列の本体
			const auto &u() const noexcept {
				return _u;
			}

			/// @brief 特異値行列の本体
			auto &s() noexcept {
				return _s;
			}
			/// @brief 特異値行列の本体
			const auto &s() const noexcept {
				return _s;
			}

			/// @brief 右特異行列の本体
			auto &vt() noexcept {
				return _vt;
			}
			/// @brief 右特異行列の本体
			const auto &vt() const noexcept {
				return _vt;
			}

			friend struct ApproximateMatrix;

		  private:
			LowRank(const std::size_t idx, const std::size_t min_i, const std::size_t min_j, const std::size_t m, const std::size_t n, const std::size_t rank) : _idx{idx}, _min_i{min_i}, _min_j{min_j}, _m{m}, _n{n}, _rank{rank}, _u{m, rank}, _s{rank, rank}, _vt{rank, n} {}

			std::size_t _idx;
			std::size_t _min_i, _min_j;
			std::size_t _m, _n, _rank;
			Matrix<T> _u;
			DiagonalMatrix<T> _s;
			Matrix<T> _vt;
		};

		/// @brief 階層分割するノード
		struct Node {
			/// @brief idxを取得
			auto idx() const noexcept {
				return _idx;
			}

			/// @brief 元の行列の何行目からを表すか
			auto min_i() const noexcept {
				return _min_i;
			}

			/// @brief 元の行列の何列目からを表すか
			auto min_j() const noexcept {
				return _min_j;
			}

			/// @brief 行数を取得
			auto msize() const noexcept {
				return _m;
			}

			/// @brief 列数を取得
			auto nsize() const noexcept {
				return _n;
			}

			/// @brief 左上の子ノードのidxを取得
			auto upper_left() const noexcept {
				return _idx_base;
			}

			/// @brief 右上の子ノードのidxを取得
			auto upper_right() const noexcept {
				return _idx_base + 1;
			}

			/// @brief 左下の子ノードのidxを取得
			auto lower_left() const noexcept {
				return _idx_base + 2;
			}

			/// @brief 右下の子ノードのidxを取得
			auto lower_right() const noexcept {
				return _idx_base + 3;
			}

			friend struct ApproximateMatrix;

		  private:
			Node(const std::size_t idx, const std::size_t min_i, const std::size_t min_j, const std::size_t m, const std::size_t n, const std::size_t m_split, const std::size_t n_split, const std::size_t idx_base) : _idx{idx}, _min_i{min_i}, _min_j{min_j}, _m{m}, _n{n}, _m_split{m_split}, _n_split{n_split}, _idx_base{idx_base} {}

			std::size_t _idx;
			std::size_t _min_i, _min_j;
			std::size_t _m, _n;
			std::size_t _m_split, _n_split;
			std::size_t _idx_base;
		};

		void print() {
			const auto m_offset = std::size_t{10}, n_offset = std::size_t{10};
			std::cerr << "<svg width=\"" << msize() + 2 * m_offset << "\" height=\"" << nsize() + 2 * n_offset << "\" xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;
			auto font_size = std::size_t{100};
			for (const auto &node : _nodes) {
				switch (node.index()) {
				case INDEX_DENSE: {
					const auto &dense = std::get<INDEX_DENSE>(node);
					font_size = std::min(font_size, std::min(dense.msize(), dense.nsize()) * 2 / 3);
					break;
				}
				case INDEX_LOW_RANK: {
					const auto &low_rank = std::get<INDEX_LOW_RANK>(node);
					font_size = std::min(font_size, low_rank.rank() * 2 / 3);
					break;
				}
				default: {
					break;
				}
				}
			}
			for (const auto &node : _nodes) {
				switch (node.index()) {
				case INDEX_DENSE: {
					const auto &dense = std::get<INDEX_DENSE>(node);
					const auto dense_rank = std::min(dense.msize(), dense.nsize());
					std::cerr << "<path d=\"m " << dense.min_i() + m_offset << " " << dense.min_j() + n_offset << " h " << dense.msize() << " v " << dense.nsize() << " h -" << dense.msize() << " z\" fill-opacity=\"0\" stroke=\"black\"/>" << std::endl;
					std::cerr << "<text x=\"" << dense.min_i() + m_offset << "\" y=\"" << dense.min_j() + dense.nsize() + n_offset << "\" font-size=\"" << font_size << "\">" << dense_rank << "</text>" << std::endl;
					break;
				}
				case INDEX_LOW_RANK: {
					const auto &low_rank = std::get<INDEX_LOW_RANK>(node);
					std::cerr << "<path d=\"m " << low_rank.min_i() + m_offset << " " << low_rank.min_j() + n_offset << " h " << low_rank.msize() << " v " << low_rank.nsize() << " h -" << low_rank.msize() << " z\" fill-opacity=\"0\" stroke=\"blue\"/>" << std::endl;
					std::cerr << "<text x=\"" << low_rank.min_i() + m_offset << "\" y=\"" << low_rank.min_j() + low_rank.nsize() + n_offset << "\" font-size=\"" << font_size << "\" fill=\"blue\">" << low_rank.rank() << "</text>" << std::endl;
					break;
				}
				default: {
					break;
				}
				}
			}
			std::cerr << "</svg>" << std::endl;
		}

	  private:
		std::size_t _m;
		std::size_t _n;
		std::vector<std::variant<Empty, Dense, LowRank, Node>> _nodes;
		std::vector<std::size_t> _leaves;
	};

	/// @brief 型Tを要素として持つベクトル
	template <class T>
	struct Vector {
		/// @brief N行の零ベクトルを生成
		explicit Vector(const std::size_t n) : _entity(n) {}

		/// @brief 行数を取得
		auto size() const noexcept {
			return _entity.size();
		}

		/// @brief Vectorの実体へのポインタを返す
		auto data() const noexcept {
			return _entity.data();
		}

		/// @brief Vectorの実体へのポインタを返す
		auto data() noexcept {
			return _entity.data();
		}

		/// @brief i番目の要素の参照を取得
		const auto &operator()(const std::size_t i) const {
			return _entity.at(i);
		}

		/// @brief i番目の要素の参照を取得
		auto &operator()(const std::size_t i) {
			return _entity.at(i);
		}

#ifdef ENABLE_TEST
		friend struct MatrixTest;
#endif
	  private:
		std::vector<T> _entity;
	};
} // namespace h_matrix

#endif // H_MATRIX_MATRIX_HPP
