#ifndef GRAVITY_OCTREE_HPP
#define GRAVITY_OCTREE_HPP

#include <complex>
#include <optional>
#include <variant>

namespace gravity {
	/// @brief Cellの子要素のindexのiterator
	struct OctreeCellsIndexIterator {
		/// @brief indexを返す
		auto operator*() const noexcept { return _it; }

		/// @brief indexが異なるかを判定する
		auto operator!=(const OctreeCellsIndexIterator &other) const noexcept { return _it != other._it; }

		/// @brief indexが等しいかを判定する
		auto operator==(const OctreeCellsIndexIterator &other) const noexcept { return _it == other._it; }

		/// @brief iteratorをインクリメントする
		auto &operator++() noexcept {
			++_it;
			return *this;
		}

		/// @brief iteratorをインクリメントする
		auto operator++(int) const noexcept {
			return OctreeCellsIndexIterator{_it + 1};
		}

		friend struct OctreeCellsIndexRange;

	  private:
		/// @brief iteratorを構築する
		explicit OctreeCellsIndexIterator(const std::size_t it) noexcept : _it{it} {}

		std::size_t _it;
	};

	/// @brief Cellの子要素のindexをiterate可能なObject
	struct OctreeCellsIndexRange {
		/// @brief 開始位置のiteratorを返す
		auto begin() const noexcept {
			return OctreeCellsIndexIterator{_it};
		}

		/// @brief 終了位置のiteratorを返す
		auto end() const noexcept {
			return OctreeCellsIndexIterator{_end};
		}

		/// @brief 子要素のindexの開始位置から範囲オブジェクトを構築する
		explicit OctreeCellsIndexRange(const std::size_t begin) noexcept : _it{begin}, _end{begin + 8} {}

	  private:
		std::size_t _it;
		std::size_t _end;
	};

	/// @brief 3次元空間を管理する八分木
	/// @details
	/// Cellは
	/// 1. ルートノードの構築のためCell(const std::vector<Point>&)が呼び出せることを要求する.
	/// 2. 中間ノードの構築のためCell(std::size_t, std::size_t, Point, Point)が呼び出せることを要求する.第一引数は_cellsにおけるindex,第二引数は親ノードの_cellsにおけるindex,第三,第四引数はセルの領域の下限,上限が渡される
	/// 3. 領域の管理のためにmin()/max()メンバ関数を持つ必要がある.戻り値はconst Point&
	/// 4. 子ノード/点の保持のためcontent()メンバ関数を持つ必要がある.戻り値はconst std::variant<std::vector<std::size_t>, std::size_t>&又はstd::variant<std::vector<std::size_t>, std::size_t>&
	/// Pointは
	/// 1. データの構築のためデフォルトコンストラクタ,xyz座標の3つの値を受け取るコンストラクタが呼び出せることを要求する.
	/// 2. 座標の管理のためにx(),y(),z()メンバ関数を持つ必要がある.戻り値は大小比較可能であることを要求する
	template <class Cell, class Point>
	struct Octree {
		/// @brief セルが葉ノードである場合のCell::content()が返すvariantのindex
		static constexpr auto INDEX_POINTS = std::size_t{0};
		/// @brief セルが中間ノードである場合のCell::content()が返すvariantのindex
		static constexpr auto INDEX_CELLS = std::size_t{1};

		/// @brief 点の集合と一つの葉ノードの持てる点の数の閾値情報から八分木を構築する
		/// @exception 点の挿入に失敗した場合std::runtime_error例外が送出される
		Octree(const std::vector<Point> &points, const std::size_t s) : _points{points} {
			// ルートノードを追加
			_cells.emplace_back(points);
			// セルを分割する閾値
			const auto threshold = static_cast<std::size_t>(std::floor(std::sqrt(8) * s));

			const auto N = points.size();
			for (auto i = decltype(N){0}; i < N; ++i) {
				if (!insert(0, i, threshold)) {
					throw std::runtime_error("failed to insert point");
				}
			}
		}

		/// @brief 八分木に含まれるすべての点を返す
		const auto &points() const noexcept {
			return _points;
		}

		/// @brief 八分木に含まれるすべてのセルを返す
		const auto &cells() const noexcept {
			return _cells;
		}

		/// @brief 八分木に含まれるすべてのセルを返す
		/// @details Cellの値を変更するためのものであり,cellsを変更することは想定されていない
		auto &cells() noexcept {
			return _cells;
		}

		/// @brief 受け取った関数オブジェクトにルートノードと追加の引数を適用する
		template <class F, class... Args>
		decltype(auto) apply(F &&f, Args &&... args) {
			return f(_cells[0], std::forward<Args>(args)...);
		}

		/// @brief 受け取った関数オブジェクトを再帰可能な関数オブジェクトに変更する.applyに適用するラムダ式で再帰処理を行いたい場合に使用する
		template <class F>
		static decltype(auto) fix(F &&f) noexcept {
			return [f = std::forward<F>(f)](auto &&... args) {
				return f(f, std::forward<decltype(args)>(args)...);
			};
		}

#ifdef ENABLE_TEST
		friend struct FMMTest;
#endif

	  private:
		/// @brief _cells[index]のセルに_points[i]の点を追加する.
		/// @return 追加に成功した場合はtrue,失敗した場合はfalseを返す.
		/// @exception subdivideの際,点の再挿入に失敗した場合std::runtime_error例外が送出される
		auto insert(const std::size_t index, const std::size_t i, const std::size_t threshold) {
			auto &cell = _cells[index];
			const auto &point = _points[i];
			const auto &min = cell.min();
			const auto &max = cell.max();
			if (point.x() < min.x() || max.x() < point.x() || point.y() < min.y() || max.y() < point.y() || point.z() < min.z() || max.z() < point.z()) {
				return false;
			}
			auto &content = cell.content();
			switch (content.index()) {
			case INDEX_POINTS: {
				auto &points = std::get<INDEX_POINTS>(content);
				if (points.size() < threshold) {
					points.emplace_back(i);
					return true;
				}
				subdivide(index, threshold);
			}
				[[fallthrough]];
			default: // INDEX_CELLS:

				// subdivideによって_cellsの再配置が発生する可能性があるためcontentを再取得
				const auto child_index_base = std::get<INDEX_CELLS>(_cells[index].content());
				for (auto child_index = child_index_base; child_index < child_index_base + 8; ++child_index) {
					if (insert(child_index, i, threshold)) {
						return true;
					}
				}
				return false;
			}
		}

		/// @brief _cells[index]を分割する
		/// @details 子ノードはcellsの末尾に連続して確保される.先頭要素からのオフセット0-7を3ビットで表現した際,最下位ビットが0のセルはx座標が大きい側,中間ビットが0のセルはy座標が大きい側,最上位ビットが0のセルはz座標の大きい側へ配置される
		/// @exception 点の再挿入に失敗した場合std::runtime_error例外が送出される
		void subdivide(const std::size_t index, const std::size_t threshold) {
			const auto cell_min = _cells[index].min();
			const auto cell_max = _cells[index].max();
			const auto mid = decltype(cell_min){(cell_max.x() + cell_min.x()) / 2, (cell_max.y() + cell_min.y()) / 2, (cell_max.z() + cell_min.z()) / 2};
			const auto child_index_base = _cells.size();
			for (auto child_index_offset = decltype(child_index_base){0}; child_index_offset < 8; ++child_index_offset) {
				const auto child_index = child_index_base + child_index_offset;
				_cells.emplace_back(Cell{child_index,
				                         index,
				                         {child_index_offset % 2 == 0 ? mid.x() : cell_min.x(), child_index_offset % 4 < 2 ? mid.y() : cell_min.y(), child_index_offset < 4 ? mid.z() : cell_min.z()},
				                         {child_index_offset % 2 == 0 ? cell_max.x() : mid.x(), child_index_offset % 4 < 2 ? cell_max.y() : mid.y(), child_index_offset < 4 ? cell_max.z() : mid.z()}});
			}
			for (const auto i : std::get<INDEX_POINTS>(_cells[index].content())) {
				auto inserted = false;
				for (auto child_index = child_index_base; child_index < child_index_base + 8; ++child_index) {
					if (insert(child_index, i, threshold)) {
						inserted = true;
						break;
					}
				}
				if (!inserted) {
					throw std::runtime_error("failed to insert point");
				}
			}
			_cells[index].content() = child_index_base;
		}

		const std::vector<Point> &_points;
		std::vector<Cell> _cells;
	};
} // namespace gravity

#endif // GRAVITY_OCTREE_HPP
