#ifndef T_SNE_BARNES_HUT_TREE_HPP
#define T_SNE_BARNES_HUT_TREE_HPP

#include <problem.hpp>
#include <stdexcept>
#include <variant>

namespace t_sne {
	/// @brief yの点間の距離に応じた斥力を近似計算するためのBarnesHutTree
	template <class Output>
	struct BarnesHutTree {
		using embedding_t = typename Problem<Output>::embedding_t;

		explicit BarnesHutTree(const std::vector<embedding_t> &y)
		    : _root(y) {
		}

		/// @brief 任意の関数オブジェクトを受け取ってrootノードに適用する.ツリー全体を走査したい場合は再帰を利用する.
		/// @details 再帰の実装方法についてはSolverRef::repulsive等を参照.
		template <typename F>
		decltype(auto) apply(F &&f) const {
			return f(_root);
		}

		/// @brief BarnesHutTreeのノード.一つの点か4つのノードを持っている場合がある.
		struct Node {
			static constexpr auto EMBEDDING_DIMENSION = Problem<Output>::EMBEDDING_DIMENSION;
			static constexpr auto WIDTH_PADDING = Problem<Output>::BARNES_HUT_WIDTH_PADDING;
			static constexpr auto INDEX_EMPTY = std::size_t{0};
			static constexpr auto INDEX_POINT = std::size_t{1};
			static constexpr auto INDEX_CHILDREN = std::size_t{2};

			/// @brief ツリーを作る時に用いられる.BarnesHutTreeのコンストラクタから呼ばれるので手動で呼び出す必要はない.
			explicit Node(const std::vector<embedding_t> &y) : _y(y), _n(), _y_cell() {
				const auto N = y.size();
				const auto min_y = min(y);
				const auto max_y = max(y);
				const auto sum_y = sum(y);
				if (N != 0) {
					for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
						const auto center = sum_y[d] / static_cast<Output>(N);
						const auto width = std::max(max_y[d] - center, center - min_y[d]) * 2 + WIDTH_PADDING;
						_min[d] = center - width / 2;
						_max[d] = center + width / 2;
					}
				} else {
					for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
						_min[d] = 0;
						_max[d] = 0;
					}
				}

				for (auto i = decltype(N){0}; i < N; ++i) {
					if (!insert(i)) {
						throw std::runtime_error("failed to insert point");
					}
				}
				calc_y_cell();
			}

			const auto &parent() const noexcept {
				return _parent;
			}

			const auto &y() const noexcept {
				return _y;
			}

			const auto &min() const noexcept {
				return _min;
			}

			const auto &max() const noexcept {
				return _max;
			}

			const auto &content() const noexcept {
				return _content;
			}

			auto n_cell() const noexcept {
				return _n;
			}

			const auto &y_cell() const noexcept {
				return _y_cell;
			}

			auto r_cell() const noexcept {
				auto r_cell2 = Output{0};
				for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
					const auto width = _max[d] - _min[d];
					r_cell2 += width * width;
				}
				return std::sqrt(r_cell2);
			}

		  private:
			/// @brief 子ノードを作る時に用いられる.親ノード,ノードの中心と管理領域の幅を与える.
			Node(const Node &parent, const embedding_t min, const embedding_t max)
			    : _parent(parent), _y(parent._y), _min(min), _max(max), _n(), _y_cell() {
			}

			/// @brief ツリー構築の最後にすべてのノードの重心を計算する.
			void calc_y_cell() {
				if (_n != 0) {
					for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
						_y_cell[d] /= _n;
					}
					if (_content.index() == INDEX_CHILDREN) {
						auto &children = std::get<INDEX_CHILDREN>(_content);
						for (auto &child : children) {
							child.calc_y_cell();
						}
					}
				}
			}

			/// @brief Nodeにyのindex番目の要素を追加する.
			/// @return 挿入に成功したらtrue,失敗したらfalseを返す
			auto insert(const std::size_t index) {
				const auto &point = _y[index];
				for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
					// pointがノードの範囲外なら挿入失敗
					if (_min[d] > point[d] || _max[d] < point[d]) {
						return false;
					}
				}

				_n++;
				for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
					_y_cell[d] += point[d];
				}

				switch (_content.index()) {
				case INDEX_EMPTY:
					_content = index;
					return true;
				case INDEX_POINT:
					subdivide();
					[[fallthrough]];
				default: // INDEX_CHILDREN:
					for (auto &node : std::get<INDEX_CHILDREN>(_content)) {
						if (node.insert(index)) {
							return true;
						}
					}
					return false;
				}
			}

			/// @brief Nodeを細分化する.
			void subdivide() {
				const auto index = std::get<INDEX_POINT>(_content);
				auto mid = embedding_t{};
				for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
					mid[d] = (_min[d] + _max[d]) / 2;
				}
				const auto w = embedding_t{_min[0], mid[1]};
				const auto n = embedding_t{mid[0], _max[1]};
				const auto s = embedding_t{mid[0], _min[1]};
				const auto e = embedding_t{_max[0], mid[1]};
				_content = std::vector<Node>{
				    Node{*this, mid, _max},
				    Node{*this, w, n},
				    Node{*this, s, e},
				    Node{*this, _min, mid}};
				auto &children = std::get<INDEX_CHILDREN>(_content);
				for (auto i = decltype(EMBEDDING_DIMENSION){0}; i < EMBEDDING_DIMENSION * EMBEDDING_DIMENSION; ++i) {
					if (children[i].insert(index)) {
						return;
					}
				}
				throw std::runtime_error("failed to subdivide Node");
			}

			static auto min(const std::vector<embedding_t> &y) noexcept {
				auto min_y = embedding_t{};
				for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
					min_y[d] = std::numeric_limits<Output>::max();
				}
				for (const auto &point : y) {
					for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
						if (point[d] < min_y[d]) {
							min_y[d] = point[d];
						}
					}
				}
				return min_y;
			}

			static auto max(const std::vector<embedding_t> &y) noexcept {
				auto max_y = embedding_t{};
				for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
					max_y[d] = std::numeric_limits<Output>::min();
				}
				for (const auto &point : y) {
					for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
						if (point[d] > max_y[d]) {
							max_y[d] = point[d];
						}
					}
				}
				return max_y;
			}

			static auto sum(const std::vector<embedding_t> &y) noexcept {
				auto sum_y = embedding_t{};
				for (const auto &point : y) {
					for (auto d = decltype(EMBEDDING_DIMENSION){0}; d < EMBEDDING_DIMENSION; ++d) {
						sum_y[d] += point[d];
					}
				}
				return sum_y;
			}

			/// @brief 親ノード(Rootノードならnullopt)
			std::optional<std::reference_wrapper<const Node>> _parent;
			const std::vector<embedding_t> &_y;
			embedding_t _min;
			embedding_t _max;
			/// @brief 領域内が空の場合はstd::monostate, 領域内に点が存在する場合はstd::size_t, 領域内に子ノードが存在する場合はstd::vector<Node>を持つ
			std::variant<std::monostate, std::size_t, std::vector<Node>> _content;
			/// @brief 領域内の点の総数
			std::size_t _n;
			/// @brief 領域内の点の重心
			embedding_t _y_cell;
		};

#ifdef ENABLE_TEST
		friend struct BarnesHutTreeTest;
#endif

	  private:
		Node _root;
	};
} // namespace t_sne

#endif // T_SNE_BARNES_HUT_TREE_HPP
