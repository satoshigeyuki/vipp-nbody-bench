#ifndef T_SNE_VANTAGE_POINT_TREE_HPP
#define T_SNE_VANTAGE_POINT_TREE_HPP

#include <queue>

namespace t_sne {
	/// @brief Xの各点に対する近傍点を探索するためのVantagePointTree.
	template <class T, auto DistanceX>
	struct VantagePointTree {
		using distance_t = std::invoke_result_t<decltype(DistanceX), const T &, const T &>;

		explicit VantagePointTree(const std::vector<T> &x) : _x(x), _tree(x.size()) {
			std::iota(_tree.begin(), _tree.end(), 0);
			build_tree(0, x.size());
		}

		/// @brief _x[index]のk個近傍の点のindexとその点までの距離をindicesとdistancesのindex*k番目から(index+1)*k番目に書き込む.
		/// @exception 探索する近傍点の数(k)が実際の点数(N)と等しくなるか上回るとstd::runtime_error例外が送出される
		void search(const std::size_t k) {
			const auto N = _x.size();
			if (k >= N) {
				throw std::runtime_error("k is too large");
			}

			_k = k;
			_indices.resize(N * k);
			_distances.resize(N * k);

#pragma omp parallel for
			for (auto index = decltype(N){0}; index < N; ++index) {
				auto queue = std::priority_queue<PointFromTarget>{};
				auto max_distance = std::numeric_limits<distance_t>::max();
				search(index, k + 1, queue, max_distance, 0, _tree.size());
				for (auto i = decltype(k){0}; i < k; ++i) {
					_indices[(index + 1) * k - i - 1] = queue.top().index;
					_distances[(index + 1) * k - i - 1] = queue.top().distance;
					queue.pop();
				}
			}
		}

		auto k() const noexcept {
			return _k;
		}

		/// @brief 点iのj+1番目の近傍点のindexを返す.
		auto nearest_index(const std::size_t i, const std::size_t j) const {
			if (i >= _x.size()) {
				throw std::runtime_error("i is too large");
			}
			if (j >= _k) {
				throw std::runtime_error("j is too large");
			}
			return _indices[i * _k + j];
		}

		/// @brief 点jが点iの近傍点であるかどうかを返す.k+1番目の近傍点であるときkを,近傍点でないときstd::nuloptを返す.
		std::optional<std::size_t> find_nearest_index(const std::size_t i, const std::size_t j) const {
			const auto begin = _indices.begin() + static_cast<std::ptrdiff_t>(i * _k);
			const auto end = _indices.begin() + static_cast<std::ptrdiff_t>((i + 1) * _k);
			const auto it = std::find(begin, end, j);
			if (it == end) {
				return std::nullopt;
			}
			return static_cast<std::size_t>(std::distance(begin, it));
		}

		/// @brief 点iのj+1番目の近傍点との距離を返す.
		auto nearest_distance(const std::size_t i, const std::size_t j) const {
			if (i >= _x.size()) {
				throw std::runtime_error("i is too large");
			}
			if (j >= _k) {
				throw std::runtime_error("j is too large");
			}
			return _distances[i * _k + j];
		}

#ifdef ENABLE_TEST
		friend struct VantagePointTreeTest;
		friend struct SolverRefTest;
#endif

	  private:
		void build_tree(const std::size_t lower, const std::size_t upper) noexcept {
			if (lower >= upper) {
				return;
			}
			const auto inner_lower = (lower + 1);
			const auto median = (lower + upper + 1) / 2;
			std::nth_element(_tree.begin() + static_cast<std::ptrdiff_t>(inner_lower),
			                 _tree.begin() + static_cast<std::ptrdiff_t>(median),
			                 _tree.begin() + static_cast<std::ptrdiff_t>(upper),
			                 [lower, this](const std::size_t lhs, const std::size_t rhs) {
				                 return DistanceX(_x[_tree[lower]], _x[lhs]) < DistanceX(_x[_tree[lower]], _x[rhs]);
			                 });
			build_tree(inner_lower, median);
			build_tree(median, upper);
		}

		struct PointFromTarget {
			std::size_t index;
			distance_t distance;
			bool operator<(const PointFromTarget &rhs) const noexcept {
				return distance < rhs.distance;
			}
		};

		void search(const std::size_t index, const std::size_t k, std::priority_queue<PointFromTarget> &queue, distance_t &max_distance, const std::size_t lower, const std::size_t upper) const {
			if (lower >= upper) {
				return;
			}
			const auto distance = DistanceX(_x[_tree[lower]], _x[index]);
			if (distance < max_distance) {
				if (queue.size() == k) {
					queue.pop();
				}
				queue.push(PointFromTarget{_tree[lower], distance});
				if (queue.size() == k) {
					max_distance = queue.top().distance;
				}
			}
			const auto inner_lower = (lower + 1);
			const auto median = (lower + upper + 1) / 2;
			if (median == upper) {
				return;
			}
			const auto radius = DistanceX(_x[_tree[lower]], _x[_tree[median]]);
			if (distance < radius) {
				search(index, k, queue, max_distance, inner_lower, median);
				if (distance + max_distance >= radius) {
					search(index, k, queue, max_distance, median, upper);
				}
			} else {
				search(index, k, queue, max_distance, median, upper);
				if (distance - max_distance <= radius) {
					search(index, k, queue, max_distance, inner_lower, median);
				}
			}
		}

		const std::vector<T> &_x;
		std::size_t _k;
		std::vector<std::size_t> _tree;
		std::vector<std::size_t> _indices;
		std::vector<distance_t> _distances;
	};
} // namespace t_sne

#endif // T_SNE_VANTAGE_POINT_TREE_HPP
