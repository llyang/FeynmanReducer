#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace block_triangular {

struct Range {
  std::size_t begin = 0;
  std::size_t end = 0;

  bool operator==(const Range&) const = default;
};

using Coordinate = std::pair<std::size_t, std::size_t>;

struct Ordering {
  /// Maps each new matrix position to its position before reordering.
  std::vector<std::size_t> old_position_by_new;
  /// Strongly connected block components in deterministic topological order.
  std::vector<Range> ranges;
};

/// Reorders square initial blocks by the strongly connected components of their
/// symbolic dependency graph. An edge row_block -> column_block constrains the
/// row block to appear no later than the column block. Positions inside an
/// initial block and initial blocks inside one component retain their order.
[[nodiscard]] Ordering
order_by_dependencies(std::size_t dimension, std::span<const Range> initial_ranges,
                      std::span<const Coordinate> nonzero_coordinates);

/// Coarsens contiguous initial ranges until every symbolic non-zero coordinate
/// lies on or above the resulting block diagonal. The worst-case result is one
/// range covering the complete matrix.
[[nodiscard]] std::vector<Range>
coarsen_to_upper_triangular(std::size_t dimension,
                            std::span<const Range> initial_ranges,
                            std::span<const Coordinate> nonzero_coordinates);

} // namespace block_triangular
