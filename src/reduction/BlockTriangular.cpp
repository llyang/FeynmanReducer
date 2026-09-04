#include "reduction/BlockTriangular.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>

namespace block_triangular {

namespace {

std::vector<std::size_t>
validate_and_map_blocks(std::size_t dimension, std::span<const Range> ranges,
                        std::span<const Coordinate> coordinates)
{
  if (dimension == 0) {
    if (!ranges.empty() || !coordinates.empty()) {
      throw std::invalid_argument("zero-dimensional block structure must be empty");
    }
    return {};
  }
  if (ranges.empty()) {
    throw std::invalid_argument("block ranges do not cover the matrix");
  }

  std::size_t cursor = 0;
  std::vector<std::size_t> block_of_position(dimension);
  for (std::size_t block = 0; block < ranges.size(); ++block) {
    const auto range = ranges[block];
    if (range.begin != cursor || range.end <= range.begin || range.end > dimension) {
      throw std::invalid_argument(
          "block ranges must be non-empty, contiguous, and in bounds");
    }
    for (std::size_t position = range.begin; position < range.end; ++position)
      block_of_position[position] = block;
    cursor = range.end;
  }
  if (cursor != dimension) {
    throw std::invalid_argument("block ranges do not cover the matrix");
  }
  for (const auto [row, column] : coordinates) {
    if (row >= dimension || column >= dimension) {
      throw std::invalid_argument("matrix coordinate is outside block ranges");
    }
  }
  return block_of_position;
}

} // namespace

Ordering order_by_dependencies(std::size_t dimension,
                               std::span<const Range> initial_ranges,
                               std::span<const Coordinate> nonzero_coordinates)
{
  const auto block_of_position =
      validate_and_map_blocks(dimension, initial_ranges, nonzero_coordinates);
  if (dimension == 0) return {};

  const std::size_t block_count = initial_ranges.size();
  std::vector<std::vector<std::size_t>> graph(block_count);
  for (const auto [row, column] : nonzero_coordinates) {
    const std::size_t source = block_of_position[row];
    const std::size_t target = block_of_position[column];
    if (source != target) graph[source].push_back(target);
  }
  for (auto& edges : graph) {
    std::ranges::sort(edges);
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  }

  const std::size_t unvisited = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> discovery(block_count, unvisited);
  std::vector<std::size_t> lowlink(block_count, 0);
  std::vector<std::size_t> stack;
  std::vector<bool> on_stack(block_count, false);
  std::vector<std::size_t> component_of(block_count, unvisited);
  std::vector<std::vector<std::size_t>> components;
  std::size_t next_discovery = 0;
  std::function<void(std::size_t)> visit = [&](std::size_t block) {
    discovery[block] = lowlink[block] = next_discovery++;
    stack.push_back(block);
    on_stack[block] = true;
    for (const std::size_t target : graph[block]) {
      if (discovery[target] == unvisited) {
        visit(target);
        lowlink[block] = std::min(lowlink[block], lowlink[target]);
      } else if (on_stack[target]) {
        lowlink[block] = std::min(lowlink[block], discovery[target]);
      }
    }
    if (lowlink[block] != discovery[block]) return;
    const std::size_t component = components.size();
    components.emplace_back();
    for (;;) {
      const std::size_t member = stack.back();
      stack.pop_back();
      on_stack[member] = false;
      component_of[member] = component;
      components.back().push_back(member);
      if (member == block) break;
    }
    std::ranges::sort(components.back());
  };
  for (std::size_t block = 0; block < block_count; ++block) {
    if (discovery[block] == unvisited) visit(block);
  }

  std::vector<std::vector<std::size_t>> component_graph(components.size());
  std::vector<std::size_t> indegree(components.size(), 0);
  for (std::size_t source = 0; source < block_count; ++source) {
    for (const std::size_t target : graph[source]) {
      const std::size_t source_component = component_of[source];
      const std::size_t target_component = component_of[target];
      if (source_component != target_component)
        component_graph[source_component].push_back(target_component);
    }
  }
  for (auto& edges : component_graph) {
    std::ranges::sort(edges);
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    for (const std::size_t target : edges)
      ++indegree[target];
  }

  using Ready = std::pair<std::size_t, std::size_t>;
  std::priority_queue<Ready, std::vector<Ready>, std::greater<>> ready;
  for (std::size_t component = 0; component < components.size(); ++component) {
    if (indegree[component] == 0)
      ready.emplace(components[component].front(), component);
  }

  Ordering result;
  result.old_position_by_new.reserve(dimension);
  result.ranges.reserve(components.size());
  while (!ready.empty()) {
    const auto [unused_key, component] = ready.top();
    static_cast<void>(unused_key);
    ready.pop();
    const std::size_t begin = result.old_position_by_new.size();
    for (const std::size_t block : components[component]) {
      const auto range = initial_ranges[block];
      for (std::size_t position = range.begin; position < range.end; ++position)
        result.old_position_by_new.push_back(position);
    }
    result.ranges.push_back({begin, result.old_position_by_new.size()});
    for (const std::size_t target : component_graph[component]) {
      if (--indegree[target] == 0) ready.emplace(components[target].front(), target);
    }
  }
  if (result.old_position_by_new.size() != dimension ||
      result.ranges.size() != components.size()) {
    throw std::logic_error("block dependency graph is not a DAG after SCC collapse");
  }
  return result;
}

std::vector<Range>
coarsen_to_upper_triangular(std::size_t dimension,
                            std::span<const Range> initial_ranges,
                            std::span<const Coordinate> nonzero_coordinates)
{
  const auto block_of_position =
      validate_and_map_blocks(dimension, initial_ranges, nonzero_coordinates);
  if (dimension == 0) return {};

  std::vector<bool> keep_boundary(initial_ranges.size() - 1, true);
  for (const auto [row, column] : nonzero_coordinates) {
    const std::size_t row_block = block_of_position[row];
    const std::size_t column_block = block_of_position[column];
    if (row_block > column_block) {
      for (std::size_t boundary = column_block; boundary < row_block; ++boundary) {
        keep_boundary[boundary] = false;
      }
    }
  }

  std::vector<Range> result;
  result.reserve(initial_ranges.size());
  std::size_t begin = 0;
  for (std::size_t boundary = 0; boundary < keep_boundary.size(); ++boundary) {
    if (keep_boundary[boundary]) {
      result.push_back({begin, initial_ranges[boundary].end});
      begin = initial_ranges[boundary].end;
    }
  }
  result.push_back({begin, dimension});
  return result;
}

} // namespace block_triangular
