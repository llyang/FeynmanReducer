#include "symmetry/detail/Graph.hpp"

#include <nausparse.h>
#include <traces.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

thread_local std::vector<VariablePermutation>* current_generators = nullptr;
thread_local std::size_t current_variable_count = 0;

void collect_automorphism(int, int* permutation, int vertex_count)
{
  if (current_generators == nullptr ||
      current_variable_count > static_cast<std::size_t>(vertex_count)) {
    return;
  }
  VariablePermutation restricted(current_variable_count);
  for (std::size_t index = 0; index < current_variable_count; ++index) {
    restricted[index] = static_cast<std::uint32_t>(permutation[index]);
  }
  current_generators->push_back(std::move(restricted));
}

class CallbackScope {
public:
  CallbackScope(std::vector<VariablePermutation>& generators,
                std::size_t variable_count)
  {
    if (current_generators != nullptr) {
      throw std::runtime_error("nested Traces symmetry search");
    }
    current_generators = &generators;
    current_variable_count = variable_count;
  }

  CallbackScope(const CallbackScope&) = delete;
  CallbackScope& operator=(const CallbackScope&) = delete;

  ~CallbackScope()
  {
    current_generators = nullptr;
    current_variable_count = 0;
  }
};

} // namespace

namespace symmetry::detail {

GraphAnalysis analyze_nauty_graph(const ColoredGraph& graph, std::size_t variable_count)
{
  if (graph.colors.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Traces graph exceeds INT_MAX vertices");
  }
  const int vertex_count = static_cast<int>(graph.colors.size());
  std::vector<std::vector<int>> adjacency(graph.colors.size());
  for (const auto& [left, right] : graph.edges) {
    if (left >= graph.colors.size() || right >= graph.colors.size()) {
      throw std::runtime_error("invalid LP symmetry graph edge");
    }
    adjacency[left].push_back(static_cast<int>(right));
    adjacency[right].push_back(static_cast<int>(left));
  }
  std::size_t directed_edge_count = 0;
  for (auto& neighbours : adjacency) {
    std::ranges::sort(neighbours);
    neighbours.erase(std::ranges::unique(neighbours).begin(), neighbours.end());
    directed_edge_count += neighbours.size();
  }

  std::vector<std::size_t> offsets(graph.colors.size(), 0);
  std::vector<int> degrees(graph.colors.size(), 0);
  std::vector<int> edges;
  edges.reserve(directed_edge_count);
  for (std::size_t vertex = 0; vertex < adjacency.size(); ++vertex) {
    offsets[vertex] = edges.size();
    degrees[vertex] = static_cast<int>(adjacency[vertex].size());
    edges.insert(edges.end(), adjacency[vertex].begin(), adjacency[vertex].end());
  }
  sparsegraph sparse{
      directed_edge_count, offsets.data(),
      vertex_count,        degrees.data(),
      edges.data(),        nullptr,
      offsets.size(),      degrees.size(),
      edges.size(),        0,
  };

  std::vector<int> lab(graph.colors.size());
  std::iota(lab.begin(), lab.end(), 0);
  std::ranges::sort(lab, {}, [&](int vertex) {
    return std::pair{graph.colors[static_cast<std::size_t>(vertex)], vertex};
  });
  std::vector<int> partition(graph.colors.size(), 0);
  for (std::size_t index = 0; index + 1 < lab.size(); ++index) {
    partition[index] = graph.colors[static_cast<std::size_t>(lab[index])] ==
                               graph.colors[static_cast<std::size_t>(lab[index + 1])]
                           ? 1
                           : 0;
  }
  std::vector<int> orbits(graph.colors.size(), 0);

  DEFAULTOPTIONS_TRACES(options);
  options.defaultptn = FALSE;
  options.getcanon = TRUE;
  options.userautomproc = collect_automorphism;
  TracesStats stats{};
  GraphAnalysis result;
  SG_DECL(canonical_graph);

  static std::mutex traces_mutex;
  const std::scoped_lock lock(traces_mutex);
  CallbackScope callback(result.generators, variable_count);
  nauty_check(WORDSIZE, SETWORDSNEEDED(vertex_count), vertex_count, NAUTYVERSIONID);
  Traces(&sparse, lab.data(), partition.data(), orbits.data(), &options, &stats,
         &canonical_graph);
  if (stats.errstatus != 0) {
    SG_FREE(canonical_graph);
    throw std::runtime_error("Traces failed with status " +
                             std::to_string(stats.errstatus));
  }
  result.canonical_labels.resize(graph.colors.size());
  for (std::size_t canonical = 0; canonical < lab.size(); ++canonical) {
    const int original = lab[canonical];
    if (original < 0 || static_cast<std::size_t>(original) >= lab.size()) {
      SG_FREE(canonical_graph);
      throw std::runtime_error("Traces returned an invalid canonical label");
    }
    result.canonical_labels[static_cast<std::size_t>(original)] =
        static_cast<std::uint32_t>(canonical);
  }
  SG_FREE(canonical_graph);
  return result;
}

} // namespace symmetry::detail
