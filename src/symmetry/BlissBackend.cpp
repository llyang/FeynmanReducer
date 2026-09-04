#include "symmetry/detail/Graph.hpp"

#include <bliss/graph.hh>
#include <bliss/stats.hh>

#include <limits>
#include <stdexcept>
#include <vector>

namespace symmetry::detail {

GraphAnalysis analyze_bliss_graph(const ColoredGraph& graph, std::size_t variable_count)
{
  if (graph.colors.size() > std::numeric_limits<unsigned int>::max()) {
    throw std::runtime_error("bliss graph exceeds UINT_MAX vertices");
  }
  bliss::Graph bliss_graph;
  for (const auto color : graph.colors) {
    bliss_graph.add_vertex(color);
  }
  for (const auto& [left, right] : graph.edges) {
    if (left >= graph.colors.size() || right >= graph.colors.size()) {
      throw std::runtime_error("invalid LP symmetry graph edge");
    }
    bliss_graph.add_edge(left, right);
  }

  GraphAnalysis result;
  bliss::Stats stats;
  const unsigned int* canonical = bliss_graph.canonical_form(
      stats, [&](unsigned int vertex_count, const unsigned int* permutation) {
        if (variable_count > vertex_count) {
          throw std::runtime_error("bliss returned an invalid automorphism size");
        }
        VariablePermutation restricted(variable_count);
        for (std::size_t index = 0; index < variable_count; ++index) {
          restricted[index] = permutation[index];
        }
        result.generators.push_back(std::move(restricted));
      });
  if (canonical == nullptr) {
    throw std::runtime_error("bliss did not return a canonical labeling");
  }
  result.canonical_labels.assign(canonical, canonical + graph.colors.size());
  return result;
}

} // namespace symmetry::detail
