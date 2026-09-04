#include "symmetry/detail/SymmetryInternal.hpp"

#include <stdexcept>
#include <vector>

namespace symmetry::detail {

void validate_canonical_labels(std::span<const std::uint32_t> labels,
                               std::size_t vertex_count)
{
  if (labels.size() != vertex_count) {
    throw std::runtime_error(
        "symmetry backend returned a canonical labeling with the wrong size");
  }
  std::vector<bool> seen(vertex_count, false);
  for (const auto label : labels) {
    if (label >= vertex_count || seen[label]) {
      throw std::runtime_error(
          "symmetry backend returned an invalid canonical labeling");
    }
    seen[label] = true;
  }
}

symmetry::detail::GraphAnalysis
analyze_graph(const symmetry::detail::ColoredGraph& graph,
              [[maybe_unused]] std::size_t variable_count, SymmetryBackend backend)
{
  symmetry::detail::GraphAnalysis result;
  switch (backend) {
  case SymmetryBackend::None:
    throw std::logic_error("none symmetry backend cannot analyze a graph");
  case SymmetryBackend::Nauty:
#if FR_HAVE_NAUTY
    result = symmetry::detail::analyze_nauty_graph(graph, variable_count);
    break;
#else
    throw std::runtime_error("nauty backend was not compiled");
#endif
  case SymmetryBackend::Bliss:
#if FR_HAVE_BLISS
    result = symmetry::detail::analyze_bliss_graph(graph, variable_count);
    break;
#else
    throw std::runtime_error("bliss backend was not compiled");
#endif
  }
  validate_canonical_labels(result.canonical_labels, graph.colors.size());
  return result;
}

} // namespace symmetry::detail
