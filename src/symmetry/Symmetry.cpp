#include "symmetry/Symmetry.hpp"

#include "symmetry/detail/Graph.hpp"
#include "symmetry/detail/SymmetryInternal.hpp"

#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace symmetry {

using detail::analyze_graph;
using detail::normalize_generators;

std::string_view backend_name(SymmetryBackend backend)
{
  switch (backend) {
  case SymmetryBackend::None:
    return "none";
  case SymmetryBackend::Nauty:
    return "nauty";
  case SymmetryBackend::Bliss:
    return "bliss";
  }
  throw std::runtime_error("unknown symmetry backend");
}

bool backend_available(SymmetryBackend backend)
{
  switch (backend) {
  case SymmetryBackend::None:
    return true;
  case SymmetryBackend::Nauty:
#if FR_HAVE_NAUTY
    return true;
#else
    return false;
#endif
  case SymmetryBackend::Bliss:
#if FR_HAVE_BLISS
    return true;
#else
    return false;
#endif
  }
  return false;
}

std::vector<SymmetryBackend> available_backends()
{
  std::vector<SymmetryBackend> result{SymmetryBackend::None};
#if FR_HAVE_NAUTY
  result.push_back(SymmetryBackend::Nauty);
#endif
#if FR_HAVE_BLISS
  result.push_back(SymmetryBackend::Bliss);
#endif
  return result;
}

std::string available_backends_string()
{
  const auto backends = available_backends();
  std::string result;
  for (std::size_t index = 0; index < backends.size(); ++index) {
    if (index != 0) {
      result += ", ";
    }
    result += backend_name(backends[index]);
  }
  return result;
}

std::vector<VariablePermutation>
find_lp_symmetry_generators(std::span<const PolynomialTerm> terms,
                            std::size_t variable_count, SymmetryBackend backend,
                            std::span<const std::uint8_t> variable_colors)
{
  if (backend == SymmetryBackend::None) return {};
  if (!backend_available(backend)) {
    throw std::runtime_error(
        std::format("requested symmetry backend '{}' is not available in this build "
                    "(available: {})",
                    backend_name(backend), available_backends_string()));
  }
  const auto palette = detail::make_lp_color_palette(terms);
  const auto graph =
      detail::make_lp_graph(terms, variable_count, palette, variable_colors);
  auto analysis = analyze_graph(graph, variable_count, backend);
  normalize_generators(analysis.generators, variable_count, terms);
  return analysis.generators;
}

} // namespace symmetry
