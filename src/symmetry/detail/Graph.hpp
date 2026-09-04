#pragma once

#include "core/Config.hpp"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace symmetry::detail {

struct ColoredGraph {
  std::vector<std::uint32_t> colors;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;
};

struct GraphColorPalette {
  std::vector<std::vector<std::int64_t>> coefficients;
  std::vector<std::uint8_t> exponents;
};

struct GraphAnalysis {
  std::vector<VariablePermutation> generators;
  // Maps each original vertex to its canonical vertex number.
  std::vector<std::uint32_t> canonical_labels;
};

[[nodiscard]] GraphColorPalette
make_lp_color_palette(std::span<const PolynomialTerm> terms);

[[nodiscard]] ColoredGraph
make_lp_graph(std::span<const PolynomialTerm> terms, std::size_t variable_count,
              const GraphColorPalette& palette,
              std::span<const std::uint8_t> variable_colors = {});

#if FR_HAVE_NAUTY
[[nodiscard]] GraphAnalysis analyze_nauty_graph(const ColoredGraph& graph,
                                                std::size_t variable_count);
#endif

#if FR_HAVE_BLISS
[[nodiscard]] GraphAnalysis analyze_bliss_graph(const ColoredGraph& graph,
                                                std::size_t variable_count);
#endif

} // namespace symmetry::detail
