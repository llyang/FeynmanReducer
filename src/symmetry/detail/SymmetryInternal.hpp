#pragma once

#include "core/Config.hpp"
#include "symmetry/detail/Graph.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace symmetry::detail {

void validate_permutation(const VariablePermutation& permutation,
                          std::size_t variable_count);
void validate_lp_invariance(const VariablePermutation& permutation,
                            std::span<const PolynomialTerm> terms);
void normalize_generators(std::vector<VariablePermutation>& generators,
                          std::size_t variable_count,
                          std::span<const PolynomialTerm> terms);
void validate_canonical_labels(std::span<const std::uint32_t> labels,
                               std::size_t vertex_count);
[[nodiscard]] GraphAnalysis analyze_graph(const ColoredGraph& graph,
                                          std::size_t variable_count,
                                          SymmetryBackend backend);

} // namespace symmetry::detail
