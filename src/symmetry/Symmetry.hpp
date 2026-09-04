#pragma once

#include "core/Config.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace symmetry {

[[nodiscard]] std::string_view backend_name(SymmetryBackend backend);

[[nodiscard]] bool backend_available(SymmetryBackend backend);

[[nodiscard]] std::vector<SymmetryBackend> available_backends();

[[nodiscard]] std::string available_backends_string();

[[nodiscard]] std::vector<VariablePermutation>
find_lp_symmetry_generators(std::span<const PolynomialTerm> terms,
                            std::size_t variable_count, SymmetryBackend backend,
                            std::span<const std::uint8_t> variable_colors = {});

struct SubsectorSymmetryAnalysis {
  std::size_t nonzero_sector_count = 0;
  std::vector<SectorSymmetryClass> classes;
};

[[nodiscard]] SubsectorSymmetryAnalysis
find_subsector_symmetry_classes(const TopologyConfig& topology, SymmetryBackend backend,
                                std::span<const VariablePermutation> global_generators);

// Returns the first monomial index in each orbit under automorphisms of the
// sector-restricted LP polynomial. Monomial powers follow the sector's active
// variables in ascending global variable order.
[[nodiscard]] std::vector<std::size_t> find_sector_monomial_orbit_representatives(
    const TopologyConfig& topology, std::uint32_t sector,
    std::span<const std::vector<int>> monomials, SymmetryBackend backend);

} // namespace symmetry
