#pragma once

#include "core/Config.hpp"
#include "masters/detail/MonomialPreference.hpp"
#include "topology/IntegralLayout.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace masters::detail {

inline std::uint32_t integral_sector(const TopologyConfig& topology,
                                     const Integral& integral)
{
  const auto active = integral_layout::project_active(topology, integral);
  if (active.indices.size() > 32) {
    throw std::invalid_argument(
        "canonical master ordering supports at most 32 propagators");
  }
  std::uint32_t sector = 0;
  for (std::size_t variable = 0; variable < active.indices.size(); ++variable) {
    if (active.indices[variable] > 0) sector |= std::uint32_t{1} << variable;
  }
  return sector;
}

inline std::vector<int> integral_monomial(const TopologyConfig& topology,
                                          const Integral& integral)
{
  const auto active = integral_layout::project_active(topology, integral);
  std::vector<int> monomial;
  for (const int index : active.indices) {
    if (index > 0) monomial.push_back(index - 1);
  }
  return monomial;
}

inline bool canonical_master_integral_preferred(const TopologyConfig& topology,
                                                const Integral& lhs,
                                                const Integral& rhs)
{
  const auto lhs_sector = integral_sector(topology, lhs);
  const auto rhs_sector = integral_sector(topology, rhs);
  const auto lhs_active = std::popcount(lhs_sector);
  const auto rhs_active = std::popcount(rhs_sector);
  if (lhs_active != rhs_active) return lhs_active < rhs_active;
  if (lhs_sector != rhs_sector) return lhs_sector < rhs_sector;

  const auto lhs_monomial = integral_monomial(topology, lhs);
  const auto rhs_monomial = integral_monomial(topology, rhs);
  if (lhs_monomial != rhs_monomial) {
    return monomial_preferred(lhs_monomial, rhs_monomial);
  }
  return lhs.indices < rhs.indices;
}

inline void canonical_sort_master_integrals(const TopologyConfig& topology,
                                            std::vector<Integral>& integrals)
{
  std::ranges::sort(integrals, [&](const auto& lhs, const auto& rhs) {
    return canonical_master_integral_preferred(topology, lhs, rhs);
  });
}

} // namespace masters::detail
