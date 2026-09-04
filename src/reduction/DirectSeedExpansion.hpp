#pragma once

#include "reduction/EquationGenerator.hpp"

#include <cstdint>
#include <map>
#include <span>
#include <vector>

class JetEquationGenerator;
class JetSymmetryCanonicalizer;

namespace reduction::detail {

struct DirectSeedGroup {
  std::uint32_t sector = 0;
  std::vector<std::uint16_t> orders;

  bool operator==(const DirectSeedGroup&) const = default;
};

struct DirectSeedGroupLess {
  bool operator()(const DirectSeedGroup& lhs, const DirectSeedGroup& rhs) const;
};

inline constexpr unsigned maximum_direct_group_expansions = 4;

struct DirectSeedExpansion {
  std::vector<std::vector<int>> points;
  std::vector<DirectSeedGroup> groups;
};

[[nodiscard]] DirectSeedGroup
direct_seed_group(std::span<const int> powers, const JetEquationGenerator& equations,
                  const JetSymmetryCanonicalizer& canonicalizer);

[[nodiscard]] DirectSeedExpansion expand_direct_seed_groups(
    std::span<const std::vector<int>> envelope,
    std::span<const DirectSeedGroup> requested_groups,
    std::map<DirectSeedGroup, unsigned, DirectSeedGroupLess>& expansion_counts,
    const JetEquationGenerator& equations,
    const JetSymmetryCanonicalizer& canonicalizer);

} // namespace reduction::detail
