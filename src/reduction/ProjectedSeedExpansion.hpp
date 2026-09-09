#pragma once

#include "reduction/EquationGenerator.hpp"

#include <cstdint>
#include <map>
#include <span>
#include <vector>

class SectorUtils;
class SymmetryCanonicalizer;

namespace reduction::detail {

struct ProjectedSeedGroup {
  unsigned g_shift = 0;
  std::uint32_t sector = 0;

  bool operator==(const ProjectedSeedGroup&) const = default;
};

struct ProjectedSeedGroupLess {
  bool operator()(const ProjectedSeedGroup& lhs, const ProjectedSeedGroup& rhs) const;
};

inline constexpr unsigned maximum_projected_group_expansions = 4;

struct ProjectedSeedExpansion {
  std::vector<std::vector<int>> points;
  std::vector<ProjectedSeedGroup> groups;
};

[[nodiscard]] unsigned projected_g_shift(std::span<const int> powers);

// Activate missing seed groups at the residual or predecessor G layer. Existing groups
// are never enlarged here; dot frontiers have their own independent budget.
[[nodiscard]] ProjectedSeedExpansion activate_projected_seed_groups(
    std::span<const std::vector<int>> envelope,
    std::span<const ProjectedSeedGroup> residual_groups,
    const TopLpTargetPlan& targets, const EquationGenerator& equations,
    const SectorUtils& sectors, const SymmetryCanonicalizer& canonicalizer);

[[nodiscard]] ProjectedSeedExpansion expand_projected_seed_groups(
    std::span<const std::vector<int>> envelope,
    std::span<const ProjectedSeedGroup> requested_groups,
    std::map<ProjectedSeedGroup, unsigned, ProjectedSeedGroupLess>& expansion_counts,
    const SectorUtils& sectors, const SymmetryCanonicalizer& canonicalizer);

} // namespace reduction::detail
