#include "reduction/ProjectedSeedExpansion.hpp"

#include "reduction/SymmetryCanonicalizer.hpp"
#include "topology/SectorUtils.hpp"

#include <algorithm>
#include <bit>
#include <format>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

namespace reduction::detail {

bool ProjectedSeedGroupLess::operator()(const ProjectedSeedGroup& lhs,
                                        const ProjectedSeedGroup& rhs) const
{
  if (lhs.g_shift != rhs.g_shift) return lhs.g_shift < rhs.g_shift;
  return std::tuple(std::popcount(lhs.sector), lhs.sector) <
         std::tuple(std::popcount(rhs.sector), rhs.sector);
}

unsigned projected_g_shift(std::span<const int> powers)
{
  if (powers.empty() || powers.front() > 0)
    throw std::logic_error("projected row has an invalid G power");
  const auto shift = -static_cast<std::int64_t>(powers.front());
  if (shift > std::numeric_limits<unsigned>::max())
    throw std::overflow_error("projected G shift exceeds unsigned range");
  return static_cast<unsigned>(shift);
}

ProjectedSeedExpansion activate_projected_seed_groups(
    std::span<const std::vector<int>> envelope,
    std::span<const ProjectedSeedGroup> residual_groups,
    const TopLpTargetPlan& targets, const EquationGenerator& equations,
    const SectorUtils& sectors, const SymmetryCanonicalizer& canonicalizer)
{
  const unsigned maximum_shift = std::max(1U, targets.maximum_g_shift);
  std::set<ProjectedSeedGroup, ProjectedSeedGroupLess> existing;
  for (const auto& seed : envelope)
    existing.insert({projected_g_shift(seed), sectors.sector_from_powers(seed)});

  // A row at q can be reached by identities seeded at q or q-1. In
  // particular, pure higher-order projections can leave an intermediate seed
  // layer empty even though elimination reports the obstruction above it.
  std::set<ProjectedSeedGroup, ProjectedSeedGroupLess> requested(
      residual_groups.begin(), residual_groups.end());
  for (const auto& key : residual_groups)
    if (key.g_shift > 1 && key.g_shift - 1 <= maximum_shift)
      requested.insert({key.g_shift - 1, key.sector});
  std::map<unsigned, std::vector<std::vector<int>>> inherited_targets;
  ProjectedSeedExpansion result;
  for (const auto& key : requested) {
    if (key.g_shift == 0 || key.g_shift > maximum_shift || existing.contains(key))
      continue;
    AnsatzSeedLayer layer{key.g_shift, {}};
    // Lower-layer seeds already contain the inherited sector caps, including any
    // completed dot frontiers. Reuse their exponents without adding another halo.
    for (const auto& seed : envelope) {
      if (projected_g_shift(seed) != key.g_shift - 1) continue;
      const auto seed_sector = sectors.sector_from_powers(seed);
      if ((seed_sector & ~key.sector) == 0)
        layer.anchors.emplace_back(seed.begin() + 1, seed.end());
    }
    auto [target_domain, inserted] = inherited_targets.try_emplace(key.g_shift);
    if (inserted) {
      AnsatzSeedLayer target_layer{key.g_shift, {}};
      for (const auto& column : targets.columns) {
        for (const auto& term : column) {
          const unsigned shift = projected_g_shift(term.powers);
          if (shift == key.g_shift || shift == key.g_shift + 1)
            target_layer.anchors.emplace_back(term.powers.begin() + 1,
                                              term.powers.end());
        }
      }
      // Inherit before symmetry transport: canonical masks need not preserve
      // inclusion. Generate each target layer at most once per activation pass.
      target_domain->second = equations.build_initial_ansatz_domain(target_layer);
      canonicalizer.canonicalize_grid(target_domain->second);
    }
    for (const auto& seed : target_domain->second) {
      if (sectors.sector_from_powers(seed) == key.sector)
        layer.anchors.emplace_back(seed.begin() + 1, seed.end());
    }
    auto points = equations.build_initial_ansatz_domain(layer);
    canonicalizer.canonicalize_grid(points);
    for (auto& point : points) {
      const ProjectedSeedGroup group{key.g_shift, sectors.sector_from_powers(point)};
      if (!existing.contains(group)) result.points.push_back(std::move(point));
    }
  }
  canonicalizer.canonicalize_grid(result.points);
  std::set<ProjectedSeedGroup, ProjectedSeedGroupLess> added;
  for (const auto& point : result.points)
    added.insert({projected_g_shift(point), sectors.sector_from_powers(point)});
  result.groups.assign(added.begin(), added.end());
  return result;
}

ProjectedSeedExpansion expand_projected_seed_groups(
    std::span<const std::vector<int>> envelope,
    std::span<const ProjectedSeedGroup> requested_groups,
    std::map<ProjectedSeedGroup, unsigned, ProjectedSeedGroupLess>& expansion_counts,
    const SectorUtils& sectors, const SymmetryCanonicalizer& canonicalizer)
{
  ProjectedSeedExpansion result;
  for (const auto& key : requested_groups) {
    const auto count = expansion_counts.find(key);
    if (count != expansion_counts.end() &&
        count->second >= maximum_projected_group_expansions) {
      throw AnsatzClosureError(std::format(
          "projected ansatz residual expansion exceeded {} frontiers for g{}:s{}",
          maximum_projected_group_expansions, key.g_shift, key.sector));
    }

    std::vector<std::vector<int>> group_points;
    for (const auto& seed : envelope) {
      const ProjectedSeedGroup seed_key{projected_g_shift(seed),
                                        sectors.sector_from_powers(seed)};
      if (seed_key != key) continue;
      for (std::size_t variable = 1; variable < seed.size(); ++variable) {
        if (seed[variable] < 0) continue;
        auto expanded = seed;
        if (expanded[variable] == std::numeric_limits<int>::max())
          throw std::overflow_error("projected residual dot frontier exceeds int range");
        ++expanded[variable];
        group_points.push_back(std::move(expanded));
      }
    }
    canonicalizer.canonicalize_grid(group_points);
    std::erase_if(group_points, [&](const auto& point) {
      const ProjectedSeedGroup point_key{projected_g_shift(point),
                                         sectors.sector_from_powers(point)};
      if (point_key != key) {
        throw std::logic_error(
            "canonical projected expansion changed its layer or sector");
      }
      return std::ranges::binary_search(envelope, point);
    });
    if (group_points.empty()) continue;

    result.points.insert(result.points.end(),
                         std::make_move_iterator(group_points.begin()),
                         std::make_move_iterator(group_points.end()));
    ++expansion_counts[key];
    result.groups.push_back(key);
  }
  return result;
}

} // namespace reduction::detail
