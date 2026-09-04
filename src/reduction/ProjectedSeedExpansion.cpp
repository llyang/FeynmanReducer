#include "reduction/ProjectedSeedExpansion.hpp"

#include "reduction/SymmetryCanonicalizer.hpp"
#include "topology/SectorUtils.hpp"

#include <algorithm>
#include <bit>
#include <format>
#include <iterator>
#include <limits>
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
