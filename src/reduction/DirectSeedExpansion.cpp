#include "reduction/DirectSeedExpansion.hpp"

#include "reduction/JetEquationGenerator.hpp"
#include "reduction/JetSymmetryCanonicalizer.hpp"

#include <algorithm>
#include <bit>
#include <format>
#include <iterator>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <tuple>

namespace reduction::detail {

namespace {

std::uint64_t total_order(std::span<const std::uint16_t> orders)
{
  return std::reduce(orders.begin(), orders.end(), std::uint64_t{0});
}

std::string format_group(const DirectSeedGroup& group)
{
  std::string orders;
  for (const auto order : group.orders) {
    if (!orders.empty()) orders += ':';
    orders += std::to_string(order);
  }
  return std::format("s{}:j[{}]", group.sector, orders);
}

} // namespace

bool DirectSeedGroupLess::operator()(const DirectSeedGroup& lhs,
                                     const DirectSeedGroup& rhs) const
{
  return std::tuple(total_order(lhs.orders), std::popcount(lhs.sector), lhs.sector,
                    lhs.orders) < std::tuple(total_order(rhs.orders),
                                             std::popcount(rhs.sector), rhs.sector,
                                             rhs.orders);
}

DirectSeedGroup direct_seed_group(std::span<const int> powers,
                                  const JetEquationGenerator& equations,
                                  const JetSymmetryCanonicalizer& canonicalizer)
{
  std::vector<int> skeleton(powers.begin(), powers.end());
  skeleton.front() = 0;
  for (int& power : skeleton | std::views::drop(1)) {
    if (power >= 0) power = 0;
  }
  skeleton = canonicalizer.canonicalize(skeleton);
  return {equations.sector_from_powers(skeleton), equations.jet_orders(skeleton)};
}

DirectSeedExpansion expand_direct_seed_groups(
    std::span<const std::vector<int>> envelope,
    std::span<const DirectSeedGroup> requested_groups,
    std::map<DirectSeedGroup, unsigned, DirectSeedGroupLess>& expansion_counts,
    const JetEquationGenerator& equations,
    const JetSymmetryCanonicalizer& canonicalizer)
{
  DirectSeedExpansion result;
  for (const auto& key : requested_groups) {
    const auto count = expansion_counts.find(key);
    if (count != expansion_counts.end() &&
        count->second >= maximum_direct_group_expansions) {
      throw AnsatzClosureError(
          std::format("direct ansatz residual expansion exceeded {} frontiers for {}",
                      maximum_direct_group_expansions, format_group(key)));
    }

    std::vector<std::vector<int>> group_points;
    for (const auto& seed : envelope) {
      if (direct_seed_group(seed, equations, canonicalizer) != key) continue;
      for (std::size_t slot = 1; slot < seed.size(); ++slot) {
        if (seed[slot] < 0) continue;
        if (seed[slot] == std::numeric_limits<int>::max())
          throw std::overflow_error("direct residual dot frontier exceeds int range");
        auto expanded = seed;
        ++expanded[slot];
        expanded = canonicalizer.canonicalize(expanded);
        if (direct_seed_group(expanded, equations, canonicalizer) != key) {
          throw std::logic_error(
              "canonical direct expansion changed its jet group or sector");
        }
        group_points.push_back(std::move(expanded));
      }
    }
    std::ranges::sort(group_points);
    group_points.erase(std::unique(group_points.begin(), group_points.end()),
                       group_points.end());
    std::erase_if(group_points, [&](const auto& point) {
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
