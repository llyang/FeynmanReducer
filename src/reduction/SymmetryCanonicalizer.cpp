#include "reduction/SymmetryCanonicalizer.hpp"

#include "topology/IntegralLayout.hpp"

#include <algorithm>
#include <bit>
#include <ranges>
#include <stdexcept>
#include <unordered_set>

SymmetryCanonicalizer::SymmetryCanonicalizer(const Config& config,
                                             const SectorUtils& sectors)
    : SymmetryCanonicalizer(config, sectors,
                            integral_layout::default_variable_slots(config))
{}

SymmetryCanonicalizer::SymmetryCanonicalizer(
    const Config& config, const SectorUtils& sectors,
    std::span<const std::uint32_t> variable_slots)
    : variable_count_(static_cast<unsigned>(variable_slots.size())), sectors_(sectors)
{
  if (!config.symmetry) {
    throw std::invalid_argument("reduction requires a completed symmetry analysis");
  }

  std::vector<std::uint32_t> local_by_slot(
      std::max<std::size_t>(config.integral_count, config.propagator_count),
      variable_count_);
  for (std::uint32_t local = 0; local < variable_count_; ++local)
    local_by_slot[variable_slots[local]] = local;
  std::vector<std::uint32_t> local_by_propagator(config.propagator_count);
  const auto propagator_slots = integral_layout::default_variable_slots(config);
  for (std::uint32_t variable = 0; variable < config.propagator_count; ++variable) {
    const auto slot = propagator_slots[variable];
    if (local_by_slot[slot] == variable_count_)
      throw std::invalid_argument("symmetry view omits an active propagator");
    local_by_propagator[variable] = local_by_slot[slot];
  }

  const auto& exact = config.symmetry->generators;
  exact_generators_.reserve(exact.size());
  for (const auto& generator : exact) {
    VariablePermutation mapped(variable_count_);
    if (generator.size() != config.propagator_count)
      throw std::runtime_error("LP symmetry generator has the wrong size");
    for (std::uint32_t source = 0; source < config.propagator_count; ++source)
      mapped[local_by_propagator[source]] = local_by_propagator[generator[source]];
    exact_generators_.push_back(std::move(mapped));
  }

  for (const auto& sector_class : config.symmetry->sector_classes) {
    const std::uint32_t representative = sector_class.representative;
    if (!sectors_.is_valid_sector(representative)) {
      throw std::runtime_error("symmetry class representative is a zero sector");
    }
    const bool representative_inserted =
        representatives_.emplace(representative, representative).second;
    if (!representative_inserted) {
      throw std::runtime_error("a symmetry sector belongs to more than one class");
    }

    auto& actions = internal_actions_[representative];
    actions.reserve(sector_class.generators.size());
    for (const auto& generator : sector_class.generators) {
      if (generator.size() != variable_count_) {
        throw std::runtime_error(
            "symmetry generator size does not match the propagator count");
      }
      Action action;
      action.target_sector = representative;
      action.variable_map.reserve(
          static_cast<std::size_t>(std::popcount(representative)));
      for (std::uint32_t source = 0; source < variable_count_; ++source) {
        if (((representative >> source) & 1U) != 0) {
          action.variable_map.emplace_back(local_by_propagator[source],
                                           local_by_propagator[generator[source]]);
        }
      }
      // Validate the lifted generator immediately, including its active mask.
      std::vector<int> probe(variable_count_ + 1, -1);
      probe[0] = 0;
      for (std::uint32_t variable = 0; variable < config.propagator_count; ++variable) {
        if (((representative >> variable) & 1U) != 0)
          probe[local_by_propagator[variable] + 1] = static_cast<int>(variable);
      }
      static_cast<void>(apply(probe, action));
      actions.push_back(std::move(action));
    }

    for (const auto& relation : sector_class.relations) {
      if (!sectors_.is_valid_sector(relation.target_sector)) {
        throw std::runtime_error("symmetry relation targets a zero sector");
      }
      const auto [member, inserted] =
          representatives_.emplace(relation.target_sector, representative);
      if (!inserted && member->second != representative) {
        throw std::runtime_error("a symmetry sector belongs to more than one class");
      }

      Action inverse;
      inverse.target_sector = representative;
      inverse.variable_map.reserve(relation.variable_map.size());
      for (const auto& [source, target] : relation.variable_map) {
        inverse.variable_map.emplace_back(local_by_propagator[target],
                                          local_by_propagator[source]);
      }
      std::ranges::sort(inverse.variable_map);
      if (!to_representative_.emplace(relation.target_sector, inverse).second) {
        throw std::runtime_error(
            "a symmetry sector has more than one representative transporter");
      }

      std::vector<int> probe(variable_count_ + 1, -1);
      probe[0] = 0;
      for (std::uint32_t variable = 0; variable < config.propagator_count; ++variable) {
        if (((relation.target_sector >> variable) & 1U) != 0)
          probe[local_by_propagator[variable] + 1] = static_cast<int>(variable);
      }
      static_cast<void>(apply(probe, inverse));
    }

    std::ranges::sort(actions, [](const Action& lhs, const Action& rhs) {
      return lhs.variable_map < rhs.variable_map;
    });
    actions.erase(std::ranges::unique(actions,
                                      [](const Action& lhs, const Action& rhs) {
                                        return lhs.variable_map == rhs.variable_map;
                                      })
                      .begin(),
                  actions.end());
  }
}

std::vector<int> SymmetryCanonicalizer::apply(std::span<const int> powers,
                                              const Action& action) const
{
  if (powers.size() != static_cast<std::size_t>(variable_count_) + 1) {
    throw std::invalid_argument(
        "integral power vector size does not match the propagator count");
  }
  std::vector<int> mapped(powers.size(), -1);
  mapped[0] = powers[0];
  for (const auto& [source, target] : action.variable_map) {
    if (source >= variable_count_ || target >= variable_count_) {
      throw std::runtime_error("symmetry relation contains an invalid variable");
    }
    mapped[target + 1] = powers[source + 1];
  }
  if (sectors_.sector_from_powers(mapped) != action.target_sector) {
    throw std::runtime_error("symmetry relation produced the wrong target sector");
  }
  return mapped;
}

std::vector<int> SymmetryCanonicalizer::canonicalize(std::span<const int> powers) const
{
  if (powers.size() != static_cast<std::size_t>(variable_count_) + 1) {
    throw std::invalid_argument(
        "integral power vector size does not match the propagator count");
  }
  std::vector<int> input(powers.begin(), powers.end());
  if (const auto cached = cache_.find(input); cached != cache_.end()) {
    return cached->second;
  }

  const bool has_delta_derivative = std::ranges::any_of(
      powers | std::views::drop(1), [](int power) { return power < -1; });
  if (has_delta_derivative) {
    std::unordered_set<std::vector<int>, VectorHash> seen;
    std::vector<std::vector<int>> orbit{input};
    seen.insert(input);
    for (std::size_t cursor = 0; cursor < orbit.size(); ++cursor) {
      for (const auto& generator : exact_generators_) {
        std::vector<int> image(powers.size());
        image[0] = orbit[cursor][0];
        for (std::size_t source = 0; source < generator.size(); ++source)
          image[generator[source] + 1] = orbit[cursor][source + 1];
        if (seen.insert(image).second) orbit.push_back(std::move(image));
      }
    }
    const auto canonical = *std::ranges::min_element(orbit);
    for (const auto& image : orbit)
      cache_.insert_or_assign(image, canonical);
    return canonical;
  }

  const std::uint32_t sector = sectors_.sector_from_powers(powers);
  const auto representative = representatives_.find(sector);
  if (representative == representatives_.end()) {
    return input;
  }

  std::vector<int> transported = input;
  if (sector != representative->second) {
    const auto action = to_representative_.find(sector);
    if (action == to_representative_.end()) {
      throw std::runtime_error(
          "symmetry class member has no representative transporter");
    }
    transported = apply(powers, action->second);
  }

  std::unordered_set<std::vector<int>, VectorHash> seen;
  std::vector<std::vector<int>> orbit;
  seen.insert(transported);
  orbit.push_back(std::move(transported));
  const auto generators = internal_actions_.find(representative->second);
  if (generators != internal_actions_.end()) {
    for (std::size_t cursor = 0; cursor < orbit.size(); ++cursor) {
      for (const auto& generator : generators->second) {
        auto image = apply(orbit[cursor], generator);
        if (seen.insert(image).second) orbit.push_back(std::move(image));
      }
    }
  }

  const auto canonical = *std::ranges::min_element(orbit);
  for (const auto& image : orbit)
    cache_.insert_or_assign(image, canonical);
  cache_.insert_or_assign(std::move(input), canonical);
  return canonical;
}

void SymmetryCanonicalizer::canonicalize_grid(std::vector<std::vector<int>>& grid) const
{
  for (auto& powers : grid)
    powers = canonicalize(powers);
  std::ranges::sort(grid);
  grid.erase(std::ranges::unique(grid).begin(), grid.end());
}
