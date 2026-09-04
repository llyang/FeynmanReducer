#include "reduction/JetSymmetryCanonicalizer.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <unordered_set>

JetSymmetryCanonicalizer::JetSymmetryCanonicalizer(
    const Config& config, const SymmetryCanonicalizer& denominator)
    : config_(config), denominator_(denominator)
{}

std::vector<int>
JetSymmetryCanonicalizer::canonicalize(std::span<const int> powers) const
{
  if (powers.size() != config_.integral_count + 1)
    throw std::invalid_argument("jet symmetry row has the wrong dimension");
  std::vector<int> input(powers.begin(), powers.end());
  if (const auto found = cache_.find(input); found != cache_.end())
    return found->second;

  const bool has_jet = std::ranges::any_of(powers | std::views::drop(1),
                                           [](int power) { return power < -1; });
  if (!has_jet) {
    std::vector<int> denominator(config_.propagator_count + 1, -1);
    denominator[0] = powers[0];
    for (std::size_t variable = 0; variable < config_.propagator_count; ++variable)
      denominator[variable + 1] = powers[config_.propagator_slots[variable] + 1];
    denominator = denominator_.canonicalize(denominator);
    std::vector<int> result(config_.integral_count + 1, -1);
    result[0] = denominator[0];
    for (std::size_t variable = 0; variable < config_.propagator_count; ++variable)
      result[config_.propagator_slots[variable] + 1] = denominator[variable + 1];
    cache_.emplace(std::move(input), result);
    return result;
  }

  std::unordered_set<std::vector<int>, VectorHash> seen;
  std::vector<std::vector<int>> orbit{input};
  seen.insert(input);
  for (std::size_t cursor = 0; cursor < orbit.size(); ++cursor) {
    for (const auto& generator : config_.extended_lp.symmetry_generators) {
      if (generator.size() != config_.integral_count)
        throw std::runtime_error("extended LP symmetry generator has the wrong size");
      std::vector<int> image(powers.size());
      image[0] = orbit[cursor][0];
      for (std::size_t source = 0; source < generator.size(); ++source)
        image[generator[source] + 1] = orbit[cursor][source + 1];
      if (seen.insert(image).second) orbit.push_back(std::move(image));
    }
  }
  const auto result = *std::ranges::min_element(orbit);
  for (auto& image : orbit)
    cache_.insert_or_assign(std::move(image), result);
  return result;
}
