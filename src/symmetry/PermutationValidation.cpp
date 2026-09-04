#include "symmetry/detail/SymmetryInternal.hpp"

#include <algorithm>
#include <map>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace symmetry::detail {

bool is_identity(const VariablePermutation& permutation)
{
  for (std::size_t index = 0; index < permutation.size(); ++index) {
    if (permutation[index] != index) {
      return false;
    }
  }
  return true;
}

void validate_permutation(const VariablePermutation& permutation,
                          std::size_t variable_count)
{
  if (permutation.size() != variable_count) {
    throw std::runtime_error(
        "symmetry backend returned a permutation with the wrong size");
  }
  std::vector<bool> seen(variable_count, false);
  for (const auto image : permutation) {
    if (image >= variable_count || seen[image]) {
      throw std::runtime_error(
          "symmetry backend returned an invalid variable permutation");
    }
    seen[image] = true;
  }
}

void validate_lp_invariance(const VariablePermutation& permutation,
                            std::span<const PolynomialTerm> terms)
{
  std::map<std::vector<std::uint8_t>, std::vector<std::int64_t>> expected;
  for (const auto& term : terms) {
    expected.emplace(term.powers, term.weights);
  }
  for (const auto& term : terms) {
    std::vector<std::uint8_t> mapped(term.powers.size(), 0);
    for (std::size_t variable = 0; variable < term.powers.size(); ++variable) {
      mapped[permutation[variable]] = term.powers[variable];
    }
    const auto found = expected.find(mapped);
    if (found == expected.end() || found->second != term.weights) {
      throw std::runtime_error(
          "symmetry backend returned a permutation that does not preserve "
          "the LP polynomial");
    }
  }
}

void normalize_generators(std::vector<VariablePermutation>& generators,
                          std::size_t variable_count,
                          std::span<const PolynomialTerm> terms)
{
  for (const auto& generator : generators) {
    validate_permutation(generator, variable_count);
    validate_lp_invariance(generator, terms);
  }
  std::erase_if(generators, is_identity);
  std::ranges::sort(generators);
  generators.erase(std::ranges::unique(generators).begin(), generators.end());
}

} // namespace symmetry::detail
