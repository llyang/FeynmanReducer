#include "reduction/TopLpTargets.hpp"
#include "topology/SectorUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace {

std::int64_t checked_mul(std::int64_t lhs, std::int64_t rhs)
{
  std::int64_t result = 0;
  if (__builtin_mul_overflow(lhs, rhs, &result))
    throw std::overflow_error("top-LP target coefficient exceeds int64");
  return result;
}

struct JetState {
  unsigned g_shift = 0;
  std::vector<int> powers;
  std::vector<std::uint32_t> factors;

  auto operator<=>(const JetState&) const = default;
};

using JetMap = std::map<JetState, std::int64_t>;

void add_state(JetMap& states, JetState state, std::int64_t weight)
{
  if (weight == 0) return;
  auto [found, inserted] = states.try_emplace(std::move(state), weight);
  if (!inserted) {
    std::int64_t sum = 0;
    if (__builtin_add_overflow(found->second, weight, &sum))
      throw std::overflow_error("top-LP target coefficient exceeds int64");
    found->second = sum;
    if (sum == 0) states.erase(found);
  }
}

} // namespace

TopLpTargetPlan compile_top_lp_target_plan(const Config& config)
{
  TopLpTargetPlan result;
  const bool has_negative_target =
      std::ranges::any_of(config.targets, [](const Integral& target) {
        return std::ranges::any_of(target.indices, [](int index) { return index < 0; });
      });
  if (!has_negative_target) return result;
  if (config.extended_lp.polynomial_terms.empty())
    throw std::invalid_argument("top-LP projection has no extended LP polynomial");

  std::map<TopLpCoefficientExpression, std::uint32_t> expression_ids;
  SectorUtils sectors(config, config.propagator_slots);
  result.columns.reserve(config.targets.size());
  for (std::size_t target_index = 0; target_index < config.targets.size();
       ++target_index) {
    const auto& target = config.targets[target_index];
    std::vector<unsigned> derivative_slots;
    std::vector<bool> boundary(config.integral_count, false);
    unsigned total_order = 0;
    for (std::size_t slot = 0; slot < config.integral_count; ++slot) {
      const int index = target.indices[slot];
      if (index <= 0) boundary[slot] = true;
      if (index < 0) {
        const auto order = static_cast<unsigned>(-static_cast<std::int64_t>(index));
        if (order > std::numeric_limits<std::uint16_t>::max())
          throw std::overflow_error("top-LP target derivative order exceeds uint16");
        if (order > std::numeric_limits<unsigned>::max() - total_order)
          throw std::overflow_error("negative-index derivative order exceeds unsigned");
        total_order += order;
        derivative_slots.insert(derivative_slots.end(), order,
                                static_cast<unsigned>(slot));
      }
    }

    JetMap states;
    states.emplace(JetState{0, std::vector<int>(config.integral_count, 0), {}}, 1);
    for (const unsigned derivative_slot : derivative_slots) {
      JetMap next;
      for (const auto& [state, state_weight] : states) {
        if (state.powers[derivative_slot] > 0) {
          auto differentiated = state;
          const int factor = differentiated.powers[derivative_slot]--;
          add_state(next, std::move(differentiated), checked_mul(state_weight, factor));
        }
        for (std::uint32_t term_index = 0;
             term_index < config.extended_lp.polynomial_terms.size(); ++term_index) {
          const auto& term = config.extended_lp.polynomial_terms[term_index];
          const auto exponent = term.powers[derivative_slot];
          if (exponent == 0) continue;
          auto differentiated = state;
          if (differentiated.g_shift == std::numeric_limits<unsigned>::max())
            throw std::overflow_error("top-LP target G shift exceeds unsigned");
          ++differentiated.g_shift;
          for (std::size_t slot = 0; slot < config.integral_count; ++slot)
            differentiated.powers[slot] += term.powers[slot];
          --differentiated.powers[derivative_slot];
          differentiated.factors.push_back(term_index);
          std::ranges::sort(differentiated.factors);
          add_state(next, std::move(differentiated),
                    checked_mul(state_weight, exponent));
        }
      }
      states = std::move(next);
    }

    using RowKey = std::vector<int>;
    std::map<RowKey, TopLpCoefficientExpression> rows;
    for (auto& [state, weight] : states) {
      bool survives = true;
      for (std::size_t slot = 0; slot < boundary.size(); ++slot) {
        if (boundary[slot] && state.powers[slot] != 0) {
          survives = false;
          break;
        }
      }
      if (!survives) continue;
      if ((total_order & 1U) != 0) weight = -weight;

      RowKey powers;
      powers.reserve(config.propagator_count + 1);
      powers.push_back(-static_cast<int>(state.g_shift));
      for (std::size_t variable = 0; variable < config.propagator_count; ++variable) {
        const auto slot = config.propagator_slots[variable];
        const int base = target.indices[slot] > 0 ? target.indices[slot] - 1 : -1;
        powers.push_back(base + state.powers[slot]);
      }
      if (state.g_shift > std::numeric_limits<std::uint16_t>::max())
        throw std::overflow_error("top-LP target falling degree exceeds uint16");
      auto& expression = rows[std::move(powers)];
      expression.atoms.push_back({weight, static_cast<std::uint16_t>(state.g_shift),
                                  std::move(state.factors)});
    }

    std::vector<Monomial> column;
    column.reserve(rows.size());
    for (auto& [powers, expression] : rows) {
      const auto sector = sectors.sector_from_powers(powers);
      if (!sectors.is_valid_sector(sector)) continue;
      std::ranges::sort(expression.atoms);
      const auto [found, inserted] = expression_ids.try_emplace(
          expression, static_cast<std::uint32_t>(result.expressions.size()));
      if (inserted) result.expressions.push_back(expression);
      column.push_back({.powers = std::move(powers),
                        .with_polynomial_coefficient = false,
                        .polynomial_term_index = 0,
                        .coeff_int = 1,
                        .coeff_minus_half_d = 0,
                        .coefficient_expression = found->second});
      result.maximum_g_shift = std::max(
          result.maximum_g_shift, static_cast<unsigned>(-column.back().powers.front()));
    }
    result.columns.push_back(column);
  }
  return result;
}
