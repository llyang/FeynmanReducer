#include "reduction/TopLpTargets.hpp"
#include "topology/IntegralLayout.hpp"
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
    throw std::overflow_error("projected target coefficient exceeds int64");
  return result;
}

std::int64_t checked_add(std::int64_t lhs, std::int64_t rhs)
{
  std::int64_t result = 0;
  if (__builtin_add_overflow(lhs, rhs, &result))
    throw std::overflow_error("projected target coefficient exceeds int64");
  return result;
}

std::int64_t binomial(unsigned n, unsigned k)
{
  if (k > n) return 0;
  k = std::min(k, n - k);
  std::int64_t result = 1;
  for (unsigned i = 1; i <= k; ++i) {
    result = checked_mul(result, static_cast<std::int64_t>(n - k + i));
    result /= static_cast<std::int64_t>(i);
  }
  return result;
}

std::int64_t falling_integer(std::int64_t value, unsigned degree)
{
  std::int64_t result = 1;
  for (unsigned i = 0; i < degree; ++i)
    result = checked_mul(result, checked_add(value, -static_cast<std::int64_t>(i)));
  return result;
}

struct JetState {
  std::int64_t g_shift = 0;
  unsigned g_derivatives = 0;
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
    found->second = checked_add(found->second, weight);
    if (found->second == 0) states.erase(found);
  }
}

bool violates_boundary(const JetState& state, const std::vector<bool>& boundary)
{
  for (std::size_t slot = 0; slot < boundary.size(); ++slot)
    if (boundary[slot] && state.powers[slot] != 0) return true;
  return false;
}

JetMap expand_positive_g_power(const Config& config, JetMap states,
                               const std::vector<bool>& boundary)
{
  while (std::ranges::any_of(
      states, [](const auto& entry) { return entry.first.g_shift < 0; })) {
    JetMap next;
    for (const auto& [state, weight] : states) {
      if (state.g_shift >= 0) {
        add_state(next, state, weight);
        continue;
      }
      for (std::uint32_t term_index = 0;
           term_index < config.extended_lp.polynomial_terms.size(); ++term_index) {
        const auto& term = config.extended_lp.polynomial_terms[term_index];
        auto expanded = state;
        ++expanded.g_shift;
        for (std::size_t slot = 0; slot < config.integral_count; ++slot) {
          const auto power = static_cast<int>(term.powers[slot]);
          if (__builtin_add_overflow(expanded.powers[slot], power,
                                     &expanded.powers[slot]))
            throw std::overflow_error("projected target power exceeds int range");
        }
        if (violates_boundary(expanded, boundary)) continue;
        expanded.factors.push_back(term_index);
        std::ranges::sort(expanded.factors);
        add_state(next, std::move(expanded), weight);
      }
    }
    states = std::move(next);
  }
  return states;
}

void append_shifted_falling_atoms(TopLpCoefficientExpression& expression,
                                  const JetState& state, std::int64_t weight,
                                  std::int64_t half_dimension_shift)
{
  if (state.g_derivatives > std::numeric_limits<std::uint16_t>::max())
    throw std::overflow_error("projected target falling degree exceeds uint16");
  // (z-q)_m = sum_j binomial(m,j) (z)_j (-q)_(m-j), z=-d/2.
  for (unsigned degree = 0; degree <= state.g_derivatives; ++degree) {
    auto coefficient = checked_mul(weight, binomial(state.g_derivatives, degree));
    coefficient =
        checked_mul(coefficient, falling_integer(-half_dimension_shift,
                                                 state.g_derivatives - degree));
    if (coefficient == 0) continue;
    expression.atoms.push_back(
        {coefficient, static_cast<std::uint16_t>(degree), state.factors});
  }
}

} // namespace

TopLpTargetPlan compile_top_lp_target_plan(const Config& config)
{
  TopLpTargetPlan result;
  const bool has_projected_target =
      std::ranges::any_of(config.targets, [](const Integral& target) {
        return target.dimension_shift != 0 ||
               std::ranges::any_of(target.indices, [](int index) { return index < 0; });
      });
  if (!has_projected_target) return result;
  if (config.extended_lp.polynomial_terms.empty())
    throw std::invalid_argument("projected target has no extended LP polynomial");

  std::map<TopLpCoefficientExpression, std::uint32_t> expression_ids;
  SectorUtils sectors(config, config.propagator_slots);
  result.columns.reserve(config.targets.size());
  for (const auto& target : config.targets) {
    integral_layout::validate(config, target);
    const std::int64_t half_dimension_shift = target.dimension_shift / 2;
    std::vector<unsigned> derivative_slots;
    std::vector<bool> boundary(config.integral_count, false);
    unsigned total_order = 0;
    for (std::size_t slot = 0; slot < config.integral_count; ++slot) {
      const int index = target.indices[slot];
      if (index <= 0) boundary[slot] = true;
      if (index < 0) {
        const auto order = static_cast<unsigned>(-static_cast<std::int64_t>(index));
        if (order > std::numeric_limits<std::uint16_t>::max())
          throw std::overflow_error("projected target derivative order exceeds uint16");
        if (order > std::numeric_limits<unsigned>::max() - total_order)
          throw std::overflow_error("negative-index derivative order exceeds unsigned");
        total_order += order;
        derivative_slots.insert(derivative_slots.end(), order,
                                static_cast<unsigned>(slot));
      }
    }

    JetMap states;
    states.emplace(
        JetState{
            half_dimension_shift, 0, std::vector<int>(config.integral_count, 0), {}},
        1);
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
          differentiated.g_shift = checked_add(differentiated.g_shift, 1);
          if (differentiated.g_derivatives == std::numeric_limits<unsigned>::max())
            throw std::overflow_error("projected target derivative count overflow");
          ++differentiated.g_derivatives;
          for (std::size_t slot = 0; slot < config.integral_count; ++slot) {
            const auto power = static_cast<int>(term.powers[slot]);
            if (__builtin_add_overflow(differentiated.powers[slot], power,
                                       &differentiated.powers[slot]))
              throw std::overflow_error("projected target power exceeds int range");
          }
          --differentiated.powers[derivative_slot];
          differentiated.factors.push_back(term_index);
          std::ranges::sort(differentiated.factors);
          add_state(next, std::move(differentiated),
                    checked_mul(state_weight, exponent));
        }
      }
      states = std::move(next);
    }
    states = expand_positive_g_power(config, std::move(states), boundary);

    using RowKey = std::vector<int>;
    std::map<RowKey, TopLpCoefficientExpression> rows;
    for (auto& [state, state_weight] : states) {
      if (violates_boundary(state, boundary)) continue;
      std::int64_t weight = (total_order & 1U) == 0 ? state_weight : -state_weight;
      if (state.g_shift < 0 ||
          state.g_shift > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("projected target G shift exceeds int range");

      RowKey powers;
      powers.reserve(config.propagator_count + 1);
      powers.push_back(-static_cast<int>(state.g_shift));
      for (std::size_t variable = 0; variable < config.propagator_count; ++variable) {
        const auto slot = config.propagator_slots[variable];
        const int base = target.indices[slot] > 0 ? target.indices[slot] - 1 : -1;
        int power = 0;
        if (__builtin_add_overflow(base, state.powers[slot], &power))
          throw std::overflow_error("projected target power exceeds int range");
        powers.push_back(power);
      }
      auto& expression = rows[std::move(powers)];
      append_shifted_falling_atoms(expression, state, weight, half_dimension_shift);
    }

    std::vector<Monomial> column;
    column.reserve(rows.size());
    for (auto& [powers, expression] : rows) {
      if (expression.atoms.empty()) continue;
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
    result.columns.push_back(std::move(column));
  }
  return result;
}
