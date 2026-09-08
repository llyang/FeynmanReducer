#include "reduction/EquationGenerator.hpp"
#include "topology/IntegralLayout.hpp"

#include <algorithm>
#include <bit>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <stdexcept>

namespace {

std::int64_t checked_multiply(std::int64_t lhs, std::int64_t rhs,
                              const char* description)
{
  std::int64_t result = 0;
  if (__builtin_mul_overflow(lhs, rhs, &result)) throw std::overflow_error(description);
  return result;
}

} // namespace

EquationGenerator::EquationGenerator(const Config& config,
                                     const SectorUtils& sector_utils)
    : EquationGenerator(config, sector_utils, nullptr)
{}

EquationGenerator::EquationGenerator(const Config& config,
                                     const SectorUtils& sector_utils,
                                     const TopLpTargetPlan* top_lp_target_plan)
    : cfg(config), sector(sector_utils), variable_slots(config.propagator_slots),
      polynomial_terms(config.polynomial_terms), target_plan(top_lp_target_plan)
{
  if (variable_slots.size() != cfg.propagator_count) {
    throw std::invalid_argument(
        "equation variable slots must match the propagator count");
  }
  if (variable_slots.size() > 31) {
    throw std::invalid_argument("equation support masks support at most 31 variables");
  }
  polynomial_term_support_masks.reserve(polynomial_terms.size());
  for (const auto& term : polynomial_terms) {
    if (term.powers.size() != variable_slots.size()) {
      throw std::invalid_argument(
          "equation polynomial powers must match the variable slots");
    }
    std::uint32_t support = 0;
    for (std::size_t variable = 0; variable < variable_slots.size(); ++variable) {
      if (term.powers[variable] != 0) support |= std::uint32_t{1} << variable;
    }
    polynomial_term_support_masks.push_back(support);
  }
}

std::vector<std::vector<Monomial>> EquationGenerator::build_basis_columns() const
{
  return build_integral_columns(cfg.basis);
}

std::vector<std::vector<Monomial>> EquationGenerator::build_target_columns() const
{
  if (target_plan != nullptr) return target_plan->columns;
  return build_integral_columns(cfg.targets);
}

std::vector<AnsatzSeedLayer> EquationGenerator::build_ansatz_seed_layers(
    std::span<const Integral> additional_anchors) const
{
  const unsigned layer_count =
      target_plan == nullptr ? 1U : std::max(2U, target_plan->maximum_g_shift);
  std::vector<AnsatzSeedLayer> layers(layer_count);
  for (unsigned g_shift = 0; g_shift < layer_count; ++g_shift) {
    layers[g_shift].g_shift = g_shift;
  }

  append_integral_anchors(layers.front().anchors, cfg.basis);
  append_integral_anchors(layers.front().anchors, additional_anchors);
  if (target_plan == nullptr) {
    append_integral_anchors(layers.front().anchors, cfg.targets);
    return layers;
  }

  for (const auto& column : target_plan->columns) {
    for (const auto& term : column) {
      if (term.powers.empty() || term.powers.front() > 0)
        throw std::logic_error("top-LP target has an invalid G-shift");
      const auto row_g_shift = static_cast<unsigned>(-term.powers.front());
      for (auto& layer : layers) {
        if (row_g_shift != layer.g_shift && row_g_shift != layer.g_shift + 1) continue;
        layer.anchors.emplace_back(term.powers.begin() + 1, term.powers.end());
      }
    }
  }
  return layers;
}

std::vector<std::vector<int>>
EquationGenerator::build_initial_ansatz_domain(const AnsatzSeedLayer& seed_layer) const
{
  if (seed_layer.anchors.empty()) return {};
  if (seed_layer.g_shift > static_cast<unsigned>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("ansatz seed G shift exceeds int range");
  }
  return enumerate_initial_dot_excess_domain(seed_layer.anchors, seed_layer.g_shift);
}

std::vector<Monomial>
EquationGenerator::build_ansatz_column(const std::vector<int>& powers,
                                       AnsatzFamily family,
                                       unsigned derivative_index) const
{
  if (powers.size() != variable_slots.size() + 1) {
    throw std::invalid_argument("ansatz powers have the wrong dimension");
  }
  switch (family) {
  case AnsatzFamily::Nabla:
    if (derivative_index == 0 || derivative_index > variable_slots.size()) {
      throw std::invalid_argument("nabla derivative index is out of range");
    }
    if (powers[derivative_index] < 0) return {};
    return nabla_k_generator(derivative_index, powers);
  case AnsatzFamily::Euler:
    if (derivative_index != 0) {
      throw std::invalid_argument("Euler ansatz does not take a derivative index");
    }
    return euler_generator(powers);
  case AnsatzFamily::DimensionShift:
    if (derivative_index != 0) {
      throw std::invalid_argument(
          "dimension-shift ansatz does not take a derivative index");
    }
    return dimension_shift_generator(powers);
  }
  throw std::logic_error("unknown ansatz family");
}

bool EquationGenerator::powers_are_allowed(const std::vector<int>& powers) const
{
  for (std::size_t variable = 0; variable < variable_slots.size(); ++variable) {
    const auto slot = variable_slots[variable];
    if (cfg.top_sector[slot] == 0 && powers[variable + 1] >= 0) return false;
  }
  return true;
}

const std::vector<unsigned>&
EquationGenerator::surviving_polynomial_terms(const std::vector<int>& powers,
                                              unsigned derivative_index) const
{
  if (derivative_index > variable_slots.size())
    throw std::invalid_argument("term-cache derivative is out of range");

  std::uint32_t negative_mask = 0;
  for (std::size_t variable = 0; variable < variable_slots.size(); ++variable) {
    if (powers[variable + 1] < 0) negative_mask |= std::uint32_t{1} << variable;
  }
  auto [entry, inserted] = surviving_term_cache.try_emplace(negative_mask);
  if (inserted) {
    entry->second.resize(variable_slots.size() + 1);
    for (unsigned term_index = 0; term_index < polynomial_terms.size(); ++term_index) {
      const auto& term = polynomial_terms[term_index];
      const std::uint32_t support = polynomial_term_support_masks[term_index];
      if ((support & negative_mask) == 0) entry->second.front().push_back(term_index);

      for (std::size_t derivative = 0; derivative < variable_slots.size();
           ++derivative) {
        if (term.powers[derivative] == 0) continue;
        std::uint32_t derivative_support = support;
        if (term.powers[derivative] == 1)
          derivative_support &= ~(std::uint32_t{1} << derivative);
        if ((derivative_support & negative_mask) == 0)
          entry->second[derivative + 1].push_back(term_index);
      }
    }
  }
  return entry->second[derivative_index];
}

[[nodiscard]] std::vector<Monomial>
EquationGenerator::dimension_shift_generator(const std::vector<int>& powers) const
{
  std::vector<Monomial> res;
  if (powers_are_allowed(powers) &&
      sector.is_valid_sector(sector.sector_from_powers(powers))) {
    // x^A
    res.push_back({.powers = powers,
                   .with_polynomial_coefficient = false,
                   .polynomial_term_index = 0,
                   .coeff_int = 1,
                   .coeff_minus_half_d = 0});
  }
  const auto emit_polynomial_term = [&](unsigned i) {
    const auto& B = polynomial_terms[i].powers;
    std::vector<int> new_powers = powers;
    new_powers[0] -= 1;
    std::transform(B.begin(), B.end(), new_powers.begin() + 1, new_powers.begin() + 1,
                   std::plus<>{});
    if (powers_are_allowed(new_powers) &&
        sector.is_valid_sector(sector.sector_from_powers(new_powers))) {
      // - c_i * x^{A+B_i} * G^{A[0]-1}
      res.push_back({.powers = new_powers,
                     .with_polynomial_coefficient = true,
                     .polynomial_term_index = i,
                     .coeff_int = -1,
                     .coeff_minus_half_d = 0});
    }
  };
  for (const unsigned i : surviving_polynomial_terms(powers, 0))
    emit_polynomial_term(i);
  return res;
}

[[nodiscard]] std::vector<Monomial>
EquationGenerator::nabla_k_generator(unsigned k, const std::vector<int>& powers) const
{
  std::vector<Monomial> res;

  if (powers[k] > 0) {
    std::vector<int> new_powers = powers;
    new_powers[k] -= 1;
    if (powers_are_allowed(new_powers) &&
        sector.is_valid_sector(sector.sector_from_powers(new_powers))) {
      // \partial_k x_k^powers[k]
      res.push_back({.powers = new_powers,
                     .with_polynomial_coefficient = false,
                     .polynomial_term_index = 0,
                     .coeff_int = powers[k],
                     .coeff_minus_half_d = 0});
    }
  }

  const auto emit_polynomial_term = [&](unsigned i) {
    const auto& B = polynomial_terms[i].powers;
    const int Bk = static_cast<int>(B[k - 1]);
    if (Bk > 0) {
      std::vector<int> new_powers = powers;
      new_powers[0] -= 1;
      std::transform(B.begin(), B.end(), new_powers.begin() + 1, new_powers.begin() + 1,
                     std::plus<>{});
      new_powers[k] -= 1;

      if (powers_are_allowed(new_powers) &&
          sector.is_valid_sector(sector.sector_from_powers(new_powers))) {
        res.push_back({.powers = new_powers,
                       .with_polynomial_coefficient = true,
                       .polynomial_term_index = i,
                       .coeff_int = checked_multiply(Bk, powers[0],
                                                     "nabla coefficient exceeds int64"),
                       .coeff_minus_half_d = Bk});
      }
    }
  };
  for (const unsigned i : surviving_polynomial_terms(powers, k))
    emit_polynomial_term(i);

  if (powers[k] == 0) {
    std::vector<int> new_powers = powers;
    new_powers[k] = -1;
    if (sector.is_valid_sector(sector.sector_from_powers(new_powers))) {
      // Boundary contribution supported at x_k = 0.
      res.push_back({.powers = new_powers,
                     .with_polynomial_coefficient = false,
                     .polynomial_term_index = 0,
                     .coeff_int = 1,
                     .coeff_minus_half_d = 0});
    }
  }
  return res;
}

[[nodiscard]] std::vector<Monomial>
EquationGenerator::euler_generator(const std::vector<int>& powers) const
{
  std::vector<Monomial> res;
  int num_vars = static_cast<int>(variable_slots.size());
  int sum_A_x = std::reduce(powers.begin() + 1, powers.begin() + num_vars + 1, 0);

  if (sum_A_x + num_vars != 0) {
    if (powers_are_allowed(powers) &&
        sector.is_valid_sector(sector.sector_from_powers(powers))) {
      res.push_back({.powers = powers,
                     .with_polynomial_coefficient = false,
                     .polynomial_term_index = 0,
                     .coeff_int = sum_A_x + num_vars,
                     .coeff_minus_half_d = 0});
    }
  }

  const auto emit_polynomial_term = [&](unsigned i) {
    const auto& B = polynomial_terms[i].powers;
    int sum_B = std::reduce(B.begin(), B.end(), 0);
    if (sum_B > 0) {
      std::vector<int> new_powers = powers;
      new_powers[0] -= 1;
      std::transform(B.begin(), B.end(), new_powers.begin() + 1, new_powers.begin() + 1,
                     std::plus<>{});

      if (powers_are_allowed(new_powers) &&
          sector.is_valid_sector(sector.sector_from_powers(new_powers))) {
        res.push_back({.powers = new_powers,
                       .with_polynomial_coefficient = true,
                       .polynomial_term_index = i,
                       .coeff_int = checked_multiply(sum_B, powers[0],
                                                     "Euler coefficient exceeds int64"),
                       .coeff_minus_half_d = sum_B});
      }
    }
  };
  for (const unsigned i : surviving_polynomial_terms(powers, 0))
    emit_polynomial_term(i);
  return res;
}

[[nodiscard]] std::vector<std::vector<Monomial>>
EquationGenerator::build_integral_columns(const std::vector<Integral>& integrals) const
{
  std::vector<std::vector<Monomial>> res;
  res.reserve(integrals.size());
  for (const auto& integral : integrals) {
    std::vector<int> powers;
    powers.reserve(variable_slots.size() + 1);
    powers.push_back(0);

    for (const auto slot : variable_slots)
      powers.push_back(integral.indices[slot] - 1);

    if (powers_are_allowed(powers) &&
        sector.is_valid_sector(sector.sector_from_powers(powers))) {
      res.push_back({{.powers = powers,
                      .with_polynomial_coefficient = false,
                      .polynomial_term_index = 0,
                      .coeff_int = 1,
                      .coeff_minus_half_d = 0}});
    } else {
      res.push_back({});
    }
  }
  return res;
}

void EquationGenerator::append_integral_anchors(
    std::vector<std::vector<int>>& anchors, std::span<const Integral> integrals) const
{
  for (const auto& integral : integrals) {
    std::vector<int> powers;
    powers.reserve(variable_slots.size());
    for (const auto slot : variable_slots)
      powers.push_back(integral.indices[slot] - 1);
    anchors.push_back(std::move(powers));
  }
}

std::vector<std::vector<int>> EquationGenerator::build_initial_ansatz_domain(
    std::span<const Integral> additional_anchors) const
{
  if (target_plan != nullptr) {
    throw std::logic_error(
        "single-layer ansatz domain requested with a multi-layer target plan");
  }
  const auto layers = build_ansatz_seed_layers(additional_anchors);
  return build_initial_ansatz_domain(layers.front());
}

std::vector<std::vector<int>> EquationGenerator::enumerate_initial_dot_excess_domain(
    const std::vector<std::vector<int>>& anchors, unsigned seed_g_shift) const
{
  if (anchors.empty()) return {};
  if (seed_g_shift > static_cast<unsigned>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("top-LP seed G shift exceeds int range");
  }

  std::vector<std::uint32_t> variable_bits(variable_slots.size(), 0);
  for (std::size_t variable = 0; variable < variable_slots.size(); ++variable) {
    if (cfg.top_sector[variable_slots[variable]] == 0) continue;
    std::vector<int> probe(variable_slots.size() + 1, -1);
    probe.front() = 0;
    probe[variable + 1] = 0;
    const auto bit = sector.sector_from_powers(probe);
    if (!std::has_single_bit(bit)) {
      throw std::logic_error("top-LP variable does not identify one sector bit");
    }
    variable_bits[variable] = bit;
  }

  std::map<std::uint32_t, std::uint64_t> anchor_excess;
  for (const auto& anchor : anchors) {
    if (anchor.size() != variable_slots.size()) {
      throw std::invalid_argument("top-LP seed-domain anchor has the wrong dimension");
    }
    std::vector<int> powers;
    powers.reserve(anchor.size() + 1);
    powers.push_back(-static_cast<int>(seed_g_shift));
    std::uint64_t excess = 0;
    for (std::size_t variable = 0; variable < anchor.size(); ++variable) {
      const int power = anchor[variable];
      if (power < -1) {
        throw std::logic_error(
            "projected top-LP anchor contains a negative integral index");
      }
      if (cfg.top_sector[variable_slots[variable]] == 0 && power != -1) {
        throw std::logic_error("projected top-LP anchor activates an ISP slot");
      }
      if (power > 0) {
        const auto dots = static_cast<std::uint64_t>(power);
        if (excess > std::numeric_limits<std::uint64_t>::max() - dots) {
          throw std::overflow_error("top-LP dot excess exceeds uint64");
        }
        excess += dots;
      }
      powers.push_back(power);
    }
    const auto anchor_sector = sector.sector_from_powers(powers);
    if (!sector.is_valid_sector(anchor_sector)) continue;
    auto [entry, inserted] = anchor_excess.try_emplace(anchor_sector, excess);
    if (!inserted) entry->second = std::max(entry->second, excess);
  }

  std::set<std::vector<int>> domain;
  for (const std::uint32_t seed_sector : sector.enumerate_nonzero_sectors()) {
    std::optional<std::uint64_t> inherited_excess;
    for (const auto& [anchor_sector, excess] : anchor_excess) {
      if ((seed_sector & ~anchor_sector) != 0) continue;
      const std::uint64_t pinch_halo =
          seed_g_shift == 0 && seed_sector != anchor_sector ? 1 : 0;
      if (excess > std::numeric_limits<std::uint64_t>::max() - pinch_halo) {
        throw std::overflow_error("top-LP pinched-sector dot excess exceeds uint64");
      }
      inherited_excess = std::max(inherited_excess.value_or(0), excess + pinch_halo);
    }
    if (!inherited_excess.has_value()) continue;
    if (*inherited_excess >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("top-LP dot-excess domain exceeds int range");
    }

    std::vector<std::size_t> active_variables;
    for (std::size_t variable = 0; variable < variable_bits.size(); ++variable) {
      if ((seed_sector & variable_bits[variable]) != 0)
        active_variables.push_back(variable);
    }
    if (active_variables.empty()) continue;

    std::vector<int> powers(variable_slots.size(), -1);
    auto distribute = [&](auto&& self, std::size_t position,
                          unsigned remaining) -> void {
      if (position == active_variables.size()) {
        auto point = powers;
        point.insert(point.begin(), -static_cast<int>(seed_g_shift));
        domain.insert(std::move(point));
        return;
      }
      const auto variable = active_variables[position];
      for (unsigned dots = 0; dots <= remaining; ++dots) {
        powers[variable] = static_cast<int>(dots);
        self(self, position + 1, remaining - dots);
      }
      powers[variable] = -1;
    };
    distribute(distribute, 0, static_cast<unsigned>(*inherited_excess));
  }
  return {std::make_move_iterator(domain.begin()),
          std::make_move_iterator(domain.end())};
}
