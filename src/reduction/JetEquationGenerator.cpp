#include "reduction/JetEquationGenerator.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

namespace {

constexpr auto maximum_jet_order = std::numeric_limits<std::uint16_t>::max();
static_assert(std::numeric_limits<int>::max() >=
                      static_cast<std::int64_t>(maximum_jet_order) &&
                  std::numeric_limits<int>::lowest() <=
                      -1 - static_cast<std::int64_t>(maximum_jet_order),
              "int cannot represent the full uint16 jet-power encoding");

std::int64_t checked_mul(std::int64_t lhs, std::int64_t rhs)
{
  std::int64_t result = 0;
  if (__builtin_mul_overflow(lhs, rhs, &result))
    throw std::overflow_error("jet coefficient exceeds int64");
  return result;
}

std::int64_t checked_add(std::int64_t lhs, std::int64_t rhs)
{
  std::int64_t result = 0;
  if (__builtin_add_overflow(lhs, rhs, &result))
    throw std::overflow_error("jet coefficient exceeds int64");
  return result;
}

std::size_t capped_composition_count(unsigned cap, std::size_t slots, std::size_t limit)
{
  // The number of nonnegative slot assignments with total at most cap is
  // binomial(cap + slots, slots). Stop as soon as the caller's useful range is
  // exceeded so even deliberately unreasonable caps cannot overflow.
  std::size_t result = 1;
  for (std::size_t index = 1; index <= slots; ++index) {
    if (index > std::numeric_limits<std::size_t>::max() - cap) return limit + 1;
    std::size_t factor = cap + index;
    std::size_t divisor = index;
    const auto factor_gcd = std::gcd(factor, divisor);
    factor /= factor_gcd;
    divisor /= factor_gcd;
    const auto result_gcd = std::gcd(result, divisor);
    const auto reduced_result = result / result_gcd;
    divisor /= result_gcd;
    if (divisor != 1 || (factor != 0 && reduced_result > limit / factor)) {
      return limit + 1;
    }
    result = reduced_result * factor;
  }
  return result;
}

} // namespace

JetEquationGenerator::JetEquationGenerator(const Config& config,
                                           const SectorUtils& sectors)
    : config_(config), sectors_(sectors)
{
  if (config_.extended_lp.polynomial_terms.empty())
    throw std::invalid_argument("jet equations require the extended LP polynomial");
}

std::vector<std::vector<int>> JetEquationGenerator::get_rank_dot_ansatz_domain(
    std::span<const Integral> additional_anchors, unsigned pinched_dot_halo) const
{
  struct Profile {
    std::uint32_t sector = 0;
    std::uint64_t dot = 0;
    std::vector<unsigned> rank;
  };

  std::vector<Profile> anchor_profiles;
  const auto append_anchors = [&](std::span<const Integral> integrals) {
    for (const auto& integral : integrals) {
      if (integral.indices.size() != config_.integral_count)
        throw std::invalid_argument("direct seed anchor has the wrong dimension");
      Profile profile{0, 0, std::vector<unsigned>(config_.integral_count, 0)};
      for (std::size_t variable = 0; variable < config_.propagator_count; ++variable) {
        const auto slot = config_.propagator_slots[variable];
        const int index = integral.indices[slot];
        if (index <= 0) continue;
        profile.sector |= std::uint32_t{1} << variable;
        const auto excess = static_cast<std::uint64_t>(index - 1);
        if (profile.dot > std::numeric_limits<std::uint64_t>::max() - excess)
          throw std::overflow_error("direct seed anchor dot excess exceeds uint64");
        profile.dot += excess;
      }
      for (std::size_t slot = 0; slot < config_.integral_count; ++slot) {
        const int index = integral.indices[slot];
        if (config_.top_sector[slot] == 0 && index > 0)
          throw std::invalid_argument("positive ISP indices are not supported");
        if (index == std::numeric_limits<int>::min())
          throw std::overflow_error("direct seed anchor rank exceeds int range");
        if (index < 0) profile.rank[slot] = static_cast<unsigned>(-index);
      }
      if (sectors_.is_valid_sector(profile.sector))
        anchor_profiles.push_back(std::move(profile));
    }
  };
  append_anchors(config_.basis);
  append_anchors(config_.targets);
  append_anchors(additional_anchors);

  std::map<std::uint32_t, std::vector<Profile>> sector_profiles;
  for (const std::uint32_t seed_sector : sectors_.enumerate_nonzero_sectors()) {
    auto& inherited = sector_profiles[seed_sector];
    for (const auto& anchor : anchor_profiles) {
      if ((seed_sector & ~anchor.sector) != 0) continue;
      const std::uint64_t halo = seed_sector != anchor.sector ? pinched_dot_halo : 0;
      if (anchor.dot > std::numeric_limits<std::uint64_t>::max() - halo)
        throw std::overflow_error("direct pinched dot cap exceeds uint64");
      inherited.push_back({seed_sector, anchor.dot + halo, anchor.rank});
    }
    std::ranges::sort(inherited, [](const Profile& lhs, const Profile& rhs) {
      if (lhs.dot != rhs.dot) return lhs.dot > rhs.dot;
      return lhs.rank > rhs.rank;
    });
    std::vector<Profile> maximal;
    for (auto& profile : inherited) {
      const bool dominated = std::ranges::any_of(maximal, [&](const Profile& other) {
        if (other.dot < profile.dot) return false;
        for (std::size_t slot = 0; slot < profile.rank.size(); ++slot) {
          if (other.rank[slot] < profile.rank[slot]) return false;
        }
        return true;
      });
      if (!dominated) maximal.push_back(std::move(profile));
    }
    inherited = std::move(maximal);
  }

  constexpr std::size_t maximum_raw_seeds = 1'000'000;
  std::set<std::vector<int>> domain;
  std::size_t estimated_raw_seeds = 0;
  for (const auto& [sector, profiles] : sector_profiles) {
    std::vector<std::size_t> active_slots;
    std::vector<std::size_t> boundary_slots;
    std::vector<bool> active(config_.integral_count, false);
    for (std::size_t variable = 0; variable < config_.propagator_count; ++variable) {
      const auto slot = config_.propagator_slots[variable];
      if (((sector >> variable) & 1U) != 0) {
        active_slots.push_back(slot);
        active[slot] = true;
      }
    }
    for (std::size_t slot = 0; slot < config_.integral_count; ++slot) {
      if (!active[slot]) boundary_slots.push_back(slot);
    }

    for (const auto& profile : profiles) {
      if (profile.dot > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("direct dot cap exceeds int range");
      }
      const auto remaining_capacity = maximum_raw_seeds - estimated_raw_seeds;
      const auto dot_count = capped_composition_count(
          static_cast<unsigned>(profile.dot), active_slots.size(), remaining_capacity);
      if (dot_count > remaining_capacity)
        throw std::runtime_error("direct rank/dot domain exceeds one million seeds");
      std::size_t rank_count = 1;
      for (const std::size_t slot : boundary_slots) {
        const std::size_t factor = static_cast<std::size_t>(profile.rank[slot]) + 1;
        if (rank_count > remaining_capacity / factor)
          throw std::runtime_error("direct rank/dot domain exceeds one million seeds");
        rank_count *= factor;
      }
      if (dot_count != 0 && rank_count > remaining_capacity / dot_count)
        throw std::runtime_error("direct rank/dot domain exceeds one million seeds");
      estimated_raw_seeds += dot_count * rank_count;

      std::vector<int> powers(config_.integral_count + 1, -1);
      powers.front() = 0;
      const auto enumerate_rank = [&](auto&& self, std::size_t position) -> void {
        if (position == boundary_slots.size()) {
          if (domain.size() >= maximum_raw_seeds)
            throw std::runtime_error(
                "direct rank/dot domain exceeds one million seeds");
          domain.insert(powers);
          return;
        }
        const auto slot = boundary_slots[position];
        for (unsigned order = 0; order <= profile.rank[slot]; ++order) {
          powers[slot + 1] = -1 - static_cast<int>(order);
          self(self, position + 1);
        }
        powers[slot + 1] = -1;
      };
      const auto enumerate_dot = [&](auto&& self, std::size_t position,
                                     unsigned remaining) -> void {
        if (position == active_slots.size()) {
          enumerate_rank(enumerate_rank, 0);
          return;
        }
        const auto slot = active_slots[position];
        for (unsigned dots = 0; dots <= remaining; ++dots) {
          powers[slot + 1] = static_cast<int>(dots);
          self(self, position + 1, remaining - dots);
        }
        powers[slot + 1] = -1;
      };
      enumerate_dot(enumerate_dot, 0, static_cast<unsigned>(profile.dot));
    }
  }
  return {domain.begin(), domain.end()};
}

std::vector<int>
JetEquationGenerator::seed_powers(std::span<const int> propagator_powers,
                                  std::span<const std::uint16_t> orders) const
{
  if (propagator_powers.size() != config_.propagator_count + 1 ||
      orders.size() != config_.integral_count) {
    throw std::invalid_argument("jet seed has the wrong dimension");
  }
  std::vector<int> result(config_.integral_count + 1, -1);
  result[0] = propagator_powers[0];
  for (std::size_t variable = 0; variable < config_.propagator_count; ++variable) {
    const auto slot = config_.propagator_slots[variable];
    result[slot + 1] = propagator_powers[variable + 1];
  }
  for (std::size_t slot = 0; slot < orders.size(); ++slot) {
    if (orders[slot] == 0) continue;
    if (result[slot + 1] >= 0)
      throw std::invalid_argument("a jet derivative is attached to an active variable");
    result[slot + 1] = -1 - static_cast<int>(orders[slot]);
  }
  return result;
}

std::uint32_t
JetEquationGenerator::sector_from_powers(std::span<const int> powers) const
{
  if (powers.size() != config_.integral_count + 1)
    throw std::invalid_argument("jet row has the wrong dimension");
  std::uint32_t sector = 0;
  for (std::size_t variable = 0; variable < config_.propagator_count; ++variable) {
    if (powers[config_.propagator_slots[variable] + 1] >= 0)
      sector |= std::uint32_t{1} << variable;
  }
  return sector;
}

std::vector<std::uint16_t>
JetEquationGenerator::jet_orders(std::span<const int> powers) const
{
  if (powers.size() != config_.integral_count + 1)
    throw std::invalid_argument("jet row has the wrong dimension");
  std::vector<std::uint16_t> result(config_.integral_count, 0);
  for (std::size_t slot = 0; slot < config_.integral_count; ++slot) {
    if (powers[slot + 1] >= -1) continue;
    const std::int64_t order = -1 - static_cast<std::int64_t>(powers[slot + 1]);
    if (order > std::numeric_limits<std::uint16_t>::max())
      throw std::overflow_error("jet order exceeds uint16");
    result[slot] = static_cast<std::uint16_t>(order);
  }
  return result;
}

bool JetEquationGenerator::row_is_allowed(std::span<const int> powers) const
{
  if (powers.size() != config_.integral_count + 1) return false;
  for (std::size_t slot = 0; slot < config_.integral_count; ++slot) {
    if (config_.top_sector[slot] == 0 && powers[slot + 1] >= 0) return false;
  }
  return sectors_.is_valid_sector(sector_from_powers(powers));
}

std::optional<JetEquationGenerator::Product>
JetEquationGenerator::multiply(std::span<const int> powers,
                               std::span<const std::uint8_t> monomial) const
{
  if (powers.size() != config_.integral_count + 1 ||
      monomial.size() != config_.integral_count) {
    throw std::invalid_argument("jet product has the wrong dimension");
  }
  Product result{std::vector<int>(powers.begin(), powers.end()), 1};
  for (std::size_t slot = 0; slot < monomial.size(); ++slot) {
    const unsigned exponent = monomial[slot];
    if (exponent == 0) continue;
    int& power = result.powers[slot + 1];
    if (power >= 0) {
      if (power > std::numeric_limits<int>::max() - static_cast<int>(exponent))
        throw std::overflow_error("jet monomial power exceeds int range");
      power += static_cast<int>(exponent);
      continue;
    }
    const std::int64_t order = -1 - static_cast<std::int64_t>(power);
    if (static_cast<std::int64_t>(exponent) > order) return std::nullopt;
    for (unsigned factor = 0; factor < exponent; ++factor) {
      result.coefficient =
          checked_mul(result.coefficient, -(order - static_cast<std::int64_t>(factor)));
    }
    power += static_cast<int>(exponent);
  }
  return result;
}

std::vector<Monomial>
JetEquationGenerator::dimension_shift(const std::vector<int>& powers) const
{
  std::vector<Monomial> result;
  if (row_is_allowed(powers)) result.push_back({powers, false, 0, 1, 0});
  for (unsigned term = 0; term < config_.extended_lp.polynomial_terms.size(); ++term) {
    auto product = multiply(powers, config_.extended_lp.polynomial_terms[term].powers);
    if (!product) continue;
    --product->powers[0];
    if (!row_is_allowed(product->powers)) continue;
    result.push_back({std::move(product->powers), true, term,
                      checked_mul(product->coefficient, -1), 0});
  }
  return result;
}

std::vector<Monomial> JetEquationGenerator::nabla(unsigned slot,
                                                  const std::vector<int>& powers,
                                                  bool allow_boundary_raise) const
{
  if (slot >= config_.integral_count)
    throw std::invalid_argument("jet derivative slot is out of range");
  std::vector<Monomial> result;
  const int seed_power = powers[slot + 1];
  if (seed_power > 0) {
    auto row = powers;
    --row[slot + 1];
    if (row_is_allowed(row))
      result.push_back({std::move(row), false, 0, seed_power, 0});
  } else if (seed_power == 0) {
    auto row = powers;
    row[slot + 1] = -1;
    if (row_is_allowed(row)) result.push_back({std::move(row), false, 0, 1, 0});
  } else if (allow_boundary_raise) {
    auto row = powers;
    if (row[slot + 1] == std::numeric_limits<int>::min())
      throw std::overflow_error("jet derivative order exceeds int range");
    --row[slot + 1];
    if (row_is_allowed(row)) result.push_back({std::move(row), false, 0, 1, 0});
  }

  for (unsigned term = 0; term < config_.extended_lp.polynomial_terms.size(); ++term) {
    const auto& polynomial = config_.extended_lp.polynomial_terms[term];
    const unsigned derivative = polynomial.powers[slot];
    if (derivative == 0) continue;
    auto differentiated = polynomial.powers;
    --differentiated[slot];
    auto product = multiply(powers, differentiated);
    if (!product) continue;
    --product->powers[0];
    if (!row_is_allowed(product->powers)) continue;
    const auto scale = checked_mul(product->coefficient, derivative);
    result.push_back(
        {std::move(product->powers), true, term, checked_mul(scale, powers[0]), scale});
  }
  return result;
}

std::vector<Monomial> JetEquationGenerator::euler(const std::vector<int>& powers) const
{
  std::vector<Monomial> result;
  std::int64_t degree = static_cast<std::int64_t>(config_.integral_count);
  for (std::size_t slot = 0; slot < config_.integral_count; ++slot)
    degree = checked_add(degree, powers[slot + 1]);
  if (degree != 0 && row_is_allowed(powers))
    result.push_back({powers, false, 0, degree, 0});

  for (unsigned term = 0; term < config_.extended_lp.polynomial_terms.size(); ++term) {
    const auto& polynomial = config_.extended_lp.polynomial_terms[term];
    const std::int64_t monomial_degree = std::reduce(
        polynomial.powers.begin(), polynomial.powers.end(), std::int64_t{0});
    if (monomial_degree == 0) continue;
    auto product = multiply(powers, polynomial.powers);
    if (!product) continue;
    --product->powers[0];
    if (!row_is_allowed(product->powers)) continue;
    const auto scale = checked_mul(product->coefficient, monomial_degree);
    result.push_back(
        {std::move(product->powers), true, term, checked_mul(scale, powers[0]), scale});
  }
  return result;
}

std::vector<Monomial>
JetEquationGenerator::build_column(const std::vector<int>& powers, AnsatzFamily family,
                                   unsigned derivative_slot,
                                   bool allow_boundary_raise) const
{
  if (powers.size() != config_.integral_count + 1)
    throw std::invalid_argument("jet ansatz powers have the wrong dimension");
  switch (family) {
  case AnsatzFamily::Nabla:
    return nabla(derivative_slot, powers, allow_boundary_raise);
  case AnsatzFamily::Euler:
    return euler(powers);
  case AnsatzFamily::DimensionShift:
    return dimension_shift(powers);
  }
  throw std::logic_error("unknown jet ansatz family");
}
