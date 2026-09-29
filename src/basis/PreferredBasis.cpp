#include "PreferredBasis.hpp"

#include "FiniteField.hpp"
#include "NativeFixedBasisOracle.hpp"
#include "core/IntegralFormatting.hpp"
#include "core/ProbeValues.hpp"
#include "masters/detail/MasterIntegralOrdering.hpp"
#include "reduction/ParameterEvaluation.hpp"
#include "topology/SectorUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <map>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace basis {
namespace {

std::uint32_t canonical_sector(const Config& config, std::uint32_t sector)
{
  if (!config.symmetry)
    throw std::invalid_argument("preferred selection needs symmetry");
  for (const auto& group : config.symmetry->sector_classes) {
    if (group.representative == sector) return sector;
    for (const auto& relation : group.relations)
      if (relation.target_sector == sector) return group.representative;
  }
  return sector;
}

struct Completion {
  std::vector<Integral> basis;
  std::vector<Integral> supplements;
};

Completion complete_at_probe(const Config& config, const FieldMatrix& preferred_rows,
                             const PrimeField& field)
{
  const auto size = config.basis.size();
  if (preferred_rows.size() != config.preferred_masters.size())
    throw std::logic_error("preferred coordinate row count is inconsistent");
  SectorUtils sectors(config);
  std::map<std::uint32_t, std::vector<std::size_t>> baseline_groups;
  std::vector<std::uint32_t> group_order;
  for (std::size_t index = 0; index < size; ++index) {
    const auto raw = masters::detail::integral_sector(config, config.basis[index]);
    const auto sector = canonical_sector(config, raw);
    auto [entry, inserted] = baseline_groups.try_emplace(sector);
    if (inserted) group_order.push_back(sector);
    entry->second.push_back(index);
  }

  std::map<std::uint32_t, std::vector<std::size_t>> preferred_groups;
  for (std::size_t index = 0; index < config.preferred_masters.size(); ++index) {
    const auto& preferred = config.preferred_masters[index];
    const auto actual = masters::detail::integral_sector(config, preferred.integral);
    if (!sectors.is_valid_sector(actual))
      throw std::runtime_error(std::format(
          "preferred master is in a zero sector: {}",
          format_mathematica_integral(config.integral_header, preferred.integral)));
    const auto raw_host = preferred.host_sector.value_or(actual);
    if (raw_host == 0 || (raw_host & ~actual) != 0 ||
        !sectors.is_valid_sector(raw_host))
      throw std::runtime_error(std::format(
          "preferred master has invalid host sector {}: {}", raw_host,
          format_mathematica_integral(config.integral_header, preferred.integral)));
    const auto host = canonical_sector(config, raw_host);
    if (!baseline_groups.contains(host))
      throw std::runtime_error(std::format(
          "preferred host sector {} has no FR master slots: {}", raw_host,
          format_mathematica_integral(config.integral_header, preferred.integral)));
    if (preferred_rows[index].size() != size)
      throw std::logic_error("preferred coordinate column count is inconsistent");
    preferred_groups[host].push_back(index);
  }

  Completion result;
  FieldMatrix complete_rows;
  result.basis.reserve(size);
  complete_rows.reserve(size);
  for (const auto host : group_order) {
    const auto& slots = baseline_groups.at(host);
    FieldMatrix leading_rows;
    if (const auto found = preferred_groups.find(host);
        found != preferred_groups.end()) {
      if (found->second.size() > slots.size())
        throw std::runtime_error(std::format(
            "preferred host sector {} has {} integrals but only {} master slots", host,
            found->second.size(), slots.size()));
      for (const auto index : found->second) {
        FieldVector leading;
        leading.reserve(slots.size());
        for (const auto slot : slots)
          leading.push_back(preferred_rows[index][slot]);
        leading_rows.push_back(std::move(leading));
        if (matrix_rank(field, leading_rows) != leading_rows.size())
          throw std::runtime_error(std::format(
              "preferred host sector {} has dependent leading rows at {}", host,
              format_mathematica_integral(config.integral_header,
                                          config.preferred_masters[index].integral)));
        result.basis.push_back(config.preferred_masters[index].integral);
        complete_rows.push_back(preferred_rows[index]);
      }
    }
    for (const auto slot : slots) {
      if (leading_rows.size() == slots.size()) break;
      FieldVector leading(slots.size(), 0);
      const auto position =
          static_cast<std::size_t>(std::ranges::find(slots, slot) - slots.begin());
      leading[position] = 1;
      auto trial = leading_rows;
      trial.push_back(leading);
      if (matrix_rank(field, trial) == trial.size()) {
        leading_rows.push_back(std::move(leading));
        FieldVector full(size, 0);
        full[slot] = 1;
        complete_rows.push_back(std::move(full));
        result.basis.push_back(config.basis[slot]);
        result.supplements.push_back(config.basis[slot]);
      }
    }
    if (leading_rows.size() != slots.size())
      throw std::runtime_error(
          std::format("preferred host sector {} could not be completed", host));
  }
  if (result.basis.size() != size || matrix_rank(field, complete_rows) != size)
    throw std::runtime_error("completed preferred basis has deficient global rank");
  return result;
}

} // namespace

std::vector<Integral> complete_preferred_basis(const Config& config,
                                               ReductionProgressCallback progress)
{
  if (config.basis.empty() || config.preferred_masters.empty())
    throw std::invalid_argument("preferred selection needs a basis and preferences");
  if (!config.dimension_parameter_index)
    throw std::invalid_argument("preferred selection requires free d");
  if (config.preferred_masters.size() > config.basis.size())
    throw std::invalid_argument("preferred masters exceed physical basis dimension");
  Config scratch = config;
  scratch.targets.clear();
  scratch.reduction_requests.clear();
  scratch.differential_equations = false;
  scratch.basis_selection = BasisSelectionPolicy::Default;
  for (const auto& preferred : config.preferred_masters)
    scratch.targets.push_back(preferred.integral);
  auto oracle =
      NativeFixedBasisOracle::prepare(std::move(scratch), config.basis, progress);
  const auto primes = reduction::detail::usable_firefly_primes(config, 2);
  std::optional<Completion> accepted;
  for (const auto prime : primes) {
    oracle->set_prime(prime);
    std::optional<Completion> at_prime;
    std::size_t accepted_points = 0;
    std::string last_error;
    for (std::size_t point = 0; point < 8 && accepted_points < 2; ++point) {
      std::vector<std::uint64_t> kinematics(config.kinematic_parameters.size());
      for (std::size_t index = 0; index < kinematics.size(); ++index)
        kinematics[index] = probe_values::field_value(prime, index, point);
      const auto dimension = probe_values::field_value(
          prime, config.kinematic_parameters.size() + 11, point);
      const auto values = oracle->evaluate(dimension, kinematics);
      if (!values) continue;
      try {
        auto completed =
            complete_at_probe(config, values->conventional, PrimeField(prime));
        if (at_prime && at_prime->basis != completed.basis)
          throw std::runtime_error(
              "preferred supplement selection changes between probe points");
        at_prime = std::move(completed);
        ++accepted_points;
      } catch (const std::runtime_error& error) {
        last_error = error.what();
      }
    }
    if (accepted_points < 2)
      throw std::runtime_error(
          "preferred selection failed at prime " + std::to_string(prime) + ": " +
          (last_error.empty() ? "no nonsingular probe points" : last_error));
    if (accepted && accepted->basis != at_prime->basis)
      throw std::runtime_error(
          "preferred supplement selection disagrees across finite-field primes");
    accepted = std::move(at_prime);
  }
  if (!accepted) throw std::logic_error("preferred selection found no usable prime");
  if (progress)
    progress(std::format("Preferred basis: required={}, supplemented={}, total={}",
                         config.preferred_masters.size(), accepted->supplements.size(),
                         accepted->basis.size()),
             ReductionProgressEvent::info);
  return accepted->basis;
}

} // namespace basis
