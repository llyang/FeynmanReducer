#include "NativeFixedBasisOracle.hpp"

#include "LpNormalization.hpp"
#include "reduction/BlackBoxFeynman.hpp"

#include <firefly/FFInt.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace basis {

NativeFixedBasisOracle::NativeFixedBasisOracle(Config config,
                                               bool compute_quotient_normalized)
    : config_(std::move(config)),
      compute_quotient_normalized_(compute_quotient_normalized)
{
  unit_rows_.resize(config_.targets.size());
  for (std::size_t target = 0; target < config_.targets.size(); ++target) {
    const auto basis = std::ranges::find(config_.basis, config_.targets[target]);
    if (basis == config_.basis.end()) continue;
    FieldVector coordinates(config_.basis.size(), 0);
    coordinates[static_cast<std::size_t>(basis - config_.basis.begin())] = 1;
    unit_rows_[target] = std::move(coordinates);
  }
}

NativeFixedBasisOracle::~NativeFixedBasisOracle() = default;

std::span<const std::uint32_t>
NativeFixedBasisOracle::reconstructed_outputs() const noexcept
{
  return black_box_->reconstructed_outputs();
}

std::size_t NativeFixedBasisOracle::total_output_count() const noexcept
{
  return black_box_->total_output_count();
}

std::unique_ptr<NativeFixedBasisOracle> NativeFixedBasisOracle::prepare(
    Config config, std::vector<Integral> basis, ReductionProgressCallback progress,
    ReplayOrientationPreference replay_orientation,
    std::span<const std::uint32_t> relation_source_sectors,
    bool compute_quotient_normalized, ReductionOptions options)
{
  if (basis.empty()) throw std::invalid_argument("fixed basis must not be empty");
  if (config.targets.empty())
    throw std::invalid_argument("native oracle target batch must not be empty");
  if (!config.dimension_parameter_index.has_value())
    throw std::invalid_argument("native oracle requires a free dimension parameter");
  if (config.kinematic_parameter_indices.size() != config.kinematic_parameters.size()) {
    throw std::invalid_argument("kinematic parameter metadata is inconsistent");
  }
  std::vector<bool> assigned_parameters(config.parameters.size(), false);
  const auto assign_parameter = [&](std::size_t index) {
    if (index >= assigned_parameters.size())
      throw std::invalid_argument("free parameter index is outside the value vector");
    if (assigned_parameters[index])
      throw std::invalid_argument("free parameter metadata contains duplicates");
    assigned_parameters[index] = true;
  };
  assign_parameter(*config.dimension_parameter_index);
  for (const auto index : config.kinematic_parameter_indices)
    assign_parameter(index);
  if (!std::ranges::all_of(assigned_parameters, [](bool value) { return value; })) {
    throw std::invalid_argument(
        "dimension and kinematic metadata do not cover every free parameter");
  }

  config.basis = std::move(basis);
  if (compute_quotient_normalized) {
    for (const auto& target : config.targets)
      (void)lp_normalization_data(config, target);
    for (const auto& master : config.basis)
      (void)lp_normalization_data(config, master);
  }

  auto result = std::unique_ptr<NativeFixedBasisOracle>(
      new NativeFixedBasisOracle(std::move(config), compute_quotient_normalized));
  options.replay_orientation = replay_orientation;
  result->black_box_ = BlackBoxFeynman::prepare(
      result->config_, relation_source_sectors, std::move(progress), options);
  std::vector<std::size_t> all_rows(result->config_.targets.size());
  std::iota(all_rows.begin(), all_rows.end(), 0);
  result->all_rows_ = result->prepare_rows(all_rows);
  return result;
}

void NativeFixedBasisOracle::set_prime(std::uint64_t prime)
{
  firefly::FFInt::set_new_prime(prime);
  black_box_->prime_changed();
  prime_ = prime;
}

std::optional<NativeOracleEvaluation>
NativeFixedBasisOracle::evaluate(std::uint64_t dimension,
                                 std::span<const std::uint64_t> kinematic_values)
{
  return evaluate_rows(dimension, kinematic_values, *all_rows_);
}

std::shared_ptr<const NativeFixedBasisOracle::PreparedRows>
NativeFixedBasisOracle::prepare_rows(std::span<const std::size_t> target_rows)
{
  std::vector<std::size_t> key(target_rows.begin(), target_rows.end());
  std::vector<bool> seen(config_.targets.size(), false);
  for (const auto row : key) {
    if (row >= config_.targets.size())
      throw std::out_of_range("native oracle target row is out of range");
    if (seen[row])
      throw std::invalid_argument("native oracle target rows must be unique");
    seen[row] = true;
  }
  std::vector<std::size_t> local_row(config_.targets.size(), config_.targets.size());
  for (std::size_t index = 0; index < key.size(); ++index)
    local_row[key[index]] = index;

  auto selection = std::shared_ptr<PreparedRows>(new PreparedRows());
  selection->owner = identity_;
  selection->target_rows = key;
  const auto positions = black_box_->reconstructed_outputs();
  selection->active_outputs.reserve(positions.size());
  selection->destinations.reserve(positions.size());
  for (std::size_t output = 0; output < positions.size(); ++output) {
    const auto target = positions[output] / config_.basis.size();
    if (target >= local_row.size())
      throw std::logic_error("native oracle output support has an invalid target row");
    if (local_row[target] == config_.targets.size()) continue;
    if (unit_rows_[target]) continue;
    selection->active_outputs.push_back(static_cast<std::uint32_t>(output));
    selection->destinations.emplace_back(local_row[target],
                                         positions[output] % config_.basis.size());
  }
  return selection;
}

std::optional<NativeOracleEvaluation>
NativeFixedBasisOracle::evaluate_rows(std::uint64_t dimension,
                                      std::span<const std::uint64_t> kinematic_values,
                                      std::span<const std::size_t> target_rows)
{
  auto prepared = last_rows_.load(std::memory_order_acquire);
  if (!prepared || !std::ranges::equal(prepared->target_rows, target_rows)) {
    prepared = prepare_rows(target_rows);
    last_rows_.store(prepared, std::memory_order_release);
  }
  return evaluate_rows(dimension, kinematic_values, *prepared);
}

std::optional<NativeOracleEvaluation>
NativeFixedBasisOracle::evaluate_rows(std::uint64_t dimension,
                                      std::span<const std::uint64_t> kinematic_values,
                                      const PreparedRows& prepared_rows)
{
  if (kinematic_values.size() != config_.kinematic_parameter_indices.size())
    throw std::invalid_argument("native oracle kinematic sample has wrong size");

  std::vector<firefly::FFInt> values(config_.parameters.size(), firefly::FFInt(0));
  const auto dimension_index = *config_.dimension_parameter_index;
  values[dimension_index] = firefly::FFInt(dimension);
  for (std::size_t index = 0; index < kinematic_values.size(); ++index) {
    const auto parameter = config_.kinematic_parameter_indices[index];
    values[parameter] = firefly::FFInt(kinematic_values[index]);
  }
  return evaluate_complete_values(values, prepared_rows);
}

std::optional<NativeOracleEvaluation>
NativeFixedBasisOracle::evaluate_rows(const std::vector<firefly::FFInt>& values,
                                      const PreparedRows& prepared_rows)
{
  return evaluate_complete_values(values, prepared_rows);
}

std::optional<NativeOracleEvaluation>
NativeFixedBasisOracle::evaluate_rows(std::span<const firefly::FFInt> values,
                                      const PreparedRows& prepared_rows)
{
  return evaluate_complete_values(
      std::vector<firefly::FFInt>(values.begin(), values.end()), prepared_rows);
}

std::optional<NativeOracleEvaluation> NativeFixedBasisOracle::evaluate_complete_values(
    const std::vector<firefly::FFInt>& values, const PreparedRows& prepared_rows)
{
  if (prepared_rows.owner.lock() != identity_)
    throw std::invalid_argument("prepared native-oracle rows belong to another oracle");
  if (!prime_.has_value())
    throw std::logic_error("native oracle prime has not been selected");
  if (firefly::FFInt::p != *prime_)
    throw std::logic_error("native oracle FireFly prime changed without notification");
  if (values.size() != config_.parameters.size())
    throw std::invalid_argument("native oracle complete sample has wrong size");

  std::vector<firefly::FFInt> compact;
  if (!prepared_rows.active_outputs.empty()) {
    compact = black_box_->eval_selected_compact(values, prepared_rows.active_outputs);
  }
  if (compact.empty() && !prepared_rows.active_outputs.empty()) return std::nullopt;
  if (compact.size() != prepared_rows.destinations.size())
    throw std::logic_error("native oracle selected output support mismatch");

  const PrimeField field(*prime_);
  NativeOracleEvaluation result;
  result.conventional.assign(prepared_rows.target_rows.size(),
                             FieldVector(config_.basis.size(), 0));
  for (std::size_t target = 0; target < prepared_rows.target_rows.size(); ++target) {
    if (unit_rows_[prepared_rows.target_rows[target]]) {
      result.conventional[target] = *unit_rows_[prepared_rows.target_rows[target]];
    }
  }
  if (compute_quotient_normalized_) result.quotient_normalized = result.conventional;
  result.selected_reconstructed_outputs = prepared_rows.active_outputs.size();
  result.replay_performed = !prepared_rows.active_outputs.empty();
  for (std::size_t index = 0; index < compact.size(); ++index) {
    const auto [target, basis] = prepared_rows.destinations[index];
    result.conventional[target][basis] = compact[index].n;
  }
  if (compute_quotient_normalized_) {
    const auto dimension = values[*config_.dimension_parameter_index].n;
    for (std::size_t target = 0; target < prepared_rows.target_rows.size(); ++target) {
      for (std::size_t basis = 0; basis < config_.basis.size(); ++basis) {
        const auto value = result.conventional[target][basis];
        result.quotient_normalized[target][basis] = field.multiply(
            value,
            lp_normalization_and_sign_ratio(
                field, config_, config_.targets[prepared_rows.target_rows[target]],
                config_.basis[basis], dimension));
      }
    }
  }
  return result;
}

} // namespace basis
