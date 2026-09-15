#include "NativeFixedBasisOracle.hpp"

#include "core/ProbeValues.hpp"
#include "reduction/BlackBoxFeynman.hpp"
#include "reduction/ParameterEvaluation.hpp"

#include <firefly/FFInt.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace basis {

std::unique_ptr<reduction::detail::RecompactSource>
NativeFixedBasisOracle::take_recompact_source()
{
  return black_box_->take_recompact_source();
}

NativeFixedBasisOracle::NativeFixedBasisOracle(Config config)
    : config_(std::move(config))
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
    std::span<const std::uint32_t> relation_source_sectors, ReductionOptions options)
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

  auto result = std::unique_ptr<NativeFixedBasisOracle>(
      new NativeFixedBasisOracle(std::move(config)));
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
  if (!all_rows_)
    throw std::logic_error("full oracle evaluation is unavailable after retention");
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
    if (retained_components_) {
      for (std::size_t basis = 0; basis < config_.basis.size(); ++basis)
        if (!std::ranges::binary_search(
                *retained_components_,
                static_cast<std::uint32_t>(row * config_.basis.size() + basis)))
          throw std::out_of_range("oracle row contains a discarded component");
    }
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

  NativeOracleEvaluation result;
  result.conventional.assign(prepared_rows.target_rows.size(),
                             FieldVector(config_.basis.size(), 0));
  for (std::size_t target = 0; target < prepared_rows.target_rows.size(); ++target) {
    if (unit_rows_[prepared_rows.target_rows[target]]) {
      result.conventional[target] = *unit_rows_[prepared_rows.target_rows[target]];
    }
  }
  result.selected_reconstructed_outputs = prepared_rows.active_outputs.size();
  result.replay_performed = !prepared_rows.active_outputs.empty();
  for (std::size_t index = 0; index < compact.size(); ++index) {
    const auto [target, basis] = prepared_rows.destinations[index];
    result.conventional[target][basis] = compact[index].n;
  }
  return result;
}

std::shared_ptr<const NativeFixedBasisOracle::PreparedComponents>
NativeFixedBasisOracle::prepare_components(std::span<const std::uint32_t> positions)
{
  auto prepared = std::make_shared<PreparedComponents>();
  prepared->owner = identity_;
  prepared->constants.resize(positions.size(), 0);
  std::vector<std::pair<std::uint32_t, std::size_t>> outputs;
  auto unique = std::vector<std::uint32_t>(positions.begin(), positions.end());
  std::ranges::sort(unique);
  if (std::ranges::adjacent_find(unique) != unique.end())
    throw std::invalid_argument("oracle component positions must be unique");
  const auto support = black_box_->reconstructed_outputs();
  for (std::size_t index = 0; index < positions.size(); ++index) {
    const auto flat = positions[index];
    if (flat >= total_output_count() ||
        (retained_components_ &&
         !std::ranges::binary_search(*retained_components_, flat)))
      throw std::out_of_range("oracle component is outside the retained domain");
    const auto row = flat / config_.basis.size();
    if (unit_rows_[row]) {
      prepared->constants[index] = (*unit_rows_[row])[flat % config_.basis.size()];
      continue;
    }
    const auto found = std::ranges::lower_bound(support, flat);
    if (found != support.end() && *found == flat)
      outputs.emplace_back(static_cast<std::uint32_t>(found - support.begin()), index);
  }
  std::ranges::sort(outputs);
  for (const auto [output, destination] : outputs) {
    prepared->active_outputs.push_back(output);
    prepared->destinations.push_back(destination);
  }
  return prepared;
}

std::optional<FieldVector>
NativeFixedBasisOracle::evaluate_components(const std::vector<firefly::FFInt>& values,
                                            const PreparedComponents& prepared)
{
  if (prepared.owner.lock() != identity_)
    throw std::invalid_argument(
        "prepared components belong to another oracle generation");
  if (!prime_ || firefly::FFInt::p != *prime_)
    throw std::logic_error("native oracle prime changed without notification");
  if (values.size() != config_.parameters.size())
    throw std::invalid_argument("native oracle complete sample has wrong size");
  auto result = prepared.constants;
  if (prepared.active_outputs.empty()) return result;
  const auto evaluated =
      black_box_->eval_selected_compact(values, prepared.active_outputs);
  if (evaluated.empty()) return std::nullopt;
  if (evaluated.size() != prepared.destinations.size())
    throw std::logic_error("native oracle component result shape mismatch");
  for (std::size_t index = 0; index < evaluated.size(); ++index)
    result[prepared.destinations[index]] = evaluated[index].n;
  return result;
}

bool NativeFixedBasisOracle::retain_components(std::span<const std::uint32_t> positions,
                                               ReductionProgressCallback progress)
{
  if (black_box_->kernel_statistics().replay_orientation == "master") {
    if (progress)
      progress("Shared kernel trim: skipped master orientation",
               ReductionProgressEvent::info);
    return false;
  }
  const auto previous_prime = firefly::FFInt::p;
  const auto old_plan = prepare_components(positions);
  struct Sample {
    std::uint64_t prime;
    std::vector<std::uint64_t> values;
    FieldVector expected;
  };
  std::vector<Sample> samples;
  const auto primes = reduction::detail::usable_firefly_primes(config_, 2);
  if (primes.size() != 2)
    throw std::runtime_error("shared trim requires two validation primes");
  try {
    for (const auto prime : primes) {
      set_prime(prime);
      std::size_t valid = 0;
      for (std::size_t attempt = 0; attempt < 16 && valid < 2; ++attempt) {
        std::vector<firefly::FFInt> values;
        for (std::size_t parameter = 0; parameter < config_.parameters.size();
             ++parameter)
          values.emplace_back(
              probe_values::field_value(prime, parameter, attempt + 19));
        auto expected = evaluate_components(values, *old_plan);
        if (!expected) continue;
        Sample sample{prime, {}, std::move(*expected)};
        for (const auto& value : values)
          sample.values.push_back(value.n);
        samples.push_back(std::move(sample));
        ++valid;
      }
      if (valid != 2)
        throw std::runtime_error("shared trim validation sampling failed");
    }
    set_prime(previous_prime);
    if (!black_box_->retain_outputs(old_plan->active_outputs, progress)) return false;
    retained_components_ =
        std::vector<std::uint32_t>(positions.begin(), positions.end());
    std::ranges::sort(*retained_components_);
    identity_ = std::make_shared<const int>(0);
    all_rows_.reset();
    last_rows_.store(nullptr, std::memory_order_release);
    const auto new_plan = prepare_components(positions);
    for (const auto& sample : samples) {
      set_prime(sample.prime);
      std::vector<firefly::FFInt> values;
      for (const auto value : sample.values)
        values.emplace_back(value);
      const auto actual = evaluate_components(values, *new_plan);
      if (!actual || *actual != sample.expected)
        throw std::runtime_error("shared trim changed a validated coefficient");
    }
    set_prime(previous_prime);
  } catch (...) {
    set_prime(previous_prime);
    throw;
  }
  if (progress)
    progress("Shared kernel trim validation: primes=2, points_per_prime=2, passed",
             ReductionProgressEvent::info);
  return true;
}

} // namespace basis
