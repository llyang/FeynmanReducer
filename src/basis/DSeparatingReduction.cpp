#include "basis/DSeparatingReduction.hpp"

#include "basis/FiniteField.hpp"

#include <algorithm>
#include <format>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace basis {
namespace {

std::size_t find_integral(std::span<const Integral> values, const Integral& integral)
{
  const auto found = std::ranges::find(values, integral);
  if (found == values.end())
    throw std::logic_error("D-separating replay row is absent from prepared batch");
  return static_cast<std::size_t>(found - values.begin());
}

} // namespace

DSeparatingReduction::DSeparatingReduction(PreparedDSeparatingBasisSearch prepared,
                                           std::vector<Integral> original_targets)
    : prepared_(std::move(prepared)), original_targets_(std::move(original_targets)),
      final_output_support_(prepared_.final_output_support)
{
  if (!prepared_.oracle || prepared_.report.selected_basis.empty())
    throw std::invalid_argument("D-separating reduction is not prepared");
  const auto initial_basis = prepared_.oracle->basis();
  const auto& selected_basis = prepared_.report.selected_basis;
  identity_basis_ = initial_basis.size() == selected_basis.size() &&
                    std::ranges::equal(initial_basis, selected_basis);

  for (const auto& swap : prepared_.report.swap_path) {
    swap_rows_.push_back(find_integral(prepared_.oracle->targets(), swap.inserted));
    swap_slots_.push_back(swap.slot);
  }
  for (const auto& target : original_targets_)
    target_rows_.push_back(find_integral(prepared_.oracle->targets(), target));

  std::vector<std::uint32_t> all_outputs(final_output_support_.size());
  std::iota(all_outputs.begin(), all_outputs.end(), std::uint32_t{0});
  full_selection_plan_ = build_selection_plan(all_outputs);
}

DSeparatingReduction::~DSeparatingReduction() = default;

std::unique_ptr<DSeparatingReduction> DSeparatingReduction::prepare(
    Config& output_config, std::vector<Integral> initial_basis,
    std::span<const std::uint32_t> relation_source_sectors,
    ReductionProgressCallback progress, ReductionOptions options)
{
  const auto original_targets = output_config.targets;
  DSeparatingBasisSearchOptions search_options;
  auto search_progress = [&](std::string_view message) {
    if (progress)
      progress(std::format("D-separating basis: {}", message),
               ReductionProgressEvent::info);
  };
  auto prepared = prepare_d_separating_basis_search(
      output_config, std::move(initial_basis), relation_source_sectors, search_options,
      search_progress, options);
  if (prepared.report.status != DSeparatingBasisSearchStatus::Passed) {
    throw std::runtime_error(std::format("D-separating basis selection failed: {}",
                                         prepared.report.message));
  }
  auto selected_basis = prepared.report.selected_basis;
  if (progress) {
    progress(std::format(
                 "D-separating basis selected: masters={}, candidates={}, states={}, "
                 "swaps={}, probes={}, elapsed_ms={:.2f}",
                 selected_basis.size(), prepared.report.candidate_pool_size,
                 prepared.report.basis_states_tested, prepared.report.swap_path.size(),
                 prepared.report.probe_statistics.selected_replay_calls,
                 prepared.report.timing.total_seconds * 1000.0),
             ReductionProgressEvent::info);
    progress(std::format("D-separating cold phases: native_prepare_ms={:.2f}, "
                         "rank_ms={:.2f}, screening_ms={:.2f}, pivot_ms={:.2f}, "
                         "strict_ms={:.2f}, probe_ms={:.2f}, rebase_ms={:.2f}, "
                         "interpolation_ms={:.2f}, signature_ms={:.2f}",
                         prepared.report.timing.native_prepare_seconds * 1000.0,
                         prepared.report.timing.rank_filter_seconds * 1000.0,
                         prepared.report.timing.screening_seconds * 1000.0,
                         prepared.report.timing.pivot_scoring_seconds * 1000.0,
                         prepared.report.timing.strict_validation_seconds * 1000.0,
                         prepared.report.timing.probe_evaluation_seconds * 1000.0,
                         prepared.report.timing.pointwise_rebase_seconds * 1000.0,
                         prepared.report.timing.rational_interpolation_seconds * 1000.0,
                         prepared.report.timing.signature_comparison_seconds * 1000.0),
             ReductionProgressEvent::info);
    progress(std::format(
                 "D-separating adaptive validation: interpolation_attempts={}, "
                 "model_cache_hits={}, fixed_degree_attempts={}, thiele_attempts={}, "
                 "exhaustive_fallbacks={}, avoided_basis_factorizations={}",
                 prepared.report.probe_statistics.rational_interpolation_attempts,
                 prepared.report.probe_statistics.rational_interpolation_cache_hits,
                 prepared.report.probe_statistics.fixed_degree_attempts,
                 prepared.report.probe_statistics.thiele_attempts,
                 prepared.report.probe_statistics.exhaustive_fallbacks,
                 prepared.report.probe_statistics.avoided_basis_factorizations),
             ReductionProgressEvent::info);
  }
  auto result = std::unique_ptr<DSeparatingReduction>(
      new DSeparatingReduction(std::move(prepared), original_targets));
  output_config.basis.swap(selected_basis);
  return result;
}

void DSeparatingReduction::prime_changed()
{
  prepared_.oracle->set_prime(firefly::FFInt::p);
}

std::shared_ptr<const DSeparatingReduction::OutputSelectionPlan>
DSeparatingReduction::build_selection_plan(
    std::span<const std::uint32_t> active_outputs)
{
  const auto basis_size = prepared_.report.selected_basis.size();
  auto plan = std::make_shared<OutputSelectionPlan>();
  plan->active_outputs.assign(active_outputs.begin(), active_outputs.end());
  plan->needed_targets.reserve(active_outputs.size());
  plan->output_basis_positions.reserve(active_outputs.size());
  for (const auto output : active_outputs) {
    if (output >= final_output_support_.size())
      throw std::out_of_range("active D-separating output is out of range");
    const auto position = final_output_support_[output];
    const auto target = position / basis_size;
    if (target >= target_rows_.size())
      throw std::out_of_range("D-separating output position is out of range");
    plan->needed_targets.push_back(target);
    plan->output_basis_positions.push_back(position % basis_size);
  }
  std::ranges::sort(plan->needed_targets);
  plan->needed_targets.erase(
      std::unique(plan->needed_targets.begin(), plan->needed_targets.end()),
      plan->needed_targets.end());

  std::vector<std::size_t> oracle_rows = swap_rows_;
  oracle_rows.reserve(swap_rows_.size() + plan->needed_targets.size());
  for (const auto target : plan->needed_targets)
    oracle_rows.push_back(target_rows_[target]);
  std::ranges::sort(oracle_rows);
  oracle_rows.erase(std::unique(oracle_rows.begin(), oracle_rows.end()),
                    oracle_rows.end());
  plan->oracle_rows = prepared_.oracle->prepare_rows(oracle_rows);

  const auto probe_position = [&](std::size_t row) {
    const auto found = std::ranges::lower_bound(oracle_rows, row);
    if (found == oracle_rows.end() || *found != row)
      throw std::logic_error("D-separating planned oracle row is absent");
    return static_cast<std::size_t>(found - oracle_rows.begin());
  };
  plan->swap_probe_positions.reserve(swap_rows_.size());
  for (const auto row : swap_rows_)
    plan->swap_probe_positions.push_back(probe_position(row));
  plan->target_probe_positions.reserve(plan->needed_targets.size());
  for (const auto target : plan->needed_targets)
    plan->target_probe_positions.push_back(probe_position(target_rows_[target]));

  std::vector<std::size_t> target_to_local(target_rows_.size(),
                                           std::numeric_limits<std::size_t>::max());
  for (std::size_t local = 0; local < plan->needed_targets.size(); ++local)
    target_to_local[plan->needed_targets[local]] = local;
  plan->output_indices_by_target.resize(plan->needed_targets.size());
  for (std::size_t output_index = 0; output_index < active_outputs.size();
       ++output_index) {
    const auto output = active_outputs[output_index];
    const auto target = final_output_support_[output] / basis_size;
    const auto local = target_to_local[target];
    if (local == std::numeric_limits<std::size_t>::max())
      throw std::logic_error("D-separating output target was not planned");
    plan->output_indices_by_target[local].push_back(output_index);
  }

  return plan;
}

std::shared_ptr<const DSeparatingReduction::OutputSelectionPlan>
DSeparatingReduction::selected_plan(std::span<const std::uint32_t> active_outputs)
{
  if (active_outputs.size() == final_output_support_.size() &&
      std::ranges::equal(active_outputs, full_selection_plan_->active_outputs)) {
    return full_selection_plan_;
  }

  auto current = selected_selection_plan_.load(std::memory_order_acquire);
  if (current && std::ranges::equal(current->active_outputs, active_outputs))
    return current;

  std::lock_guard lock(selection_plan_mutex_);
  current = selected_selection_plan_.load(std::memory_order_relaxed);
  if (current && std::ranges::equal(current->active_outputs, active_outputs))
    return current;

  auto plan = build_selection_plan(active_outputs);
  selected_selection_plan_.store(plan, std::memory_order_release);
  return plan;
}

std::vector<firefly::FFInt>
DSeparatingReduction::evaluate_plan(const std::vector<firefly::FFInt>& values,
                                    const OutputSelectionPlan& plan)
{
  const auto& config = prepared_.oracle->config();
  if (!config.dimension_parameter_index || values.size() != config.parameters.size())
    throw std::invalid_argument("D-separating replay parameter shape mismatch");
  if (plan.active_outputs.empty()) return {};
  const auto probe = prepared_.oracle->evaluate_rows(values, *plan.oracle_rows);
  if (!probe) throw std::runtime_error("D-separating shared replay failed at a probe");

  std::vector<firefly::FFInt> result(plan.active_outputs.size(), firefly::FFInt(0));
  if (identity_basis_) {
    for (std::size_t target = 0; target < plan.needed_targets.size(); ++target) {
      const auto& coordinates =
          probe->conventional[plan.target_probe_positions[target]];
      for (const auto output : plan.output_indices_by_target[target])
        result[output] =
            firefly::FFInt(coordinates[plan.output_basis_positions[output]]);
    }
    return result;
  }

  FieldMatrix inserted;
  inserted.reserve(plan.swap_probe_positions.size());
  for (const auto position : plan.swap_probe_positions)
    inserted.push_back(probe->conventional[position]);
  const PrimeField field(firefly::FFInt::p);
  const auto tape = build_basis_swap_coordinate_tape(field, inserted, swap_slots_);
  if (!tape) throw std::runtime_error("D-separating swap pivot vanished at a probe");
  for (std::size_t target = 0; target < plan.needed_targets.size(); ++target) {
    const auto rebased = rebase_after_basis_swaps(
        field, probe->conventional[plan.target_probe_positions[target]], *tape);
    for (const auto output : plan.output_indices_by_target[target])
      result[output] = firefly::FFInt(rebased[plan.output_basis_positions[output]]);
  }
  return result;
}

std::vector<firefly::FFInt> DSeparatingReduction::eval_selected_compact(
    const std::vector<firefly::FFInt>& values,
    const std::vector<std::uint32_t>& active_outputs)
{
  if (!std::ranges::is_sorted(active_outputs) ||
      std::ranges::adjacent_find(active_outputs) != active_outputs.end()) {
    throw std::invalid_argument(
        "active D-separating outputs must be sorted and unique");
  }
  return evaluate_plan(values, *selected_plan(active_outputs));
}

std::vector<firefly::FFInt>
DSeparatingReduction::eval_selected(const std::vector<firefly::FFInt>& values,
                                    const std::vector<std::uint32_t>& active_outputs)
{
  const auto compact = eval_selected_compact(values, active_outputs);
  std::vector<firefly::FFInt> result(final_output_support_.size(), firefly::FFInt(0));
  for (std::size_t index = 0; index < active_outputs.size(); ++index)
    result[active_outputs[index]] = compact[index];
  return result;
}

} // namespace basis
