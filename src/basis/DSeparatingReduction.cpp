#include "basis/DSeparatingReduction.hpp"

#include "basis/FiniteField.hpp"
#include "core/IntegralFormatting.hpp"
#include "reduction/DifferentialEquations.hpp"
#include "reduction/KernelPlanning.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <limits>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace basis {

std::unique_ptr<reduction::detail::RecompactSource>
DSeparatingReduction::take_recompact_source()
{
  return prepared_.oracle->take_recompact_source();
}

namespace {

std::size_t find_integral(std::span<const Integral> values, const Integral& integral)
{
  const auto found = std::ranges::find(values, integral);
  if (found == values.end())
    throw std::logic_error("D-separating replay row is absent from prepared batch");
  return static_cast<std::size_t>(found - values.begin());
}

void compact_differential_targets(Config& config,
                                  std::vector<std::uint32_t>& output_support,
                                  std::size_t basis_size)
{
  if (config.targets.empty())
    throw std::logic_error("differential target registry is empty");
  std::vector<bool> used(config.targets.size(), false);
  for (const auto& request : config.reduction_requests) {
    for (const auto& term : request.terms) {
      if (term.target >= used.size())
        throw std::logic_error("differential request target is out of range");
      used[term.target] = true;
    }
  }
  // An all-zero system still needs one native target to prepare the oracle.
  if (std::ranges::none_of(used, [](bool value) { return value; })) used[0] = true;

  const auto invalid = std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> remap(config.targets.size(), invalid);
  std::vector<Integral> compact;
  compact.reserve(config.targets.size());
  for (std::size_t old = 0; old < config.targets.size(); ++old) {
    if (!used[old]) continue;
    remap[old] = static_cast<std::uint32_t>(compact.size());
    compact.push_back(config.targets[old]);
  }
  for (auto& request : config.reduction_requests)
    for (auto& term : request.terms)
      term.target = remap.at(term.target);

  std::vector<std::uint32_t> compact_support;
  compact_support.reserve(output_support.size());
  for (const std::uint32_t position : output_support) {
    const std::size_t old_target = position / basis_size;
    const std::size_t basis = position % basis_size;
    if (old_target >= remap.size())
      throw std::logic_error("D-separating output support target is out of range");
    if (remap[old_target] == invalid) continue;
    const std::size_t next =
        static_cast<std::size_t>(remap[old_target]) * basis_size + basis;
    if (next > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("compacted D-separating output exceeds 32-bit ids");
    compact_support.push_back(static_cast<std::uint32_t>(next));
  }
  config.targets = std::move(compact);
  output_support = std::move(compact_support);
}

} // namespace

DSeparatingReduction::DSeparatingReduction(PreparedDSeparatingBasisSearch prepared,
                                           std::vector<Integral> original_targets,
                                           DSeparatingSharedOptimization optimization,
                                           const ReductionProgressCallback& progress)
    : prepared_(std::move(prepared)), original_targets_(std::move(original_targets)),
      final_output_support_(std::move(prepared_.final_output_support))
{
  if (!prepared_.oracle || prepared_.report.selected_basis.empty())
    throw std::invalid_argument("D-separating reduction is not prepared");
  const auto initial_basis = prepared_.oracle->basis();
  const auto& selected_basis = prepared_.report.selected_basis;
  auto slots = prepared_.selected_basis_to_swap_slot;
  std::ranges::sort(slots);
  if (slots.size() != selected_basis.size())
    throw std::logic_error("D-separating basis permutation has the wrong size");
  for (std::size_t index = 0; index < slots.size(); ++index)
    if (slots[index] != index)
      throw std::logic_error("D-separating basis permutation is not bijective");
  identity_basis_ = prepared_.report.swap_path.empty() &&
                    initial_basis.size() == selected_basis.size() &&
                    std::ranges::equal(initial_basis, selected_basis);

  for (const auto& swap : prepared_.report.swap_path) {
    swap_rows_.push_back(find_integral(prepared_.oracle->targets(), swap.inserted));
    swap_slots_.push_back(swap.slot);
  }
  for (const auto& target : original_targets_)
    target_rows_.push_back(find_integral(prepared_.oracle->targets(), target));

  std::vector<std::uint32_t> all_outputs(final_output_support_.size());
  std::iota(all_outputs.begin(), all_outputs.end(), std::uint32_t{0});
  const auto specialization_started =
      optimization == DSeparatingSharedOptimization::None
          ? std::chrono::steady_clock::time_point{}
          : std::chrono::steady_clock::now();
  if (optimization != DSeparatingSharedOptimization::None) {
    auto rows = target_rows_;
    rows.insert(rows.end(), swap_rows_.begin(), swap_rows_.end());
    std::ranges::sort(rows);
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    std::vector<std::uint32_t> inputs;
    for (const auto row : rows)
      for (std::size_t slot = 0; slot < selected_basis.size(); ++slot)
        inputs.push_back(
            static_cast<std::uint32_t>(row * selected_basis.size() + slot));
    prepared_.oracle->retain_components(inputs, progress);
    if (progress)
      progress(std::format("Shared dependency selection: components={}, "
                           "original_row_components={}, mode={}",
                           inputs.size(), rows.size() * selected_basis.size(), "rows"),
               ReductionProgressEvent::info);
  }
  full_selection_plan_ = build_selection_plan(all_outputs);
  if (optimization != DSeparatingSharedOptimization::None && progress)
    progress(std::format("Shared specialization: elapsed_ms={:.2f}",
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - specialization_started)
                             .count()),
             ReductionProgressEvent::info);
}

DSeparatingReduction::~DSeparatingReduction() = default;

std::unique_ptr<DSeparatingReduction> DSeparatingReduction::prepare(
    Config& output_config, std::vector<Integral> initial_basis,
    std::span<const std::uint32_t> relation_source_sectors,
    ReductionProgressCallback progress, ReductionOptions options)
{
  DSeparatingBasisSearchOptions search_options;
  search_options.strategy = options.d_separating_search;
  if (progress)
    progress(std::format("D-separating search strategy: {}",
                         static_cast<unsigned>(search_options.strategy)),
             ReductionProgressEvent::info);
  auto search_progress = [&](std::string_view message) {
    if (progress)
      progress(std::format("D-separating search: {}", message),
               ReductionProgressEvent::info);
  };

  auto search = prepare_d_separating_basis_search(
      output_config, initial_basis, relation_source_sectors, search_options,
      search_progress, options);
  if (search.report.status != DSeparatingBasisSearchStatus::Passed) {
    throw std::runtime_error(
        std::format("D-separating basis selection failed: {}", search.report.message));
  }
  const auto selection_report = search.report;
  auto selected_basis = search.report.selected_basis;
  search.oracle.reset();

  Config final_config = output_config;
  if (final_config.differential_equations) {
    const auto differential_sources =
        reduction::detail::prepare_differential_equation_sources(final_config,
                                                                 selected_basis);
    reduction::detail::append_differential_equation_requests(
        final_config, differential_sources, selected_basis);
  }
  auto validation_progress = [&](std::string_view message) {
    if (progress)
      progress(std::format("D-separating validation: {}", message),
               ReductionProgressEvent::info);
  };
  auto prepared = prepare_d_separating_fixed_basis_validation(
      final_config, selected_basis, relation_source_sectors, search_options,
      validation_progress, options);
  if (prepared.report.status != DSeparatingBasisSearchStatus::Passed) {
    std::string detail = prepared.report.message;
    if (const auto& witness = prepared.report.primary_witness) {
      detail += std::format(
          "; target={}, component={}, sector={}, mixed_degree={}",
          format_mathematica_integral(final_config.integral_header, witness->target),
          witness->component, witness->sector, witness->mixed_degree);
      if (witness->component < selected_basis.size())
        detail += std::format(", master={}", format_mathematica_integral(
                                                 final_config.integral_header,
                                                 selected_basis[witness->component]));
    }
    throw std::runtime_error(
        std::format("D-separating fixed-basis validation failed: {}", detail));
  }
  if (prepared.report.selected_basis != selected_basis)
    throw std::logic_error("fixed D-separating validation changed the selected basis");
  if (final_config.differential_equations)
    compact_differential_targets(final_config, prepared.final_output_support,
                                 selected_basis.size());
  output_config = std::move(final_config);
  auto original_targets = output_config.targets;
  if (progress) {
    progress(std::format(
                 "D-separating basis selected: masters={}, candidates={}, states={}, "
                 "swaps={}, probes={}, elapsed_ms={:.2f}",
                 selected_basis.size(), selection_report.candidate_pool_size,
                 selection_report.basis_states_tested,
                 selection_report.swap_path.size(),
                 selection_report.probe_statistics.selected_replay_calls,
                 selection_report.timing.total_seconds * 1000.0),
             ReductionProgressEvent::info);
    progress(std::format("D-separating fixed validation: targets={}, probes={}, "
                         "elapsed_ms={:.2f}",
                         prepared.report.validation_targets.size(),
                         prepared.report.probe_statistics.selected_replay_calls,
                         prepared.report.timing.total_seconds * 1000.0),
             ReductionProgressEvent::info);
    progress(std::format("D-separating validation cold phases: native_prepare_ms={:.2f}, "
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
                 "D-separating validation sampling: interpolation_attempts={}, "
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
  auto result = std::unique_ptr<DSeparatingReduction>(new DSeparatingReduction(
      std::move(prepared), std::move(original_targets),
      (options.d_separating_kernel == DSeparatingKernelStrategy::SharedOracle ||
       options.d_separating_kernel == DSeparatingKernelStrategy::Auto)
          ? options.d_separating_shared
          : DSeparatingSharedOptimization::None,
      progress));
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

  current.reset();
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
      result[output] =
          firefly::FFInt(rebased[prepared_.selected_basis_to_swap_slot
                                     [plan.output_basis_positions[output]]]);
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
