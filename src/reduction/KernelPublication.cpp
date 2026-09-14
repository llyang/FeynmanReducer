#include "reduction/BlackBoxFeynman.hpp"
#include "reduction/KernelErrors.hpp"
#include "reduction/KernelPlanning.hpp"
#include "reduction/ParameterEvaluation.hpp"

#include <algorithm>
#include <format>
#include <type_traits>
#include <utility>

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

void BlackBoxFeynman::publish_kernel_plan(
    reduction::detail::KernelPublicationInput&& input,
    const EvaluatedCoeffs<firefly::FFInt>& coeffs,
    const std::vector<firefly::FFInt>& values)
{
  square_dim = input.solution_columns.size();
  if (input.row_sectors.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("reference evaluator row count exceeds 32-bit ids");

  const auto reference_start = std::chrono::steady_clock::now();
  build_reference_evaluation_plan(input);
  const auto reference_end = std::chrono::steady_clock::now();
  build_pending_block_replay(input, coeffs, values);
  const auto recording_end = std::chrono::steady_clock::now();
  kernel_statistics_.reference_plan_seconds +=
      std::chrono::duration<double>(reference_end - reference_start).count();
  kernel_statistics_.block_recording_seconds +=
      std::chrono::duration<double>(recording_end - reference_end).count();

  if (recompact_source_) {
    auto& source = *recompact_source_;
    source.integrals = cfg.targets;
    if (input.checkpoint) {
      source.initial_basis = cfg.basis;
      for (const auto& value : values)
        input.checkpoint->parameters.push_back(value.n);
      source.checkpoint = std::move(input.checkpoint);
    }
    source.integral_columns = std::move(input.target_columns);
    source.target_sectors = std::move(input.target_sectors);
    source.ordered_groups = std::move(input.ordered_groups);
    source.dot_ordering =
        resolve_ansatz_dot_ordering(ansatz_dot_ordering, source.maximum_g_shift != 0,
                                    kernel_statistics_.ansatz_expansion_rounds);
    // Preserve alternative relations as well as the selected pivots. The search
    // replay above still uses its compact system; these raw columns are only
    // consumed after the final basis is known. Keep original row ids and every
    // closure term, and transfer ownership without copying the relation pool.
    source.ansatz_columns = std::move(input.ansatz_columns);
    source.ansatz_metadata = std::move(input.ansatz_metadata);
    source.row_sectors = std::move(input.row_sectors);
    source.row_groups = std::move(input.row_groups);
    const auto bytes = [](const auto& vector) {
      return vector.capacity() *
             sizeof(typename std::decay_t<decltype(vector)>::value_type);
    };
    source.retained_bytes =
        sizeof(source) + bytes(source.integrals) +
        bytes(source.integral_columns.terms) + bytes(source.integral_columns.offsets) +
        bytes(source.ansatz_columns.terms) + bytes(source.ansatz_columns.offsets) +
        bytes(source.ansatz_metadata) + bytes(source.row_sectors) +
        bytes(source.row_groups) + bytes(source.ordered_groups) +
        bytes(source.target_sectors) + bytes(source.expressions);
    if (source.checkpoint) source.retained_bytes += source.checkpoint->bytes();
    source.retained_bytes += bytes(source.initial_basis);
    for (const auto& integral : source.initial_basis)
      source.retained_bytes += bytes(integral.indices);
    for (const auto& integral : source.integrals)
      source.retained_bytes += bytes(integral.indices);
    for (const auto& expression : source.expressions) {
      source.retained_bytes += bytes(expression.atoms);
      for (const auto& atom : expression.atoms)
        source.retained_bytes += bytes(atom.polynomial_factors);
    }
  }
}

std::unique_ptr<reduction::detail::RecompactSource>
BlackBoxFeynman::take_recompact_source()
{
  if (!recompact_source_)
    throw std::logic_error("kernel has no retained recompaction source");
  return std::move(recompact_source_);
}

std::unique_ptr<BlackBoxFeynman> BlackBoxFeynman::prepare_from_source(
    const Config& config, std::unique_ptr<reduction::detail::RecompactSource> source,
    ReductionProgressCallback progress, ReductionOptions options)
{
  if (!source) throw std::invalid_argument("recompaction source is missing");
  if (options.numerator_strategy != NumeratorReductionStrategy::Projected)
    throw std::invalid_argument(
        "final-basis recompaction only supports projected kernels");
  if (progress)
    progress(
        std::format("Recompact source: rows={}, relations={}, integral_columns={}, "
                    "retained_bytes={}",
                    source->row_sectors.size(), source->ansatz_columns.size(),
                    source->integrals.size(), source->retained_bytes),
        ReductionProgressEvent::info);
  if (source->checkpoint && progress)
    progress(
        std::format("Provisional checkpoint: operations={}, rank={}, retained_bytes={}",
                    source->checkpoint->operations.size(),
                    source->checkpoint->solution_columns.size(),
                    source->checkpoint->bytes()),
        ReductionProgressEvent::info);
  return prepare_impl(config, progress, options, {}, source.get());
}

void BlackBoxFeynman::plan_from_source(const reduction::detail::RecompactSource& source,
                                       const EvaluatedCoeffs<firefly::FFInt>& coeffs,
                                       const std::vector<firefly::FFInt>& values,
                                       const ReductionProgressCallback& progress,
                                       bool reuse_checkpoint, bool reselect_support)
{
  reduction::detail::KernelPublicationInput input;
  const auto extract = [&](std::span<const Integral> integrals,
                           reduction::detail::IndexedColumns& columns,
                           std::vector<std::uint32_t>* sectors) {
    for (const auto& integral : integrals) {
      const auto found = std::ranges::find(source.integrals, integral);
      if (found == source.integrals.end())
        throw std::invalid_argument(
            "final integral is absent from recompaction source");
      const auto index = static_cast<std::size_t>(found - source.integrals.begin());
      const auto terms = source.integral_columns.column(index);
      columns.terms.insert(columns.terms.end(), terms.begin(), terms.end());
      columns.offsets.push_back(columns.terms.size());
      if (sectors) sectors->push_back(source.target_sectors.at(index));
    }
  };
  extract(cfg.basis, input.basis_columns, nullptr);
  extract(cfg.targets, input.target_columns, &input.target_sectors);
  input.ansatz_columns = source.ansatz_columns;
  input.ansatz_metadata = source.ansatz_metadata;
  input.row_sectors = source.row_sectors;
  input.row_groups = source.row_groups;
  input.ordered_groups = source.ordered_groups;
  const auto terms = reduction::detail::reduction_polynomial_terms(cfg);
  for (const auto& term : terms)
    input.polynomial_values.push_back(
        reduction::detail::evaluate_polynomial_coefficient(cfg, term, values));
  const auto* checkpoint = reuse_checkpoint ? source.checkpoint.get() : nullptr;
  if (checkpoint && source.initial_basis.size() != checkpoint->basis_columns)
    throw std::logic_error("checkpoint initial basis shape mismatch");
  if (checkpoint && !checkpoint->matches(firefly::FFInt::p, values))
    throw std::logic_error("checkpoint parameter point mismatch");
  const auto select = [&](const reduction::detail::ProvisionalCheckpoint* saved) {
    return reduction::detail::plan_compact_kernel(
        input.basis_columns, input.target_columns, input.ansatz_columns,
        input.ansatz_metadata, input.row_sectors, input.polynomial_values,
        coeffs.top_lp_coefficients, coeffs.minus_half_d, input.row_groups,
        input.ordered_groups, source.dot_ordering, cfg.threads,
        check_master_independence, false, saved);
  };
  auto selection = select(checkpoint);
  reduction::detail::LocalReselectionStatistics local;
  if (reselect_support && checkpoint && selection.closed) {
    local = reduction::detail::reselect_compact_support(
        selection, input, coeffs.top_lp_coefficients, coeffs.minus_half_d,
        source.dot_ordering, cfg.threads);
    if (progress)
      progress(std::format("Local relation reselection: attempted={}, envelope_rows={}, "
                           "eligible_relations={}, added_relations={}, selected_new_relations={}, "
                           "closed={}, initial_ms={:.2f}, scan_ms={:.2f}, reselect_ms={:.2f}",
                           local.attempted, local.envelope_rows, local.eligible_relations,
                           local.added_relations, local.selected_new_relations, selection.closed,
                           local.initial_ms, local.scan_ms, local.reselect_ms),
               ReductionProgressEvent::info);
  }
  if (checkpoint && !selection.closed) {
    if (progress)
      progress("Provisional reuse fallback: saved frame or changed basis did not close",
               ReductionProgressEvent::info);
    selection = select(nullptr);
  }
  if (selection.reused_provisional && progress)
    progress(selection.provisional_rhs_fallback
                 ? "Provisional reuse accepted: full physical RHS support"
                 : "Provisional reuse accepted: projected physical RHS support",
             ReductionProgressEvent::info);
  if (!selection.closed) throw AnsatzClosureError();
  kernel_statistics_.compact_policy = selection.reused_provisional
                                          ? (local.attempted ? "local-reselect-final-basis"
                                                             : "reuse-provisional-final-basis")
                                          : "all-relations-final-basis";
  publish_ansatz_statistics(selection, input.ansatz_metadata, input.row_sectors.size(),
                            input.basis_columns.size() + input.ansatz_columns.size(), 0,
                            0, 0);
  if (progress) {
    progress(std::format("Recompact selection: source_relations={}, live_relations={}, "
                         "compact_dimension={}, elapsed_ms={:.2f}",
                         input.ansatz_columns.size(), selection.ansatz_order.size(),
                         selection.solution_columns.size(), selection.timings.total_ms),
             ReductionProgressEvent::info);
    progress(
        std::format("Recompact phases: provisional_build_ms={:.2f}, "
                    "provisional_elimination_ms={:.2f}, support_selection_ms={:.2f}, "
                    "compact_build_ms={:.2f}, compact_elimination_ms={:.2f}",
                    selection.timings.provisional_build_ms,
                    selection.timings.provisional_elimination_ms,
                    selection.timings.support_selection_ms,
                    selection.timings.compact_build_ms,
                    selection.timings.compact_elimination_ms),
        ReductionProgressEvent::info);
  }
  input.ansatz_order = std::move(selection.ansatz_order);
  input.solution_columns = std::move(selection.solution_columns);
  input.elimination_row_map = std::move(selection.elimination_row_map);
  publish_kernel_plan(std::move(input), coeffs, values);
}
