#include "reduction/KernelPlanning.hpp"

#include "reduction/EliminationTape.hpp"
#include "reduction/FiniteFieldArithmetic.hpp"
#include "reduction/MasterIndependence.hpp"
#include "reduction/SparseMarkowitz.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

namespace reduction::detail {

namespace {

using NativeFieldElement = finite_field::MontgomeryFieldElement;

[[nodiscard]] std::vector<NativeFieldElement>
encode_field_values(std::span<const firefly::FFInt> source)
{
  std::vector<NativeFieldElement> result;
  result.reserve(source.size());
  for (const auto& value : source)
    result.push_back(NativeFieldElement::from_residue(value.n));
  return result;
}

template <typename T> struct ProbeSystem {
  linalg::SparseMatrix<T> matrix;
  std::vector<T> rhs;
  std::size_t rhs_columns = 0;
};

template <typename T> class ProbeSystemAssembler {
public:
  ProbeSystemAssembler(std::span<const T> polynomial_values,
                       std::span<const T> top_lp_coefficients, const T& minus_half_d)
      : polynomial_values_(polynomial_values),
        top_lp_coefficients_(top_lp_coefficients), minus_half_d_(minus_half_d)
  {}

  [[nodiscard]] T evaluate(const IndexedTerm& term) const
  {
    if (term.coefficient_expression != std::numeric_limits<std::uint32_t>::max()) {
      if (term.coefficient_expression >= top_lp_coefficients_.size())
        throw std::logic_error("top-LP coefficient expression is out of range");
      return T(term.coeff_int) * top_lp_coefficients_[term.coefficient_expression];
    }
    T value = T(term.coeff_int) + T(term.coeff_minus_half_d) * minus_half_d_;
    if (term.with_polynomial_coefficient) {
      if (term.polynomial_term_index >= polynomial_values_.size())
        throw std::logic_error("polynomial coefficient is out of range");
      value = value * polynomial_values_[term.polynomial_term_index];
    }
    return value;
  }

  void add_matrix_column(linalg::SparseMatrix<T>& matrix, const IndexedColumns& columns,
                         std::size_t source, std::size_t destination) const
  {
    for (const auto& term : columns.column(source)) {
      const T value = evaluate(term);
      if (value == T(0)) continue;
      auto& row = matrix[term.row];
      if (!row.empty() && row.back().column == destination) {
        row.back().value = row.back().value + value;
        if (row.back().value == T(0)) row.pop_back();
      } else {
        row.push_back({destination, value});
      }
    }
  }

  void add_rhs_column(std::vector<T>& rhs, std::size_t rhs_columns,
                      const IndexedColumns& columns, std::size_t source,
                      std::size_t destination) const
  {
    for (const auto& term : columns.column(source)) {
      auto& value = rhs[static_cast<std::size_t>(term.row) * rhs_columns + destination];
      value = value + evaluate(term);
    }
  }

private:
  std::span<const T> polynomial_values_;
  std::span<const T> top_lp_coefficients_;
  const T& minus_half_d_;
};

void assign_markowitz_statistics(CompactSelection& selection,
                                 const linalg::SparseMarkowitzStatistics& statistics)
{
  selection.timings.provisional_relation_elimination_ms =
      statistics.relation_elimination_ms;
  selection.timings.provisional_score_refresh_ms = statistics.score_refresh_ms;
  selection.timings.provisional_row_elimination_ms = statistics.row_elimination_ms;
  selection.provisional_relation_pivots = statistics.relation_pivots;
  selection.provisional_score_refresh_columns = statistics.score_refresh_columns;
  selection.provisional_incidence_records_scanned =
      statistics.incidence_records_scanned;
  selection.provisional_parallel_refresh_batches = statistics.parallel_refresh_batches;
  selection.provisional_parallel_refresh_columns = statistics.parallel_refresh_columns;
  selection.provisional_stale_choice_pops = statistics.stale_choice_pops;
  selection.provisional_choice_queue_compactions = statistics.choice_queue_compactions;
  selection.provisional_compacted_choice_entries = statistics.compacted_choice_entries;
  selection.provisional_maximum_choice_queue_size =
      statistics.maximum_choice_queue_size;
  selection.provisional_row_eliminations = statistics.row_eliminations;
  selection.provisional_parallel_row_batches = statistics.parallel_row_batches;
  selection.provisional_parallel_row_eliminations =
      statistics.parallel_row_eliminations;
}

linalg::ColumnComplexityOrder complexity_order(AnsatzDotOrdering ordering)
{
  switch (ordering) {
  case AnsatzDotOrdering::Auto:
    throw std::logic_error("automatic ansatz dot ordering was not resolved");
  case AnsatzDotOrdering::Markowitz:
    return linalg::ColumnComplexityOrder::Ignore;
  case AnsatzDotOrdering::LowFirst:
    return linalg::ColumnComplexityOrder::Ascending;
  case AnsatzDotOrdering::HighFirst:
    return linalg::ColumnComplexityOrder::Descending;
  }
  throw std::logic_error("unknown ansatz dot ordering");
}

template <typename T>
std::vector<std::size_t>
collect_residual_rows(const linalg::EliminationResult& elimination,
                      std::span<const T> rhs, std::size_t rhs_columns)
{
  if (elimination.closed) return {};
  if (rhs_columns == 0 || rhs.size() % rhs_columns != 0)
    throw std::logic_error("compact residual RHS shape is inconsistent");
  const std::size_t row_count = rhs.size() / rhs_columns;
  if (elimination.row_map.size() != row_count ||
      elimination.solution_cols.size() > elimination.row_map.size()) {
    throw std::logic_error("compact residual row map is inconsistent");
  }

  std::vector<std::size_t> residual;
  for (std::size_t position = elimination.solution_cols.size();
       position < elimination.row_map.size(); ++position) {
    const std::size_t row = elimination.row_map[position];
    for (std::size_t column = 0; column < rhs_columns; ++column) {
      if (rhs[row * rhs_columns + column] != T(0)) {
        residual.push_back(row);
        break;
      }
    }
  }
  return residual;
}

// Express B_new and T in [B_old, R_pivots], then remove the relation
// component of B_new: alpha = C^-1 c_T, beta = r_T - D alpha.
std::optional<std::vector<std::size_t>>
reuse_provisional_support(const ProvisionalCheckpoint& checkpoint,
                          const ProbeSystemAssembler<NativeFieldElement>& assembler,
                          const IndexedColumns& basis, const IndexedColumns& targets,
                          bool projected)
{
  using T = NativeFieldElement;
  const auto n = basis.size();
  const auto target_width = projected ? std::size_t{2} : targets.size();
  const auto width = n + target_width;
  if (checkpoint.basis_columns != n)
    throw std::logic_error("checkpoint basis dimension mismatch");
  std::vector<T> rhs(checkpoint.row_map.size() * width, T(0));
  for (std::size_t i = 0; i < n; ++i)
    assembler.add_rhs_column(rhs, width, basis, i, i);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    for (const auto& term : targets.column(i)) {
      const T value = assembler.evaluate(term);
      if (projected) {
        const T weight(i + 1);
        rhs[term.row * width + n] = rhs[term.row * width + n] + value * weight;
        rhs[term.row * width + n + 1] =
            rhs[term.row * width + n + 1] + value * weight * (weight + T(1));
      } else
        rhs[term.row * width + n + i] = rhs[term.row * width + n + i] + value;
    }
  }
  auto solution = checkpoint.solve(std::move(rhs), width, firefly::FFInt::p);
  if (!solution) return std::nullopt;
  if (checkpoint.solution_columns.size() < n)
    throw std::logic_error("checkpoint has no full basis prefix");
  std::vector<T> change(solution->begin(),
                        solution->begin() + static_cast<std::ptrdiff_t>(n * width));
  for (std::size_t k = 0; k < n; ++k) {
    if (checkpoint.solution_columns[k] != k)
      throw std::logic_error("checkpoint basis prefix is inconsistent");
    std::size_t pivot = k;
    while (pivot < n && change[pivot * width + k] == T(0))
      ++pivot;
    if (pivot == n) return std::nullopt;
    if (pivot != k)
      for (std::size_t j = 0; j < width; ++j)
        std::swap(change[k * width + j], change[pivot * width + j]);
    const T inverse = T(1) / change[k * width + k];
    for (std::size_t j = k; j < width; ++j)
      change[k * width + j] = change[k * width + j] * inverse;
    for (std::size_t i = 0; i < n; ++i) {
      if (i == k) continue;
      const T factor = change[i * width + k];
      for (std::size_t j = k; j < width; ++j)
        change[i * width + j] = change[i * width + j] - factor * change[k * width + j];
    }
  }
  std::vector<std::size_t> support;
  for (std::size_t i = n; i < checkpoint.solution_columns.size(); ++i) {
    bool live = false;
    for (std::size_t j = 0; j < target_width; ++j) {
      T value = (*solution)[i * width + n + j];
      for (std::size_t k = 0; k < n; ++k)
        value = value - (*solution)[i * width + k] * change[k * width + n + j];
      live = live || value != T(0);
    }
    if (live) support.push_back(checkpoint.solution_columns[i] - n);
  }
  return support;
}

CompactSelection plan_compact_kernel_impl(
    const IndexedColumns& basis_columns, const IndexedColumns& target_columns,
    const IndexedColumns& ansatz_columns,
    std::span<const AnsatzColumnMeta> ansatz_metadata,
    std::span<const std::uint32_t> row_sectors,
    std::span<const firefly::FFInt> polynomial_values,
    std::span<const firefly::FFInt> top_lp_coefficients,
    const firefly::FFInt& minus_half_d, bool project_target_support,
    std::span<const std::uint32_t> explicit_row_groups,
    std::span<const std::uint32_t> ordered_groups, AnsatzDotOrdering dot_ordering,
    std::size_t planning_threads, bool check_master_independence,
    bool capture_provisional, const ProvisionalCheckpoint* checkpoint)
{
  using T = NativeFieldElement;
  using Clock = std::chrono::high_resolution_clock;
  const auto milliseconds = [](auto begin, auto end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  };
  const std::size_t num_rows = row_sectors.size();
  const std::size_t num_basis_cols = basis_columns.size();
  const std::size_t num_targets = target_columns.size();
  const auto row_groups =
      explicit_row_groups.empty() ? row_sectors : explicit_row_groups;
  if (row_groups.size() != num_rows)
    throw std::logic_error("compact planner row-group shape is inconsistent");
  T::set_prime(firefly::FFInt::p);
  const auto native_polynomial_values = encode_field_values(polynomial_values);
  const auto native_top_lp_coefficients = encode_field_values(top_lp_coefficients);
  const T native_minus_half_d = T::from_residue(minus_half_d.n);
  const ProbeSystemAssembler<T> assembler(
      native_polynomial_values, native_top_lp_coefficients, native_minus_half_d);
  const auto build_probe_system = [&](std::span<const std::size_t> ordered_ansatz,
                                      bool project_targets) {
    const std::size_t rhs_columns = project_targets ? 2 : num_targets;
    ProbeSystem system{linalg::SparseMatrix<T>(num_rows),
                       std::vector<T>(num_rows * rhs_columns, T(0)), rhs_columns};
    for (std::size_t column = 0; column < basis_columns.size(); ++column)
      assembler.add_matrix_column(system.matrix, basis_columns, column, column);
    for (std::size_t column = 0; column < ordered_ansatz.size(); ++column)
      assembler.add_matrix_column(system.matrix, ansatz_columns, ordered_ansatz[column],
                                  num_basis_cols + column);
    for (std::size_t target = 0; target < target_columns.size(); ++target) {
      for (const auto& term : target_columns.column(target)) {
        const T value = assembler.evaluate(term);
        if (project_targets) {
          const T weight = T(static_cast<std::int64_t>(target + 1));
          system.rhs[term.row * rhs_columns] =
              system.rhs[term.row * rhs_columns] + value * weight;
          system.rhs[term.row * rhs_columns + 1] =
              system.rhs[term.row * rhs_columns + 1] + value * weight * (weight + T(1));
        } else {
          system.rhs[term.row * rhs_columns + target] =
              system.rhs[term.row * rhs_columns + target] + value;
        }
      }
    }
    return system;
  };
  const auto basis_is_full_rank = [&](const auto& columns) {
    return columns.size() >= num_basis_cols &&
           std::ranges::equal(columns | std::views::take(num_basis_cols),
                              std::views::iota(std::size_t{0}, num_basis_cols));
  };
  const auto column_sector_groups = [&](std::span<const std::size_t> ordered_ansatz) {
    std::vector<std::uint32_t> groups;
    groups.reserve(num_basis_cols + ordered_ansatz.size());
    for (std::size_t column = 0; column < basis_columns.size(); ++column) {
      const auto terms = basis_columns.column(column);
      if (terms.empty()) throw std::logic_error("basis column has no equation row");
      groups.push_back(row_groups[terms.front().row]);
    }
    for (const std::size_t column : ordered_ansatz)
      groups.push_back(ansatz_pivot_group(ansatz_metadata[column]));
    return groups;
  };
  const auto column_priorities = [&](std::span<const std::size_t> ordered_ansatz) {
    std::vector<std::uint32_t> priorities(num_basis_cols + ordered_ansatz.size());
    for (std::size_t column = 0; column < num_basis_cols; ++column)
      priorities[column] = static_cast<std::uint32_t>(column);
    std::vector<std::size_t> positions(ordered_ansatz.size());
    std::iota(positions.begin(), positions.end(), 0);
    std::ranges::sort(positions, [&](std::size_t lhs, std::size_t rhs) {
      const auto& a = ansatz_metadata[ordered_ansatz[lhs]];
      const auto& b = ansatz_metadata[ordered_ansatz[rhs]];
      if (a.family != b.family) return a.family < b.family;
      if (a.grid_index != b.grid_index) return a.grid_index < b.grid_index;
      if (a.derivative_index != b.derivative_index)
        return a.derivative_index < b.derivative_index;
      return ordered_ansatz[lhs] < ordered_ansatz[rhs];
    });
    for (std::size_t rank = 0; rank < positions.size(); ++rank) {
      if (num_basis_cols + positions[rank] >
          std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Markowitz column priority exceeds 32-bit ids");
      }
      priorities[num_basis_cols + positions[rank]] =
          static_cast<std::uint32_t>(num_basis_cols + rank);
    }
    return priorities;
  };
  const auto column_complexities = [&](std::span<const std::size_t> ordered_ansatz) {
    std::vector<std::uint32_t> complexities(num_basis_cols + ordered_ansatz.size());
    for (std::size_t position = 0; position < ordered_ansatz.size(); ++position) {
      complexities[num_basis_cols + position] =
          ansatz_metadata[ordered_ansatz[position]].seed_dot_excess;
    }
    return complexities;
  };
  CompactSelection selection;
  selection.provisional_rhs_columns = project_target_support ? 2 : num_targets;
  Clock::time_point start;
  Clock::time_point provisional_eliminated;
  // Only the selected support and statistics cross into the final assembly.
  // Release the larger provisional echelon before allocating that system.
  if (checkpoint) {
    start = Clock::now();
    if (checkpoint->row_map.size() != num_rows ||
        checkpoint->relation_columns != ansatz_columns.size())
      throw std::logic_error("checkpoint source shape mismatch");
    if (check_master_independence && !checkpoint->master_independence_checked)
      return selection;
    auto support = reuse_provisional_support(*checkpoint, assembler, basis_columns,
                                             target_columns, project_target_support);
    if (!support) return selection;
    selection.ansatz_order = std::move(*support);
    selection.provisional_dimension = checkpoint->solution_columns.size();
    selection.reused_provisional = true;
    provisional_eliminated = start;
  } else {
    std::vector<std::size_t> provisional_order(ansatz_columns.size());
    std::iota(provisional_order.begin(), provisional_order.end(), 0);
    const std::size_t generated_column_count =
        num_basis_cols + provisional_order.size();
    start = Clock::now();
    auto provisional_system =
        build_probe_system(provisional_order, project_target_support);
    const auto provisional_groups = column_sector_groups(provisional_order);
    const auto provisional_priorities = column_priorities(provisional_order);
    const auto provisional_complexities = column_complexities(provisional_order);
    const auto provisional_built = Clock::now();
    linalg::SparseMarkowitzStatistics markowitz_statistics;
    // A broad target batch already requests a wide support, so a wider score
    // refresh batch substantially reduces planning work on large systems without
    // the sparse-support instability seen for narrow projected target sets. Keep
    // smaller systems on the tighter policy because their refresh cost is modest.
    const std::size_t score_refresh_interval = linalg::compact_score_refresh_interval(
        target_columns.size(), generated_column_count);
    std::vector<linalg::NumericalRowOperation<T>> trace;
    auto provisional_result = linalg::sparse_markowitz_elimination(
        provisional_system.matrix, generated_column_count, provisional_system.rhs,
        provisional_system.rhs_columns, num_basis_cols, row_groups, provisional_groups,
        provisional_priorities, ordered_groups, provisional_complexities,
        complexity_order(dot_ordering), &markowitz_statistics, planning_threads,
        score_refresh_interval, capture_provisional ? &trace : nullptr);
    if (check_master_independence) {
      const auto audit =
          audit_master_independence(provisional_system.matrix, provisional_result,
                                    generated_column_count, num_basis_cols);
      selection.timings.master_rank_completion_ms = audit.completion_ms;
      selection.timings.master_rank_check_ms = audit.check_ms;
    }
    provisional_eliminated = Clock::now();
    selection.timings.provisional_build_ms = milliseconds(start, provisional_built);
    selection.timings.provisional_elimination_ms =
        milliseconds(provisional_built, provisional_eliminated);
    assign_markowitz_statistics(selection, markowitz_statistics);
    if (!provisional_result.closed) {
      auto residual = collect_residual_rows(provisional_result,
                                            std::span<const T>(provisional_system.rhs),
                                            provisional_system.rhs_columns);
      selection.residual_rows = std::move(residual);
      selection.timings.total_ms = milliseconds(start, provisional_eliminated);
      return selection;
    }
    if (!basis_is_full_rank(provisional_result.solution_cols))
      throw std::runtime_error("master basis is not full rank");

    if (capture_provisional) {
      selection.checkpoint = std::make_shared<ProvisionalCheckpoint>();
      selection.checkpoint->master_independence_checked = check_master_independence;
      selection.checkpoint->capture(provisional_system.matrix, provisional_result,
                                    trace, num_basis_cols, ansatz_columns.size(),
                                    firefly::FFInt::p);
    }
    auto provisional_solution = linalg::back_substitute_sparse_echelon(
        provisional_system.matrix, provisional_system.rhs,
        provisional_system.rhs_columns, provisional_result.solution_cols,
        provisional_result.row_map);
    selection.provisional_dimension = provisional_result.solution_cols.size();
    selection.ansatz_order.reserve(provisional_result.solution_cols.size() -
                                   num_basis_cols);
    for (std::size_t position = num_basis_cols;
         position < provisional_result.solution_cols.size(); ++position) {
      bool live = false;
      for (std::size_t rhs = 0; rhs < provisional_system.rhs_columns; ++rhs) {
        if (provisional_solution[position * provisional_system.rhs_columns + rhs] !=
            T(0)) {
          live = true;
          break;
        }
      }
      if (!live) continue;
      const std::size_t provisional_column = provisional_result.solution_cols[position];
      if (provisional_column < num_basis_cols ||
          provisional_column - num_basis_cols >= provisional_order.size()) {
        throw std::logic_error("live ansatz column is out of range");
      }
      selection.ansatz_order.push_back(
          provisional_order[provisional_column - num_basis_cols]);
    }
  }
  const auto support_selected = Clock::now();

  const std::size_t total_cols = num_basis_cols + selection.ansatz_order.size();
  auto final_system = build_probe_system(selection.ansatz_order, false);
  const auto final_groups = column_sector_groups(selection.ansatz_order);
  const auto compact_built = Clock::now();
  auto final_result = linalg::sparse_gaussian_elimination(
      final_system.matrix, total_cols, final_system.rhs, num_targets, num_basis_cols,
      row_groups, final_groups);
  const auto compact_eliminated = Clock::now();
  if (!final_result.closed && (project_target_support || checkpoint)) {
    selection.closed = false;
    selection.timings.total_ms = milliseconds(start, compact_eliminated);
    return selection;
  }
  if (!final_result.closed)
    throw std::runtime_error(
        "target-support compaction did not preserve ansatz closure");
  if (!basis_is_full_rank(final_result.solution_cols))
    throw std::runtime_error("master basis is not full rank");

  selection.closed = true;
  selection.cross_group_pivots = final_result.cross_group_pivots;
  selection.solution_columns = std::move(final_result.solution_cols);
  selection.elimination_row_map = std::move(final_result.row_map);
  selection.timings.support_selection_ms =
      milliseconds(provisional_eliminated, support_selected);
  selection.timings.compact_build_ms = milliseconds(support_selected, compact_built);
  selection.timings.compact_elimination_ms =
      milliseconds(compact_built, compact_eliminated);
  selection.timings.total_ms = milliseconds(start, compact_eliminated);
  return selection;
}

} // namespace

CompactSelection plan_compact_kernel(
    const IndexedColumns& basis_columns, const IndexedColumns& target_columns,
    const IndexedColumns& ansatz_columns,
    std::span<const AnsatzColumnMeta> ansatz_metadata,
    std::span<const std::uint32_t> row_sectors,
    std::span<const firefly::FFInt> polynomial_values,
    std::span<const firefly::FFInt> top_lp_coefficients,
    const firefly::FFInt& minus_half_d, std::span<const std::uint32_t> row_groups,
    std::span<const std::uint32_t> ordered_groups, AnsatzDotOrdering dot_ordering,
    std::size_t planning_threads, bool check_master_independence,
    bool capture_provisional, const ProvisionalCheckpoint* checkpoint)
{
  CompactPhaseTimings projected_timings;
  if (target_columns.size() > 2) {
    auto projected = plan_compact_kernel_impl(
        basis_columns, target_columns, ansatz_columns, ansatz_metadata, row_sectors,
        polynomial_values, top_lp_coefficients, minus_half_d, true, row_groups,
        ordered_groups, dot_ordering, planning_threads, check_master_independence,
        capture_provisional, checkpoint);
    if (projected.closed) return projected;
    projected_timings = projected.timings;
  }
  auto full = plan_compact_kernel_impl(
      basis_columns, target_columns, ansatz_columns, ansatz_metadata, row_sectors,
      polynomial_values, top_lp_coefficients, minus_half_d, false, row_groups,
      ordered_groups, dot_ordering, planning_threads, check_master_independence,
      capture_provisional, checkpoint);
  full.timings.master_rank_completion_ms += projected_timings.master_rank_completion_ms;
  full.timings.master_rank_check_ms += projected_timings.master_rank_check_ms;
  full.provisional_rhs_fallback = target_columns.size() > 2;
  full.timings.provisional_build_ms += projected_timings.provisional_build_ms;
  full.timings.provisional_elimination_ms +=
      projected_timings.provisional_elimination_ms;
  full.timings.provisional_relation_elimination_ms +=
      projected_timings.provisional_relation_elimination_ms;
  full.timings.provisional_score_refresh_ms +=
      projected_timings.provisional_score_refresh_ms;
  full.timings.provisional_row_elimination_ms +=
      projected_timings.provisional_row_elimination_ms;
  full.timings.support_selection_ms += projected_timings.support_selection_ms;
  full.timings.compact_build_ms += projected_timings.compact_build_ms;
  full.timings.compact_elimination_ms += projected_timings.compact_elimination_ms;
  full.timings.total_ms += projected_timings.total_ms;
  return full;
}

LocalReselectionStatistics
reselect_compact_support(CompactSelection& selection,
                         const KernelPublicationInput& source,
                         std::span<const firefly::FFInt> top_lp_coefficients,
                         const firefly::FFInt& minus_half_d,
                         AnsatzDotOrdering dot_ordering, std::size_t planning_threads)
{
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const auto elapsed = [](auto begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
  };
  if (!selection.closed || !selection.reused_provisional)
    throw std::logic_error("local reselection requires a closed provisional support");
  const auto row_count = source.row_sectors.size();
  const auto relation_count = source.ansatz_columns.size();
  if (source.ansatz_metadata.size() != relation_count ||
      (!source.row_groups.empty() && source.row_groups.size() != row_count))
    throw std::logic_error("local reselection source metadata mismatch");
  LocalReselectionStatistics statistics;
  statistics.initial_ms = selection.timings.total_ms;
  std::vector<bool> rows(row_count, false), original(relation_count, false);
  const auto mark = [&](std::span<const IndexedTerm> terms) {
    for (const auto& term : terms)
      rows.at(term.row) = true;
  };
  mark(source.basis_columns.terms);
  mark(source.target_columns.terms);
  for (auto id : selection.ansatz_order) {
    original.at(id) = true;
    mark(source.ansatz_columns.column(id));
  }
  const auto absent = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> row_map(row_count, absent), original_rows, eligible;
  KernelPublicationInput local;
  for (std::size_t row = 0; row < row_count; ++row) {
    if (!rows[row]) continue;
    row_map[row] = original_rows.size();
    original_rows.push_back(row);
    local.row_sectors.push_back(source.row_sectors[row]);
    if (!source.row_groups.empty()) local.row_groups.push_back(source.row_groups[row]);
  }
  statistics.envelope_rows = original_rows.size();
  // Eligibility uses every symbolic term, including terms vanishing at this point.
  for (std::size_t id = 0; id < relation_count; ++id) {
    const auto terms = source.ansatz_columns.column(id);
    if (std::ranges::all_of(terms,
                            [&](const auto& term) { return rows.at(term.row); })) {
      eligible.push_back(id);
      statistics.added_relations += !original[id];
    }
  }
  statistics.eligible_relations = eligible.size();
  if (statistics.added_relations == 0) {
    statistics.scan_ms = elapsed(start);
    selection.timings.total_ms += statistics.scan_ms;
    return statistics;
  }
  const auto append = [&](const IndexedColumns& columns, std::size_t id,
                          IndexedColumns& destination) {
    for (auto term : columns.column(id)) {
      if (row_map.at(term.row) == absent)
        throw std::logic_error("local relation escaped symbolic row envelope");
      term.row = static_cast<std::uint32_t>(row_map[term.row]);
      destination.terms.push_back(term);
    }
    destination.offsets.push_back(destination.terms.size());
  };
  for (std::size_t id = 0; id < source.basis_columns.size(); ++id)
    append(source.basis_columns, id, local.basis_columns);
  for (std::size_t id = 0; id < source.target_columns.size(); ++id)
    append(source.target_columns, id, local.target_columns);
  for (auto id : eligible) {
    append(source.ansatz_columns, id, local.ansatz_columns);
    local.ansatz_metadata.push_back(source.ansatz_metadata[id]);
  }
  // The Markowitz order must name exactly the relation groups still present.
  // Restrict the original order without changing its relative ordering.
  if (!source.ordered_groups.empty()) {
    std::vector<std::uint32_t> present;
    for (const auto& meta : local.ansatz_metadata)
      present.push_back(ansatz_pivot_group(meta));
    std::ranges::sort(present);
    present.erase(std::unique(present.begin(), present.end()), present.end());
    auto supplied = source.ordered_groups;
    std::ranges::sort(supplied);
    if (std::ranges::adjacent_find(supplied) != supplied.end())
      throw std::logic_error(
          "local reselection source group order contains duplicates");
    for (auto group : source.ordered_groups)
      if (std::ranges::binary_search(present, group))
        local.ordered_groups.push_back(group);
    if (local.ordered_groups.size() != present.size())
      throw std::logic_error("local reselection source group order is incomplete");
  }
  statistics.scan_ms = elapsed(start);
  statistics.attempted = true;
  const auto reselect_started = Clock::now();
  auto result = plan_compact_kernel(
      local.basis_columns, local.target_columns, local.ansatz_columns,
      local.ansatz_metadata, local.row_sectors, source.polynomial_values,
      top_lp_coefficients, minus_half_d, local.row_groups, local.ordered_groups,
      dot_ordering, planning_threads);
  statistics.reselect_ms = elapsed(reselect_started);
  for (auto& id : result.ansatz_order) {
    id = eligible.at(id);
    statistics.selected_new_relations += !original.at(id);
  }
  for (auto& row : result.elimination_row_map)
    row = original_rows.at(row);
  if (result.closed) {
    // Publication expects a permutation of the full original row catalogue.
    for (std::size_t row = 0; row < row_count; ++row)
      if (!rows[row]) result.elimination_row_map.push_back(row);
  }
  for (auto& row : result.residual_rows)
    row = original_rows.at(row);
  result.reused_provisional = true;
  result.timings.total_ms = statistics.initial_ms + elapsed(start);
  selection = std::move(result);
  return statistics;
}

} // namespace reduction::detail
