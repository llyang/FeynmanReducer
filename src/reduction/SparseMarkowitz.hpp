#pragma once

#include "core/ParallelForExecutor.hpp"
#include "reduction/EliminationTape.hpp"

#include <bit>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace linalg {

enum class ColumnComplexityOrder {
  Ignore,
  Ascending,
  Descending,
};

struct SparseMarkowitzStatistics {
  double relation_elimination_ms = 0.0;
  std::size_t relation_pivots = 0;
  double score_refresh_ms = 0.0;
  double row_elimination_ms = 0.0;
  std::size_t score_refresh_columns = 0;
  std::size_t incidence_records_scanned = 0;
  std::size_t parallel_refresh_batches = 0;
  std::size_t parallel_refresh_columns = 0;
  std::size_t stale_choice_pops = 0;
  std::size_t choice_queue_compactions = 0;
  std::size_t compacted_choice_entries = 0;
  std::size_t maximum_choice_queue_size = 0;
  std::size_t row_eliminations = 0;
  std::size_t parallel_row_batches = 0;
  std::size_t parallel_row_eliminations = 0;
};

[[nodiscard]] inline constexpr std::size_t
compact_score_refresh_interval(std::size_t target_count,
                               std::size_t candidate_column_count) noexcept
{
  constexpr std::size_t broad_target_support = 16;
  constexpr std::size_t large_candidate_system = 65536;
  return target_count >= broad_target_support &&
                 candidate_column_count >= large_candidate_system
             ? 32
             : 16;
}

/// Group-aware sparse elimination with a deterministic Markowitz pivot rule.
/// Basis columns form a fixed required prefix. Remaining columns follow the
/// supplied group order (or its deterministic default); inside one group both
/// the column and pivot row are selected dynamically by the Markowitz product,
/// preferring a row from the column's matching group when one is available.
template <typename T, bool CompactChoiceQueues = true>
[[nodiscard]] EliminationResult sparse_markowitz_elimination(
    SparseMatrix<T>& matrix, size_t num_cols, std::vector<T>& right_hand_side,
    size_t rhs_cols, size_t required_prefix_cols,
    std::span<const std::uint32_t> row_groups,
    std::span<const std::uint32_t> column_groups,
    std::span<const std::uint32_t> column_priorities,
    std::span<const std::uint32_t> explicit_group_order = {},
    std::span<const std::uint32_t> column_complexities = {},
    ColumnComplexityOrder complexity_order = ColumnComplexityOrder::Ignore,
    SparseMarkowitzStatistics* statistics = nullptr, std::size_t planning_threads = 1,
    std::size_t score_refresh_interval_hint = 0)
{
  using Clock = std::chrono::high_resolution_clock;
  const auto milliseconds = [](auto begin, auto end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  };
  const size_t num_rows = matrix.size();
  planning_threads = std::max<std::size_t>(planning_threads, 1);
  if (rhs_cols > 0 && right_hand_side.size() != num_rows * rhs_cols)
    throw std::runtime_error("sparse RHS shape mismatch");
  if (required_prefix_cols > num_cols)
    throw std::runtime_error("required column prefix exceeds matrix width");
  if (row_groups.size() != num_rows || column_groups.size() != num_cols ||
      column_priorities.size() != num_cols ||
      (!column_complexities.empty() && column_complexities.size() != num_cols)) {
    throw std::runtime_error("Markowitz metadata shape mismatch");
  }
  if (num_rows > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("Markowitz row count exceeds 32-bit encoding");
  if (num_rows == 0) return {{}, true, {}};

  const std::size_t active_rhs_cols = rhs_cols;
  MatrixView<T> rhs{right_hand_side.data(), active_rhs_cols};
  std::vector<size_t> rhs_row_nonzeros(num_rows, 0);
  auto count_rhs_row_nonzeros = [&](size_t row) {
    size_t count = 0;
    for (size_t column = 0; column < active_rhs_cols; ++column) {
      if (rhs(row, column) != T(0)) ++count;
    }
    return count;
  };
  size_t residual_nonzero_rows = 0;
  for (size_t row = 0; row < num_rows; ++row) {
    rhs_row_nonzeros[row] = count_rhs_row_nonzeros(row);
    if (rhs_row_nonzeros[row] != 0) ++residual_nonzero_rows;
  }

  struct ColumnIncidence {
    std::uint32_t row;
    std::uint32_t position;
    std::uint32_t row_version;
  };
  static_assert(sizeof(ColumnIncidence) == 3 * sizeof(std::uint32_t));
  std::vector<std::vector<ColumnIncidence>> column_rows(num_cols);
  std::vector<std::uint32_t> row_versions(num_rows, 0);
  std::vector<size_t> row_active_nonzeros(num_rows, 0);
  std::vector<size_t> column_active_nonzeros(num_cols, 0);
  for (size_t row = 0; row < num_rows; ++row) {
    for (size_t position = 0; position < matrix[row].size(); ++position) {
      const auto& entry = matrix[row][position];
      if (entry.column >= num_cols)
        throw std::runtime_error("sparse matrix column out of range");
      if (position > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("sparse row position exceeds 32-bit encoding");
      column_rows[entry.column].push_back(
          {static_cast<std::uint32_t>(row), static_cast<std::uint32_t>(position), 0});
      ++row_active_nonzeros[row];
      ++column_active_nonzeros[entry.column];
    }
  }

  auto find_incidence = [&](size_t column, ColumnIncidence& incidence) -> const T* {
    const auto& row = matrix[incidence.row];
    if (incidence.row_version == row_versions[incidence.row] &&
        incidence.position < row.size() && row[incidence.position].column == column) {
      return &row[incidence.position].value;
    }
    const auto found = sparse_lower_bound(row, column);
    if (found == row.end() || found->column != column) return nullptr;
    const size_t position = static_cast<size_t>(found - row.begin());
    if (position > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("sparse row position exceeds 32-bit encoding");
    incidence.position = static_cast<std::uint32_t>(position);
    incidence.row_version = row_versions[incidence.row];
    return &found->value;
  };

  std::vector<bool> active_columns(num_cols, true);
  std::vector<bool> used_rows(num_rows, false);
  struct ScanWorkspace {
    explicit ScanWorkspace(std::size_t rows)
        : seen_rows(rows, std::numeric_limits<std::uint32_t>::max())
    {}
    std::vector<std::uint32_t> seen_rows;
    std::uint32_t generation = 0;
  };
  std::vector<ScanWorkspace> scan_workspaces;
  scan_workspaces.reserve(planning_threads);
  scan_workspaces.emplace_back(num_rows);
  std::vector<size_t> row_map;
  row_map.reserve(num_rows);
  std::vector<size_t> solution_cols;
  solution_cols.reserve(std::min(num_rows, num_cols));
  size_t cross_group_pivots = 0;
  const auto elimination_needed = [&] { return residual_nonzero_rows != 0; };

  std::uint32_t current_group = std::numeric_limits<std::uint32_t>::max();
  std::vector<bool> dirty_columns(num_cols, false);
  std::vector<size_t> dirty_list;
  auto mark_dirty = [&](size_t column) {
    if (!active_columns[column] || column_groups[column] != current_group ||
        dirty_columns[column]) {
      return;
    }
    dirty_columns[column] = true;
    dirty_list.push_back(column);
  };

  struct PivotChoice {
    size_t column = std::numeric_limits<size_t>::max();
    size_t row = std::numeric_limits<size_t>::max();
    size_t markowitz = std::numeric_limits<size_t>::max();
    size_t column_nonzeros = std::numeric_limits<size_t>::max();
    size_t row_nonzeros = std::numeric_limits<size_t>::max();
    std::uint32_t priority = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t complexity = 0;
    std::uint32_t version = 0;
    std::uint32_t row_position = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t row_version = 0;
  };
  const auto better = [complexity_order](const PivotChoice& lhs,
                                         const PivotChoice& rhs) {
    if (lhs.complexity != rhs.complexity) {
      if (complexity_order == ColumnComplexityOrder::Ascending)
        return lhs.complexity < rhs.complexity;
      if (complexity_order == ColumnComplexityOrder::Descending)
        return lhs.complexity > rhs.complexity;
    }
    return std::tuple(lhs.markowitz, lhs.column_nonzeros, lhs.row_nonzeros,
                      lhs.priority, lhs.column, lhs.row) <
           std::tuple(rhs.markowitz, rhs.column_nonzeros, rhs.row_nonzeros,
                      rhs.priority, rhs.column, rhs.row);
  };
  auto saturating_product = [](size_t lhs, size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
      return std::numeric_limits<size_t>::max();
    return lhs * rhs;
  };

  struct ColumnChoices {
    PivotChoice matching;
    PivotChoice fallback;
    std::size_t incidence_records_scanned = 0;
  };
  auto for_each_live_column_row = [&](size_t column, ScanWorkspace& workspace,
                                      auto&& visitor) {
    auto& incidence = column_rows[column];
    const std::size_t records_scanned = incidence.size();
    size_t write = 0;
    if (workspace.generation == std::numeric_limits<std::uint32_t>::max()) {
      std::ranges::fill(workspace.seen_rows, std::numeric_limits<std::uint32_t>::max());
      workspace.generation = 0;
    }
    const std::uint32_t current_generation = workspace.generation++;
    for (auto item : incidence) {
      const size_t row = item.row;
      if (workspace.seen_rows[row] == current_generation) continue;
      workspace.seen_rows[row] = current_generation;
      if (used_rows[row]) continue;
      const T* value = find_incidence(column, item);
      if (value == nullptr || *value == T(0)) continue;
      incidence[write++] = item;
      visitor(item, *value);
    }
    incidence.resize(write);
    return records_scanned;
  };
  auto choices_for_column = [&](size_t column, ScanWorkspace& workspace) {
    ColumnChoices choices;
    const size_t column_nonzeros = column_active_nonzeros[column];
    choices.incidence_records_scanned = for_each_live_column_row(
        column, workspace, [&](const ColumnIncidence& incidence, const T&) {
          const size_t row = incidence.row;
          const size_t row_nonzeros = row_active_nonzeros[row];
          if (row_nonzeros == 0) return;
          PivotChoice candidate;
          candidate.column = column;
          candidate.row = row;
          candidate.markowitz =
              saturating_product(row_nonzeros - 1, column_nonzeros - 1);
          candidate.column_nonzeros = column_nonzeros;
          candidate.row_nonzeros = row_nonzeros;
          candidate.priority = column_priorities[column];
          if (!column_complexities.empty())
            candidate.complexity = column_complexities[column];
          candidate.row_position = incidence.position;
          candidate.row_version = incidence.row_version;
          if (choices.fallback.column == std::numeric_limits<size_t>::max() ||
              better(candidate, choices.fallback)) {
            choices.fallback = candidate;
          }
          if (row_groups[row] == column_groups[column] &&
              (choices.matching.column == std::numeric_limits<size_t>::max() ||
               better(candidate, choices.matching))) {
            choices.matching = candidate;
          }
        });
    return choices;
  };

  auto choice_value = [&](const PivotChoice& choice) -> const T* {
    const auto& row = matrix[choice.row];
    if (choice.row_version == row_versions[choice.row] &&
        choice.row_position < row.size() &&
        row[choice.row_position].column == choice.column) {
      return &row[choice.row_position].value;
    }
    return sparse_find(row, choice.column);
  };

  enum class RowChangeKind : std::uint8_t {
    New,
    Removed,
  };
  struct RowChange {
    std::size_t column;
    std::size_t position;
    RowChangeKind kind;
  };
  struct RowEliminationTask {
    std::uint32_t row;
    T factor;
    std::uint32_t updated_version;
    std::size_t updated_rhs_nonzeros;
    std::vector<RowChange> changes;
  };
  std::unique_ptr<core::ParallelForExecutor> planning_executor;
  std::vector<RowEliminationTask> parallel_row_tasks;

  auto apply_pivot = [&](const PivotChoice& choice) {
    const size_t column = choice.column;
    const size_t best_row = choice.row;
    if (rhs_row_nonzeros[best_row] != 0) --residual_nonzero_rows;
    used_rows[best_row] = true;
    row_map.push_back(best_row);
    solution_cols.push_back(column);
    if (statistics != nullptr) ++statistics->relation_pivots;

    auto& pivot_row = matrix[best_row];
    const T* pivot_value = choice_value(choice);
    if (pivot_value == nullptr || *pivot_value == T(0))
      throw std::logic_error("Markowitz pivot disappeared");
    const T inverse = T(1) / *pivot_value;
    for (auto& entry : pivot_row) {
      if (active_columns[entry.column]) entry.value = entry.value * inverse;
    }
    for (size_t rhs_column = 0; rhs_column < active_rhs_cols; ++rhs_column) {
      if (rhs(best_row, rhs_column) != T(0))
        rhs(best_row, rhs_column) = rhs(best_row, rhs_column) * inverse;
    }

    for (const auto& entry : pivot_row) {
      if (!active_columns[entry.column]) continue;
      if (column_active_nonzeros[entry.column] == 0)
        throw std::logic_error("Markowitz column count underflow");
      --column_active_nonzeros[entry.column];
      mark_dirty(entry.column);
    }
    row_active_nonzeros[best_row] = 0;

    // The row merge replaces the source vector, so keep the factor by value for
    // the subsequent RHS update.
    const auto row_elimination_start = Clock::now();
    const auto eliminate_serial_row = [&](const ColumnIncidence& incidence, T factor) {
      const size_t row = incidence.row;
      if (statistics != nullptr) ++statistics->row_eliminations;
      const bool rhs_was_nonzero = rhs_row_nonzeros[row] != 0;
      if (row_versions[row] == std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Markowitz row version exceeds 32-bit encoding");
      const std::uint32_t updated_version = row_versions[row] + 1;
      sparse_eliminate_active_row(
          matrix[row], pivot_row, column, active_columns, factor,
          [&](size_t new_column, size_t new_position) {
            if (new_position > std::numeric_limits<std::uint32_t>::max())
              throw std::runtime_error("sparse row position exceeds 32-bit encoding");
            column_rows[new_column].push_back({static_cast<std::uint32_t>(row),
                                               static_cast<std::uint32_t>(new_position),
                                               updated_version});
            ++column_active_nonzeros[new_column];
            mark_dirty(new_column);
          },
          [&](size_t removed_column) {
            if (column_active_nonzeros[removed_column] == 0)
              throw std::logic_error("Markowitz column count underflow");
            --column_active_nonzeros[removed_column];
            mark_dirty(removed_column);
          },
          [&](size_t retained_column, size_t) { mark_dirty(retained_column); });
      row_versions[row] = updated_version;
      row_active_nonzeros[row] = matrix[row].size();
      for (size_t rhs_column = 0; rhs_column < active_rhs_cols; ++rhs_column) {
        if (rhs(best_row, rhs_column) != T(0)) {
          const bool was_nonzero = rhs(row, rhs_column) != T(0);
          const T updated = rhs(row, rhs_column) - factor * rhs(best_row, rhs_column);
          rhs(row, rhs_column) = updated;
          const bool is_nonzero = updated != T(0);
          if (was_nonzero != is_nonzero) {
            if (is_nonzero)
              ++rhs_row_nonzeros[row];
            else
              --rhs_row_nonzeros[row];
          }
        }
      }
      const bool rhs_is_nonzero = rhs_row_nonzeros[row] != 0;
      if (rhs_was_nonzero != rhs_is_nonzero) {
        if (rhs_is_nonzero)
          ++residual_nonzero_rows;
        else
          --residual_nonzero_rows;
      }
    };
    std::size_t pivot_incidence_records = 0;
    const std::size_t affected_rows = column_active_nonzeros[column];
    const std::size_t row_work_lower_bound =
        saturating_product(affected_rows, pivot_row.size() + active_rhs_cols);
    constexpr std::size_t min_parallel_row_work = 4096;
    const bool use_parallel_rows = planning_threads > 1 &&
                                   affected_rows >= 4 * planning_threads &&
                                   row_work_lower_bound >= min_parallel_row_work;
    if (!use_parallel_rows) {
      pivot_incidence_records = for_each_live_column_row(
          column, scan_workspaces.front(), eliminate_serial_row);
    } else {
      std::size_t task_count = 0;
      pivot_incidence_records = for_each_live_column_row(
          column, scan_workspaces.front(),
          [&](const ColumnIncidence& incidence, T factor) {
            const size_t row = incidence.row;
            if (row_versions[row] == std::numeric_limits<std::uint32_t>::max()) {
              throw std::runtime_error("Markowitz row version exceeds 32-bit encoding");
            }
            if (task_count == parallel_row_tasks.size()) {
              parallel_row_tasks.push_back({incidence.row,
                                            factor,
                                            row_versions[row] + 1,
                                            rhs_row_nonzeros[row],
                                            {}});
            } else {
              auto& task = parallel_row_tasks[task_count];
              task.row = incidence.row;
              task.factor = factor;
              task.updated_version = row_versions[row] + 1;
              task.updated_rhs_nonzeros = rhs_row_nonzeros[row];
              task.changes.clear();
            }
            ++task_count;
          });
      if (planning_executor == nullptr) {
        planning_executor =
            std::make_unique<core::ParallelForExecutor>(planning_threads);
      }
      // Each task owns a distinct matrix/RHS row. The executor dynamically
      // balances task indices; result slots and the serial metadata commit below
      // retain their original deterministic order.
      planning_executor->run(task_count, [&](std::size_t index, std::size_t) {
        auto& task = parallel_row_tasks[index];
        const size_t row = task.row;
        sparse_eliminate_active_row(
            matrix[row], pivot_row, column, active_columns, task.factor,
            [&](size_t new_column, size_t new_position) {
              if (new_position > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("sparse row position exceeds 32-bit encoding");
              task.changes.push_back({new_column, new_position, RowChangeKind::New});
            },
            [&](size_t removed_column) {
              task.changes.push_back({removed_column, 0, RowChangeKind::Removed});
            },
            [&](size_t, size_t) {});
        std::size_t updated_rhs_nonzeros = task.updated_rhs_nonzeros;
        for (size_t rhs_column = 0; rhs_column < active_rhs_cols; ++rhs_column) {
          if (rhs(best_row, rhs_column) == T(0)) continue;
          const bool was_nonzero = rhs(row, rhs_column) != T(0);
          const T updated =
              rhs(row, rhs_column) - task.factor * rhs(best_row, rhs_column);
          rhs(row, rhs_column) = updated;
          const bool is_nonzero = updated != T(0);
          if (was_nonzero != is_nonzero) {
            if (is_nonzero)
              ++updated_rhs_nonzeros;
            else
              --updated_rhs_nonzeros;
          }
        }
        task.updated_rhs_nonzeros = updated_rhs_nonzeros;
      });
      for (std::size_t index = 0; index < task_count; ++index) {
        auto& task = parallel_row_tasks[index];
        const size_t row = task.row;
        const bool rhs_was_nonzero = rhs_row_nonzeros[row] != 0;
        for (const auto& change : task.changes) {
          const size_t changed_column = change.column;
          if (change.kind == RowChangeKind::New) {
            column_rows[changed_column].push_back(
                {task.row, static_cast<std::uint32_t>(change.position),
                 task.updated_version});
            ++column_active_nonzeros[changed_column];
          } else {
            if (column_active_nonzeros[changed_column] == 0)
              throw std::logic_error("Markowitz column count underflow");
            --column_active_nonzeros[changed_column];
          }
          mark_dirty(changed_column);
        }
        for (const auto& entry : matrix[row])
          mark_dirty(entry.column);
        row_versions[row] = task.updated_version;
        row_active_nonzeros[row] = matrix[row].size();
        rhs_row_nonzeros[row] = task.updated_rhs_nonzeros;
        const bool rhs_is_nonzero = rhs_row_nonzeros[row] != 0;
        if (rhs_was_nonzero != rhs_is_nonzero) {
          if (rhs_is_nonzero)
            ++residual_nonzero_rows;
          else
            --residual_nonzero_rows;
        }
      }
      if (statistics != nullptr) {
        ++statistics->parallel_row_batches;
        statistics->parallel_row_eliminations += task_count;
        statistics->row_eliminations += task_count;
      }
    }
    if (statistics != nullptr) {
      statistics->incidence_records_scanned += pivot_incidence_records;
      statistics->row_elimination_ms +=
          milliseconds(row_elimination_start, Clock::now());
    }
    active_columns[column] = false;
    column_active_nonzeros[column] = 0;
    // The completed column is never scanned again. Keep the echelon row for
    // back substitution, but release its now-unused reverse incidence list.
    std::vector<ColumnIncidence>().swap(column_rows[column]);
  };

  // The master-basis prefix is part of the public reduction convention and is
  // never reordered by the fill-reducing ansatz strategy.
  const auto prefix_start = Clock::now();
  for (size_t column = 0; column < required_prefix_cols; ++column) {
    const auto choices = choices_for_column(column, scan_workspaces.front());
    if (statistics != nullptr)
      statistics->incidence_records_scanned += choices.incidence_records_scanned;
    const PivotChoice choice =
        choices.matching.column != std::numeric_limits<size_t>::max()
            ? choices.matching
            : choices.fallback;
    if (choice.column == std::numeric_limits<size_t>::max()) {
      active_columns[column] = false;
      std::vector<ColumnIncidence>().swap(column_rows[column]);
      continue;
    }
    if (row_groups[choice.row] != column_groups[column]) ++cross_group_pivots;
    apply_pivot(choice);
  }
  if (statistics != nullptr) {
    statistics->relation_elimination_ms += milliseconds(prefix_start, Clock::now());
  }

  auto group_less = [](std::uint32_t lhs, std::uint32_t rhs) {
    const int lhs_count = std::popcount(lhs);
    const int rhs_count = std::popcount(rhs);
    return lhs_count != rhs_count ? lhs_count < rhs_count : lhs < rhs;
  };
  std::vector<std::uint32_t> ordered_groups;
  if (explicit_group_order.empty()) {
    ordered_groups.reserve(num_cols - required_prefix_cols);
    for (size_t column = required_prefix_cols; column < num_cols; ++column)
      ordered_groups.push_back(column_groups[column]);
    std::ranges::sort(ordered_groups, group_less);
    ordered_groups.erase(std::unique(ordered_groups.begin(), ordered_groups.end()),
                         ordered_groups.end());
  } else {
    ordered_groups.assign(explicit_group_order.begin(), explicit_group_order.end());
    std::vector<std::uint32_t> present;
    present.reserve(num_cols - required_prefix_cols);
    for (size_t column = required_prefix_cols; column < num_cols; ++column)
      present.push_back(column_groups[column]);
    std::ranges::sort(present);
    present.erase(std::unique(present.begin(), present.end()), present.end());
    auto supplied = ordered_groups;
    std::ranges::sort(supplied);
    if (std::ranges::adjacent_find(supplied) != supplied.end())
      throw std::runtime_error("explicit Markowitz group order contains duplicates");
    supplied.erase(std::unique(supplied.begin(), supplied.end()), supplied.end());
    if (supplied != present)
      throw std::runtime_error("explicit Markowitz group order is incomplete");
  }

  // Group membership is static. Build the index once so every group pass only
  // touches its own columns instead of rescanning the complete ansatz width.
  std::unordered_map<std::uint32_t, std::size_t> group_positions;
  group_positions.reserve(ordered_groups.size());
  std::vector<std::vector<size_t>> columns_by_group(ordered_groups.size());
  for (std::size_t index = 0; index < ordered_groups.size(); ++index)
    group_positions.emplace(ordered_groups[index], index);
  for (size_t column = required_prefix_cols; column < num_cols; ++column) {
    const auto found = group_positions.find(column_groups[column]);
    if (found == group_positions.end())
      throw std::logic_error("Markowitz column group is absent from group order");
    columns_by_group[found->second].push_back(column);
  }

  std::vector<std::uint32_t> score_versions(num_cols, 0);
  for (std::size_t group_index = 0; group_index < ordered_groups.size();
       ++group_index) {
    const auto group = ordered_groups[group_index];
    const auto& group_columns = columns_by_group[group_index];
    const auto group_start = Clock::now();
    current_group = group;
    dirty_list.clear();
    auto worse = [&](const PivotChoice& lhs, const PivotChoice& rhs) {
      return better(rhs, lhs);
    };
    using ChoiceQueueBase =
        std::priority_queue<PivotChoice, std::vector<PivotChoice>, decltype(worse)>;
    struct ChoiceQueue : ChoiceQueueBase {
      using ChoiceQueueBase::ChoiceQueueBase;
      auto& entries()
      {
        return this->c;
      }
      void rebuild()
      {
        std::make_heap(this->c.begin(), this->c.end(), this->comp);
      }
    };
    std::size_t active_group_columns = 0;
    ChoiceQueue matching_choices(worse);
    ChoiceQueue fallback_choices(worse);
    std::vector<size_t> refresh_columns;
    std::vector<ColumnChoices> refreshed_choices;
    auto refresh_batch = [&](std::span<const size_t> requested, bool dirty_only) {
      const auto refresh_start = Clock::now();
      refresh_columns.clear();
      std::size_t incidence_work = 0;
      for (const size_t column : requested) {
        if (dirty_only && !dirty_columns[column]) continue;
        dirty_columns[column] = false;
        if (!active_columns[column] || column_groups[column] != group) continue;
        ++score_versions[column];
        refresh_columns.push_back(column);
        incidence_work += column_rows[column].size();
      }
      refreshed_choices.resize(refresh_columns.size());
      const bool use_parallel = planning_threads > 1 &&
                                refresh_columns.size() >= 2 * planning_threads &&
                                incidence_work >= 32768;
      if (use_parallel) {
        while (scan_workspaces.size() < planning_threads)
          scan_workspaces.emplace_back(num_rows);
        if (planning_executor == nullptr) {
          planning_executor =
              std::make_unique<core::ParallelForExecutor>(planning_threads);
        }
      }
      const auto compute = [&](std::size_t index, std::size_t worker) {
        refreshed_choices[index] =
            choices_for_column(refresh_columns[index], scan_workspaces[worker]);
      };
      if (use_parallel)
        planning_executor->run(refresh_columns.size(), compute);
      else
        for (std::size_t index = 0; index < refresh_columns.size(); ++index)
          compute(index, 0);
      for (std::size_t index = 0; index < refresh_columns.size(); ++index) {
        const size_t column = refresh_columns[index];
        auto& choices = refreshed_choices[index];
        if (statistics != nullptr) {
          statistics->incidence_records_scanned += choices.incidence_records_scanned;
        }
        if (choices.matching.column != std::numeric_limits<size_t>::max()) {
          choices.matching.version = score_versions[column];
          matching_choices.push(choices.matching);
        }
        if (choices.fallback.column != std::numeric_limits<size_t>::max()) {
          choices.fallback.version = score_versions[column];
          fallback_choices.push(choices.fallback);
        }
      }
      const auto compact_queue = [&](ChoiceQueue& queue) {
        if (statistics != nullptr)
          statistics->maximum_choice_queue_size =
              std::max(statistics->maximum_choice_queue_size, queue.size());
        if constexpr (CompactChoiceQueues) {
          if (queue.size() <= std::max<std::size_t>(1024, 4 * active_group_columns))
            return;
          const auto removed = std::erase_if(queue.entries(), [&](const auto& choice) {
            // Current-version entries must survive even when their pivot value
            // vanished: discard_stale uses them to trigger a fresh column score.
            return !active_columns[choice.column] ||
                   score_versions[choice.column] != choice.version;
          });
          queue.rebuild();
          if (statistics != nullptr) {
            ++statistics->choice_queue_compactions;
            statistics->compacted_choice_entries += removed;
          }
        }
      };
      compact_queue(matching_choices);
      compact_queue(fallback_choices);
      if (statistics != nullptr) {
        statistics->score_refresh_ms += milliseconds(refresh_start, Clock::now());
        statistics->score_refresh_columns += refresh_columns.size();
        if (use_parallel) {
          ++statistics->parallel_refresh_batches;
          statistics->parallel_refresh_columns += refresh_columns.size();
        }
      }
    };
    auto refresh_column = [&](size_t column) {
      refresh_batch(std::span<const size_t>(&column, 1), false);
    };
    std::vector<size_t> initial_columns;
    initial_columns.reserve(group_columns.size());
    for (const size_t column : group_columns) {
      if (active_columns[column]) initial_columns.push_back(column);
    }
    active_group_columns = initial_columns.size();
    refresh_batch(initial_columns, false);
    // The compact planner supplies a single-pass interval selected from target-
    // support width and system size. Narrow supports retain the interval-16
    // policy because a wider batch can select a much larger replay tape.
    std::size_t refresh_interval =
        score_refresh_interval_hint == 0 ? 16 : score_refresh_interval_hint;
    if (complexity_order == ColumnComplexityOrder::Descending || num_cols < 4096) {
      refresh_interval = 8;
    }
    std::size_t pivots_since_refresh = 0;
    for (;;) {
      if (!elimination_needed()) break;
      if (pivots_since_refresh >= refresh_interval) {
        refresh_batch(dirty_list, true);
        dirty_list.clear();
        pivots_since_refresh = 0;
      }
      auto discard_stale = [&](auto& choices) {
        while (!choices.empty()) {
          const auto& choice = choices.top();
          if (active_columns[choice.column] &&
              score_versions[choice.column] == choice.version) {
            const T* value = used_rows[choice.row] ? nullptr : choice_value(choice);
            if (value != nullptr && *value != T(0)) break;
            const size_t column = choice.column;
            if (statistics != nullptr) ++statistics->stale_choice_pops;
            choices.pop();
            refresh_column(column);
            continue;
          }
          if (statistics != nullptr) ++statistics->stale_choice_pops;
          choices.pop();
        }
      };
      discard_stale(matching_choices);
      discard_stale(fallback_choices);
      if (matching_choices.empty() && fallback_choices.empty() && !dirty_list.empty()) {
        refresh_batch(dirty_list, true);
        dirty_list.clear();
        pivots_since_refresh = 0;
        discard_stale(matching_choices);
        discard_stale(fallback_choices);
      }
      if (matching_choices.empty() && fallback_choices.empty()) {
        for (const size_t column : group_columns) {
          if (!active_columns[column]) continue;
          auto& workspace = scan_workspaces.front();
          if (workspace.generation == std::numeric_limits<std::uint32_t>::max()) {
            std::ranges::fill(workspace.seen_rows,
                              std::numeric_limits<std::uint32_t>::max());
            workspace.generation = 0;
          }
          const std::uint32_t current_generation = workspace.generation++;
          for (auto& incidence : column_rows[column]) {
            const size_t row = incidence.row;
            if (workspace.seen_rows[row] == current_generation) continue;
            workspace.seen_rows[row] = current_generation;
            if (used_rows[row]) continue;
            if (find_incidence(column, incidence) == nullptr) continue;
            if (row_active_nonzeros[row] == 0)
              throw std::logic_error("Markowitz row count underflow");
            --row_active_nonzeros[row];
          }
          active_columns[column] = false;
          column_active_nonzeros[column] = 0;
          std::vector<ColumnIncidence>().swap(column_rows[column]);
        }
        break;
      }
      const PivotChoice choice =
          !matching_choices.empty() ? matching_choices.top() : fallback_choices.top();
      if (row_groups[choice.row] != column_groups[choice.column]) ++cross_group_pivots;
      apply_pivot(choice);
      --active_group_columns;
      ++pivots_since_refresh;
    }
    if (statistics != nullptr)
      statistics->relation_elimination_ms += milliseconds(group_start, Clock::now());
    if (!elimination_needed()) break;
  }

  for (size_t row = 0; row < num_rows; ++row) {
    if (!used_rows[row]) row_map.push_back(row);
  }
  return {std::move(solution_cols), residual_nonzero_rows == 0, std::move(row_map),
          cross_group_pivots};
}

} // namespace linalg
