#include "reduction/BlackBoxFeynman.hpp"
#include "reduction/BlockTriangular.hpp"
#include "reduction/KernelPlanning.hpp"
#include "reduction/ParameterEvaluation.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

void BlackBoxFeynman::build_pending_block_replay(
    const reduction::detail::KernelPublicationInput& input,
    const EvaluatedCoeffs<firefly::FFInt>& coeffs,
    const std::vector<firefly::FFInt>& values)
{
  using T = firefly::FFInt;
  using reduction::detail::checked_add_i64;
  const auto polynomial_terms =
      reduction::detail::reduction_polynomial_terms(cfg, numerator_strategy);
  const std::size_t num_params = reduction::detail::coefficient_parameter_count(cfg);
  const std::size_t num_bilinear_basis = 2 * num_params;
  const auto& indexed_basis_cols = input.basis_columns;
  const auto& indexed_target_cols = input.target_columns;
  const auto& indexed_ansatz_cols = input.ansatz_columns;
  const auto& ansatz_metadata = input.ansatz_metadata;
  const auto& row_sectors = input.row_sectors;
  const auto& row_groups =
      input.row_groups.empty() ? input.row_sectors : input.row_groups;
  const auto& target_sectors = input.target_sectors;
  const auto& ansatz_order = input.ansatz_order;
  const auto& solution_cols = input.solution_columns;
  const auto& elim_row_map = input.elimination_row_map;
  const std::size_t num_rows = row_sectors.size();
  const std::size_t num_basis_cols = indexed_basis_cols.size();
  const std::size_t num_targets = indexed_target_cols.size();
  if (target_sectors.size() != num_targets)
    throw std::logic_error("target sector schedule shape is inconsistent");
  const auto for_each_bilinear_weight = [&](const auto& term, auto&& emit) {
    reduction::detail::for_each_bilinear_weight(term, polynomial_terms, num_params,
                                                std::forward<decltype(emit)>(emit));
  };
  // Build sector execution blocks from the selected full-rank square. Each
  // block records an independent sparse tape; exact off-diagonal entries are
  // retained as couplings and applied during high-to-low back substitution.
  if (square_dim == 0) {
    throw std::logic_error("selected reduction square is empty");
  }
  if (num_basis_cols > square_dim) {
    throw std::logic_error("basis columns exceed the selected square");
  }

  auto sector_less = [](std::uint32_t lhs, std::uint32_t rhs) {
    const int lhs_count = std::popcount(lhs);
    const int rhs_count = std::popcount(rhs);
    return lhs_count != rhs_count ? lhs_count < rhs_count : lhs < rhs;
  };
  std::vector<std::uint32_t> nonzero_target_sectors;
  nonzero_target_sectors.reserve(target_sectors.size());
  for (const auto sector : target_sectors) {
    if (sector != 0) nonzero_target_sectors.push_back(sector);
  }
  std::ranges::sort(nonzero_target_sectors, sector_less);
  nonzero_target_sectors.erase(
      std::unique(nonzero_target_sectors.begin(), nonzero_target_sectors.end()),
      nonzero_target_sectors.end());
  kernel_statistics_.physical_target_rhs = num_targets;
  kernel_statistics_.target_sector_batches = nonzero_target_sectors.size();
  if (numerator_strategy == NumeratorReductionStrategy::Direct) {
    kernel_statistics_.block_layout = "jet-sector-scc";
  } else if (!input.ordered_groups.empty()) {
    kernel_statistics_.block_layout = "g-layer-sector-scc";
  } else {
    kernel_statistics_.block_layout = "seed-sector-scc";
  }

  std::unordered_map<std::uint32_t, std::size_t> group_rank;
  group_rank.reserve(input.ordered_groups.size());
  for (std::size_t rank = 0; rank < input.ordered_groups.size(); ++rank) {
    if (!group_rank.emplace(input.ordered_groups[rank], rank).second)
      throw std::logic_error("replay group order contains a duplicate group");
  }
  const auto replay_group_rank = [&](std::uint32_t group) {
    const auto found = group_rank.find(group);
    if (found == group_rank.end())
      throw std::logic_error("selected relation has no replay group rank");
    return found->second;
  };

  struct GroupPosition {
    std::uint32_t group;
    std::size_t original;
    std::size_t priority;
  };
  std::vector<GroupPosition> ansatz_rows;
  std::vector<GroupPosition> ansatz_columns;
  ansatz_rows.reserve(square_dim - num_basis_cols);
  ansatz_columns.reserve(square_dim - num_basis_cols);
  for (std::size_t index = num_basis_cols; index < square_dim; ++index) {
    const std::size_t column = solution_cols[index];
    if (column < num_basis_cols) {
      throw std::logic_error("selected ansatz columns overlap the basis prefix");
    }
    const std::size_t ordered = column - num_basis_cols;
    if (ordered >= ansatz_order.size()) {
      throw std::logic_error("selected ansatz column is out of range");
    }
    ansatz_rows.push_back(
        {row_groups[elim_row_map[index]], elim_row_map[index], index});
    ansatz_columns.push_back(
        {reduction::detail::ansatz_pivot_group(ansatz_metadata[ansatz_order[ordered]]),
         column, index});
  }
  std::vector<std::size_t> selected_rows;
  std::vector<std::size_t> selected_columns;
  selected_rows.reserve(square_dim);
  selected_columns.reserve(square_dim);
  for (std::size_t index = 0; index < num_basis_cols; ++index) {
    if (solution_cols[index] != index) {
      throw std::logic_error("selected square does not preserve the basis prefix");
    }
    selected_rows.push_back(elim_row_map[index]);
    selected_columns.push_back(index);
  }
  for (const auto& row : ansatz_rows)
    selected_rows.push_back(row.original);
  for (const auto& column : ansatz_columns)
    selected_columns.push_back(column.original);

  std::vector<block_triangular::Range> ranges;
  if (num_basis_cols != 0) ranges.push_back({0, num_basis_cols});
  // Relations are equations, so the primary block key is their planner group.
  // Pivot-row groups are deliberately not allowed to split one relation block;
  // the dependency SCCs below only coarsen these blocks when exact matrix
  // couplings require it.
  std::size_t block_begin = num_basis_cols;
  for (std::size_t index = 1; index < ansatz_rows.size(); ++index) {
    if (ansatz_columns[index - 1].group == ansatz_columns[index].group) continue;
    ranges.push_back({block_begin, num_basis_cols + index});
    block_begin = num_basis_cols + index;
  }
  if (!ansatz_rows.empty()) ranges.push_back({block_begin, square_dim});

  std::vector<ptrdiff_t> row_to_solver(num_rows, -1);
  for (std::size_t position = 0; position < square_dim; ++position) {
    row_to_solver[selected_rows[position]] = static_cast<ptrdiff_t>(position);
  }

  struct CoordinateTerm {
    std::uint32_t row;
    std::uint32_t column;
    std::uint32_t coefficient;
    bool is_top_lp_expression;
    std::int64_t weight;
  };
  std::vector<CoordinateTerm> coordinate_terms;
  auto append_coordinate_term = [&](std::size_t row, std::size_t column,
                                    std::size_t coefficient, bool is_top_lp_expression,
                                    std::int64_t weight) {
    if (weight == 0) return;
    if (row > std::numeric_limits<std::uint32_t>::max() ||
        column > std::numeric_limits<std::uint32_t>::max() ||
        coefficient > std::numeric_limits<std::uint32_t>::max() ||
        (!is_top_lp_expression &&
         coefficient > std::numeric_limits<std::uint8_t>::max())) {
      throw std::logic_error("block coordinate exceeds compact encoding");
    }
    coordinate_terms.push_back(
        {static_cast<std::uint32_t>(row), static_cast<std::uint32_t>(column),
         static_cast<std::uint32_t>(coefficient), is_top_lp_expression, weight});
  };
  auto append_column_terms = [&](std::size_t solver_column, const auto& column) {
    for (const auto& term : column) {
      const ptrdiff_t solver_row = row_to_solver[term.row];
      if (solver_row < 0) continue;
      if (term.coefficient_expression != std::numeric_limits<std::uint32_t>::max()) {
        append_coordinate_term(static_cast<std::size_t>(solver_row), solver_column,
                               term.coefficient_expression, true, term.coeff_int);
      } else {
        for_each_bilinear_weight(term, [&](std::size_t bb_idx, std::int64_t weight) {
          append_coordinate_term(static_cast<std::size_t>(solver_row), solver_column,
                                 bb_idx, false, weight);
        });
      }
    }
  };
  for (std::size_t position = 0; position < selected_columns.size(); ++position) {
    const std::size_t column = selected_columns[position];
    if (column < num_basis_cols) {
      append_column_terms(position, indexed_basis_cols.column(column));
    } else {
      const std::size_t ordered = column - num_basis_cols;
      append_column_terms(position, indexed_ansatz_cols.column(ansatz_order[ordered]));
    }
  }
  std::ranges::sort(coordinate_terms, [](const auto& lhs, const auto& rhs) {
    if (lhs.row != rhs.row) return lhs.row < rhs.row;
    if (lhs.column != rhs.column) return lhs.column < rhs.column;
    if (lhs.is_top_lp_expression != rhs.is_top_lp_expression)
      return lhs.is_top_lp_expression < rhs.is_top_lp_expression;
    return lhs.coefficient < rhs.coefficient;
  });
  std::vector<CoordinateTerm> merged_terms;
  merged_terms.reserve(coordinate_terms.size());
  for (const auto& term : coordinate_terms) {
    if (!merged_terms.empty() && merged_terms.back().row == term.row &&
        merged_terms.back().column == term.column &&
        merged_terms.back().is_top_lp_expression == term.is_top_lp_expression &&
        merged_terms.back().coefficient == term.coefficient) {
      merged_terms.back().weight =
          checked_add_i64(merged_terms.back().weight, term.weight);
    } else {
      merged_terms.push_back(term);
    }
  }
  std::erase_if(merged_terms, [](const auto& term) { return term.weight == 0; });
  coordinate_terms.clear();
  coordinate_terms.shrink_to_fit();

  std::vector<block_triangular::Coordinate> nonzero_coordinates;
  nonzero_coordinates.reserve(merged_terms.size());
  for (std::size_t begin = 0; begin < merged_terms.size();) {
    std::size_t end = begin + 1;
    while (end < merged_terms.size() &&
           merged_terms[end].row == merged_terms[begin].row &&
           merged_terms[end].column == merged_terms[begin].column) {
      ++end;
    }
    nonzero_coordinates.emplace_back(merged_terms[begin].row,
                                     merged_terms[begin].column);
    begin = end;
  }

  std::vector<block_triangular::Coordinate> ordering_coordinates = nonzero_coordinates;
  if (num_basis_cols != 0) {
    ordering_coordinates.reserve(ordering_coordinates.size() + ranges.size());
    for (const auto range : ranges | std::views::drop(1))
      ordering_coordinates.emplace_back(0, range.begin);
  }
  const auto ordering =
      block_triangular::order_by_dependencies(square_dim, ranges, ordering_coordinates);
  std::vector<std::size_t> old_to_new(square_dim);
  std::vector<std::size_t> reordered_rows(square_dim);
  std::vector<std::size_t> reordered_columns(square_dim);
  for (std::size_t position = 0; position < square_dim; ++position) {
    const std::size_t old_position = ordering.old_position_by_new[position];
    old_to_new[old_position] = position;
    reordered_rows[position] = selected_rows[old_position];
    reordered_columns[position] = selected_columns[old_position];
  }
  selected_rows = std::move(reordered_rows);
  selected_columns = std::move(reordered_columns);
  for (auto& term : merged_terms) {
    term.row = static_cast<std::uint32_t>(old_to_new[term.row]);
    term.column = static_cast<std::uint32_t>(old_to_new[term.column]);
  }
  ranges = ordering.ranges;
  // SCC membership is structural; within each final block restore the
  // fill-reducing row priority and deterministic sector order of columns.
  std::vector<std::size_t> row_priority(num_rows,
                                        std::numeric_limits<std::size_t>::max());
  std::vector<std::size_t> column_priority(num_basis_cols + ansatz_order.size(),
                                           std::numeric_limits<std::size_t>::max());
  std::vector<std::uint32_t> selected_column_group(num_basis_cols + ansatz_order.size(),
                                                   0);
  for (const auto& row : ansatz_rows) {
    row_priority[row.original] = row.priority;
  }
  for (const auto& column : ansatz_columns) {
    column_priority[column.original] = column.priority;
    selected_column_group[column.original] = column.group;
  }
  std::vector<std::size_t> row_old_to_new(square_dim);
  std::vector<std::size_t> column_old_to_new(square_dim);
  std::iota(row_old_to_new.begin(), row_old_to_new.end(), std::size_t{0});
  std::iota(column_old_to_new.begin(), column_old_to_new.end(), std::size_t{0});
  reordered_rows = selected_rows;
  reordered_columns = selected_columns;
  for (std::size_t block = num_basis_cols == 0 ? 0 : 1; block < ranges.size();
       ++block) {
    std::vector<std::size_t> positions(ranges[block].end - ranges[block].begin);
    std::iota(positions.begin(), positions.end(), ranges[block].begin);
    std::ranges::sort(positions, [&](std::size_t lhs, std::size_t rhs) {
      const auto lhs_priority = row_priority[selected_rows[lhs]];
      const auto rhs_priority = row_priority[selected_rows[rhs]];
      return lhs_priority != rhs_priority ? lhs_priority < rhs_priority
                                          : selected_rows[lhs] < selected_rows[rhs];
    });
    for (std::size_t offset = 0; offset < positions.size(); ++offset) {
      const std::size_t new_position = ranges[block].begin + offset;
      const std::size_t old_position = positions[offset];
      row_old_to_new[old_position] = new_position;
      reordered_rows[new_position] = selected_rows[old_position];
    }
    std::iota(positions.begin(), positions.end(), ranges[block].begin);
    std::ranges::sort(positions, [&](std::size_t lhs, std::size_t rhs) {
      const auto lhs_group = selected_column_group[selected_columns[lhs]];
      const auto rhs_group = selected_column_group[selected_columns[rhs]];
      if (lhs_group != rhs_group) {
        if (input.ordered_groups.empty()) return sector_less(lhs_group, rhs_group);
        return replay_group_rank(lhs_group) < replay_group_rank(rhs_group);
      }
      const auto lhs_priority = column_priority[selected_columns[lhs]];
      const auto rhs_priority = column_priority[selected_columns[rhs]];
      return lhs_priority != rhs_priority
                 ? lhs_priority < rhs_priority
                 : selected_columns[lhs] < selected_columns[rhs];
    });
    for (std::size_t offset = 0; offset < positions.size(); ++offset) {
      const std::size_t new_position = ranges[block].begin + offset;
      const std::size_t old_position = positions[offset];
      column_old_to_new[old_position] = new_position;
      reordered_columns[new_position] = selected_columns[old_position];
    }
  }
  selected_rows = std::move(reordered_rows);
  selected_columns = std::move(reordered_columns);
  for (auto& term : merged_terms) {
    term.row = static_cast<std::uint32_t>(row_old_to_new[term.row]);
    term.column = static_cast<std::uint32_t>(column_old_to_new[term.column]);
  }
  std::ranges::sort(merged_terms, [](const auto& lhs, const auto& rhs) {
    if (lhs.row != rhs.row) return lhs.row < rhs.row;
    if (lhs.column != rhs.column) return lhs.column < rhs.column;
    if (lhs.is_top_lp_expression != rhs.is_top_lp_expression)
      return lhs.is_top_lp_expression < rhs.is_top_lp_expression;
    return lhs.coefficient < rhs.coefficient;
  });
  for (std::size_t position = 0; position < square_dim; ++position)
    row_to_solver[selected_rows[position]] = static_cast<ptrdiff_t>(position);

  nonzero_coordinates.clear();
  nonzero_coordinates.reserve(merged_terms.size());
  for (std::size_t begin = 0; begin < merged_terms.size();) {
    std::size_t end = begin + 1;
    while (end < merged_terms.size() &&
           merged_terms[end].row == merged_terms[begin].row &&
           merged_terms[end].column == merged_terms[begin].column) {
      ++end;
    }
    nonzero_coordinates.emplace_back(merged_terms[begin].row,
                                     merged_terms[begin].column);
    begin = end;
  }
  const auto checked_ranges = block_triangular::coarsen_to_upper_triangular(
      square_dim, ranges, nonzero_coordinates);
  if (checked_ranges != ranges) {
    throw std::logic_error("SCC block ordering left a lower-triangular dependency");
  }
  std::vector<block_triangular::Coordinate>().swap(nonzero_coordinates);

  std::vector<std::size_t> block_of_position(square_dim);
  std::vector<BlockReplayProgram> programs;
  programs.reserve(ranges.size());
  for (std::size_t block = 0; block < ranges.size(); ++block) {
    const auto range = ranges[block];
    if (range.end - range.begin > std::numeric_limits<std::uint32_t>::max())
      throw std::logic_error("execution block exceeds 32-bit dimensions");
    for (std::size_t position = range.begin; position < range.end; ++position)
      block_of_position[position] = block;
    BlockReplayProgram program;
    program.row_begin = static_cast<std::uint32_t>(range.begin);
    program.dimension = static_cast<std::uint32_t>(range.end - range.begin);
    programs.push_back(std::move(program));
  }

  std::vector<T> bilinear_basis(num_bilinear_basis);
  bilinear_basis[0] = T(1);
  for (std::size_t parameter = 0; parameter < cfg.kinematic_parameters.size();
       ++parameter) {
    bilinear_basis[parameter + 1] =
        reduction::detail::evaluate_kinematic_parameter(cfg, values, parameter);
  }
  bilinear_basis[num_params] = coeffs.minus_half_d;
  for (std::size_t parameter = 1; parameter < num_params; ++parameter) {
    bilinear_basis[num_params + parameter] =
        coeffs.minus_half_d * bilinear_basis[parameter];
  }

  std::vector<linalg::SparseMatrix<T>> diagonal_matrices;
  diagonal_matrices.reserve(ranges.size());
  std::vector<std::uint32_t> next_initial_slot(ranges.size(), 0);
  for (const auto range : ranges)
    diagonal_matrices.emplace_back(range.end - range.begin);

  for (std::size_t begin = 0; begin < merged_terms.size();) {
    std::size_t end = begin + 1;
    while (end < merged_terms.size() &&
           merged_terms[end].row == merged_terms[begin].row &&
           merged_terms[end].column == merged_terms[begin].column) {
      ++end;
    }
    const std::size_t row = merged_terms[begin].row;
    const std::size_t column = merged_terms[begin].column;
    const std::size_t row_block = block_of_position[row];
    const std::size_t column_block = block_of_position[column];
    if (row_block > column_block)
      throw std::logic_error("coarsened execution blocks are not upper triangular");
    T value(0);
    for (std::size_t index = begin; index < end; ++index) {
      const auto weight = merged_terms[index].weight;
      const T coefficient =
          merged_terms[index].is_top_lp_expression
              ? coeffs.top_lp_coefficients[merged_terms[index].coefficient]
              : bilinear_basis[merged_terms[index].coefficient];
      value = value + (weight >= 0 ? T(weight) : T(0) - T(-weight)) * coefficient;
    }
    if (row_block == column_block) {
      if (value == T(0)) {
        throw std::runtime_error(
            "block matrix coordinate vanishes at the planning anchor");
      }
      auto& program = programs[row_block];
      const auto slot = next_initial_slot[row_block]++;
      const std::size_t local_row = row - program.row_begin;
      const std::size_t local_column = column - program.row_begin;
      diagonal_matrices[row_block][local_row].emplace_back(local_column, slot, value);
      for (std::size_t index = begin; index < end; ++index) {
        if (merged_terms[index].is_top_lp_expression) {
          program.top_lp_skeleton_M.push_back(
              {slot, merged_terms[index].coefficient, merged_terms[index].weight});
        } else {
          program.skeleton_M.push_back(
              {slot, static_cast<std::uint8_t>(merged_terms[index].coefficient),
               merged_terms[index].weight});
        }
      }
    } else {
      auto& program = programs[column_block];
      if (program.couplings.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::logic_error("block coupling count exceeds compact encoding");
      const auto coupling_index = static_cast<std::uint32_t>(program.couplings.size());
      program.couplings.push_back(
          {static_cast<std::uint32_t>(row), static_cast<std::uint32_t>(column)});
      for (std::size_t index = begin; index < end; ++index) {
        if (merged_terms[index].is_top_lp_expression) {
          program.top_lp_skeleton_couplings.push_back({coupling_index,
                                                       merged_terms[index].coefficient,
                                                       merged_terms[index].weight});
        } else {
          program.skeleton_couplings.push_back(
              {coupling_index,
               static_cast<std::uint8_t>(merged_terms[index].coefficient),
               merged_terms[index].weight});
        }
      }
    }
    begin = end;
  }
  merged_terms.clear();
  merged_terms.shrink_to_fit();

  std::vector<std::uint8_t> probe_rhs_support(square_dim * num_targets, 0);
  for (std::size_t target = 0; target < indexed_target_cols.size(); ++target) {
    for (const auto& term : indexed_target_cols.column(target)) {
      if (sector_less(target_sectors[target], row_sectors[term.row]))
        throw std::logic_error("target term is above its scheduled seed sector");
      const ptrdiff_t solver_row = row_to_solver[term.row];
      if (solver_row >= 0) {
        const std::size_t slot =
            static_cast<std::size_t>(solver_row) * num_targets + target;
        probe_rhs_support[slot] = 1;
      }
    }
  }

  std::vector<std::uint32_t> solution_row_by_column(square_dim);
  std::vector<linalg::TapeResult> recorded_blocks(programs.size());
  std::vector<std::vector<linalg::Instruction>> operation_tapes(programs.size());
  for (std::size_t reverse = programs.size(); reverse > 0; --reverse) {
    const std::size_t block = reverse - 1;
    auto& program = programs[block];
    const std::size_t dimension = program.dimension;
    std::vector<T> local_rhs(dimension * num_targets, T(0));
    const std::size_t rhs_begin =
        static_cast<std::size_t>(program.row_begin) * num_targets;
    std::vector<std::uint8_t> local_rhs_support(local_rhs.size());
    std::copy_n(probe_rhs_support.begin() + static_cast<ptrdiff_t>(rhs_begin),
                local_rhs_support.size(), local_rhs_support.begin());
    linalg::TapeResult recorded;
    const bool basis_identity_block =
        block == 0 && program.row_begin == 0 && dimension == num_basis_cols;
    if (basis_identity_block) {
      for (std::size_t row = 0; row < dimension; ++row) {
        const auto& entries = diagonal_matrices[block][row];
        if (entries.size() != 1 || entries.front().column != row ||
            entries.front().value != T(1)) {
          throw std::logic_error("basis execution block is not the identity");
        }
      }
      recorded.perm.resize(dimension);
      std::iota(recorded.perm.begin(), recorded.perm.end(), std::size_t{0});
      recorded.rhs_slot_count = static_cast<std::uint32_t>(local_rhs.size());
    } else {
      recorded = linalg::record_sparse_tape(diagonal_matrices[block], local_rhs,
                                            num_targets, local_rhs_support, false);
    }
    std::copy(local_rhs_support.begin(), local_rhs_support.end(),
              probe_rhs_support.begin() + static_cast<ptrdiff_t>(rhs_begin));
    for (std::size_t column = 0; column < dimension; ++column) {
      solution_row_by_column[static_cast<std::size_t>(program.row_begin) + column] =
          static_cast<std::uint32_t>(static_cast<std::size_t>(program.row_begin) +
                                     recorded.perm[column]);
    }
    for (std::size_t coupling_index = 0; coupling_index < program.couplings.size();
         ++coupling_index) {
      auto& coupling = program.couplings[coupling_index];
      coupling.source_row = solution_row_by_column[coupling.source_row];
      const std::size_t destination =
          static_cast<std::size_t>(coupling.destination_row) * num_targets;
      const std::size_t source =
          static_cast<std::size_t>(coupling.source_row) * num_targets;
      for (std::size_t target = 0; target < num_targets; ++target) {
        if (probe_rhs_support[source + target] != 0)
          probe_rhs_support[destination + target] = 1;
      }
    }
    operation_tapes[block] = std::move(recorded.tape);
    recorded_blocks[block] = std::move(recorded);
    linalg::SparseMatrix<T>().swap(diagonal_matrices[block]);
  }
  std::vector<std::uint32_t> solution_row_by_original_column(
      num_basis_cols + ansatz_order.size(), std::numeric_limits<std::uint32_t>::max());
  for (std::size_t position = 0; position < selected_columns.size(); ++position) {
    const std::size_t original_column = selected_columns[position];
    if (original_column >= solution_row_by_original_column.size())
      throw std::logic_error("selected replay column is out of range");
    solution_row_by_original_column[original_column] = solution_row_by_column[position];
  }
  std::vector<std::uint32_t> reference_solution_row_by_column(solution_cols.size());
  for (std::size_t reference_column = 0; reference_column < solution_cols.size();
       ++reference_column) {
    const std::size_t original_column = solution_cols[reference_column];
    if (original_column >= solution_row_by_original_column.size() ||
        solution_row_by_original_column[original_column] ==
            std::numeric_limits<std::uint32_t>::max()) {
      throw std::logic_error("block replay lost a reference solution column");
    }
    reference_solution_row_by_column[reference_column] =
        solution_row_by_original_column[original_column];
  }
  for (std::size_t basis = 0; basis < num_basis_cols; ++basis)
    solution_row_by_column[basis] = solution_row_by_original_column[basis];
  std::vector<std::uint8_t>().swap(probe_rhs_support);

  std::vector<AnsatzEntry> block_rhs_skeleton;
  std::vector<TopLpRhsEntry> block_top_lp_rhs_skeleton;
  auto append_rhs_weight = [&](std::size_t flat_idx, std::size_t bb_idx,
                               std::int64_t weight) {
    if (weight == 0) return;
    if (flat_idx > std::numeric_limits<std::uint32_t>::max())
      throw std::logic_error("block RHS slot exceeds compact encoding");
    block_rhs_skeleton.push_back({static_cast<std::uint32_t>(flat_idx),
                                  static_cast<std::uint8_t>(bb_idx), weight});
  };
  auto append_top_lp_rhs_weight = [&](std::size_t flat_idx, std::uint32_t expression,
                                      std::int64_t weight) {
    if (weight == 0) return;
    if (flat_idx > std::numeric_limits<std::uint32_t>::max() ||
        expression >= top_lp_expression_programs.size()) {
      throw std::logic_error("block top-LP RHS coordinate exceeds encoding");
    }
    block_top_lp_rhs_skeleton.push_back(
        {static_cast<std::uint32_t>(flat_idx), expression, weight});
  };
  for (std::size_t target = 0; target < indexed_target_cols.size(); ++target) {
    for (const auto& term : indexed_target_cols.column(target)) {
      const ptrdiff_t solver_row = row_to_solver[term.row];
      if (solver_row < 0) continue;
      const std::size_t flat_idx =
          static_cast<std::size_t>(solver_row) * num_targets + target;
      if (term.coefficient_expression != std::numeric_limits<std::uint32_t>::max()) {
        append_top_lp_rhs_weight(flat_idx, term.coefficient_expression, term.coeff_int);
      } else {
        for_each_bilinear_weight(term, [&](std::size_t bb_idx, std::int64_t weight) {
          append_rhs_weight(flat_idx, bb_idx, weight);
        });
      }
    }
  }
  std::ranges::sort(block_rhs_skeleton, [](const auto& lhs, const auto& rhs) {
    if (lhs.flat_idx != rhs.flat_idx) return lhs.flat_idx < rhs.flat_idx;
    return lhs.bb_idx < rhs.bb_idx;
  });
  std::vector<AnsatzEntry> merged_rhs;
  merged_rhs.reserve(block_rhs_skeleton.size());
  for (const auto& entry : block_rhs_skeleton) {
    if (!merged_rhs.empty() && merged_rhs.back().flat_idx == entry.flat_idx &&
        merged_rhs.back().bb_idx == entry.bb_idx) {
      merged_rhs.back().weight =
          checked_add_i64(merged_rhs.back().weight, entry.weight);
    } else {
      merged_rhs.push_back(entry);
    }
  }
  std::erase_if(merged_rhs, [](const auto& entry) { return entry.weight == 0; });
  std::ranges::sort(block_top_lp_rhs_skeleton, [](const auto& lhs, const auto& rhs) {
    if (lhs.flat_idx != rhs.flat_idx) return lhs.flat_idx < rhs.flat_idx;
    return lhs.expression < rhs.expression;
  });
  std::vector<TopLpRhsEntry> merged_top_lp_rhs;
  merged_top_lp_rhs.reserve(block_top_lp_rhs_skeleton.size());
  for (const auto& entry : block_top_lp_rhs_skeleton) {
    if (!merged_top_lp_rhs.empty() &&
        merged_top_lp_rhs.back().flat_idx == entry.flat_idx &&
        merged_top_lp_rhs.back().expression == entry.expression) {
      merged_top_lp_rhs.back().weight =
          checked_add_i64(merged_top_lp_rhs.back().weight, entry.weight);
    } else {
      merged_top_lp_rhs.push_back(entry);
    }
  }
  std::erase_if(merged_top_lp_rhs, [](const auto& entry) { return entry.weight == 0; });

  pending_block_replay = std::make_unique<PendingBlockReplay>();
  pending_block_replay->programs = std::move(programs);
  pending_block_replay->recorded_blocks = std::move(recorded_blocks);
  pending_block_replay->operation_tapes = std::move(operation_tapes);
  pending_block_replay->solution_row_by_column = std::move(solution_row_by_column);
  pending_block_replay->reference_solution_row_by_column =
      std::move(reference_solution_row_by_column);
  pending_block_replay->block_of_position = std::move(block_of_position);
  pending_block_replay->rhs_skeleton = std::move(merged_rhs);
  pending_block_replay->top_lp_rhs_skeleton = std::move(merged_top_lp_rhs);
}
