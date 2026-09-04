#include "reduction/BlackBoxFeynman.hpp"
#include "reduction/KernelPlanning.hpp"
#include "reduction/ParameterEvaluation.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

void BlackBoxFeynman::build_reference_evaluation_plan(
    const reduction::detail::KernelPublicationInput& input)
{
  using reduction::detail::checked_add_i64;
  const auto polynomial_terms =
      reduction::detail::reduction_polynomial_terms(cfg, numerator_strategy);
  const std::size_t num_params = reduction::detail::coefficient_parameter_count(cfg);
  const auto& indexed_basis_cols = input.basis_columns;
  const auto& indexed_target_cols = input.target_columns;
  const auto& indexed_ansatz_cols = input.ansatz_columns;
  const auto& row_sectors = input.row_sectors;
  const auto& ansatz_order = input.ansatz_order;
  const auto& solution_cols = input.solution_columns;
  const auto& elim_row_map = input.elimination_row_map;
  const std::size_t num_rows = row_sectors.size();
  const std::size_t num_basis_cols = indexed_basis_cols.size();
  const auto for_each_bilinear_weight = [&](const auto& term, auto&& emit) {
    reduction::detail::for_each_bilinear_weight(term, polynomial_terms, num_params,
                                                std::forward<decltype(emit)>(emit));
  };

  auto reference_plan = std::make_unique<ReferenceEvaluationPlan>();
  reference_plan->dimension = static_cast<std::uint32_t>(square_dim);
  reference_plan->closure_row_count = static_cast<std::uint32_t>(num_rows);
  reference_plan->reference_row_by_closure_row.assign(
      num_rows, std::numeric_limits<std::uint32_t>::max());
  for (std::size_t row = 0; row < square_dim; ++row)
    reference_plan->reference_row_by_closure_row[elim_row_map[row]] =
        static_cast<std::uint32_t>(row);

  auto append_matrix = [&](auto& output, std::size_t row, std::size_t column,
                           std::size_t bb_idx, std::int64_t weight) {
    if (weight == 0) return;
    if (row > std::numeric_limits<std::uint32_t>::max() ||
        column > std::numeric_limits<std::uint32_t>::max() ||
        bb_idx > std::numeric_limits<std::uint8_t>::max()) {
      throw std::runtime_error("reference evaluator matrix coordinate overflow");
    }
    output.push_back({static_cast<std::uint32_t>(row),
                      static_cast<std::uint32_t>(column),
                      static_cast<std::uint8_t>(bb_idx), weight});
  };
  auto append_reference_column = [&](std::size_t reference_column, const auto& column) {
    for (const auto& term : column) {
      if (term.coefficient_expression != std::numeric_limits<std::uint32_t>::max()) {
        if (term.coefficient_expression >= top_lp_expression_programs.size())
          throw std::logic_error(
              "reference evaluator top-LP expression is out of range");
        reference_plan->closure_top_lp_matrix_terms.push_back(
            {term.row, static_cast<std::uint32_t>(reference_column),
             term.coefficient_expression, term.coeff_int});
        continue;
      }
      auto append_weight = [&](std::size_t bb_idx, std::int64_t weight) {
        append_matrix(reference_plan->closure_matrix_terms, term.row, reference_column,
                      bb_idx, weight);
      };
      for_each_bilinear_weight(term, append_weight);
    }
  };
  for (std::size_t reference_column = 0; reference_column < solution_cols.size();
       ++reference_column) {
    const std::size_t original_column = solution_cols[reference_column];
    if (original_column < num_basis_cols) {
      append_reference_column(reference_column,
                              indexed_basis_cols.column(original_column));
    } else {
      const std::size_t ordered = original_column - num_basis_cols;
      append_reference_column(reference_column,
                              indexed_ansatz_cols.column(ansatz_order[ordered]));
    }
  }
  auto merge_matrix_terms = [](auto& terms) {
    std::ranges::sort(terms, [](const auto& lhs, const auto& rhs) {
      if (lhs.row != rhs.row) return lhs.row < rhs.row;
      if (lhs.column != rhs.column) return lhs.column < rhs.column;
      return lhs.bb_idx < rhs.bb_idx;
    });
    std::vector<ReferenceMatrixTerm> merged;
    merged.reserve(terms.size());
    for (const auto& term : terms) {
      if (!merged.empty() && merged.back().row == term.row &&
          merged.back().column == term.column && merged.back().bb_idx == term.bb_idx) {
        merged.back().weight = checked_add_i64(merged.back().weight, term.weight);
      } else {
        merged.push_back(term);
      }
    }
    std::erase_if(merged, [](const auto& term) { return term.weight == 0; });
    terms = std::move(merged);
  };
  merge_matrix_terms(reference_plan->closure_matrix_terms);
  std::ranges::sort(reference_plan->closure_top_lp_matrix_terms,
                    [](const auto& lhs, const auto& rhs) {
                      if (lhs.row != rhs.row) return lhs.row < rhs.row;
                      if (lhs.column != rhs.column) return lhs.column < rhs.column;
                      return lhs.expression < rhs.expression;
                    });
  std::vector<ReferenceTopLpMatrixTerm> merged_expression_matrix;
  merged_expression_matrix.reserve(reference_plan->closure_top_lp_matrix_terms.size());
  for (const auto& term : reference_plan->closure_top_lp_matrix_terms) {
    if (!merged_expression_matrix.empty() &&
        merged_expression_matrix.back().row == term.row &&
        merged_expression_matrix.back().column == term.column &&
        merged_expression_matrix.back().expression == term.expression) {
      merged_expression_matrix.back().weight =
          checked_add_i64(merged_expression_matrix.back().weight, term.weight);
    } else {
      merged_expression_matrix.push_back(term);
    }
  }
  std::erase_if(merged_expression_matrix,
                [](const auto& term) { return term.weight == 0; });
  reference_plan->closure_top_lp_matrix_terms = std::move(merged_expression_matrix);

  // Pre-merge the two coefficient streams by matrix coordinate. Validation
  // points now only evaluate the listed ranges and never sort temporary
  // coordinates in the preparation hot path.
  const auto coordinate_of = [](const auto& term) {
    return std::pair(term.row, term.column);
  };
  std::size_t matrix_position = 0;
  std::size_t top_lp_position = 0;
  while (matrix_position < reference_plan->closure_matrix_terms.size() ||
         top_lp_position < reference_plan->closure_top_lp_matrix_terms.size()) {
    const auto matrix_coordinate =
        matrix_position < reference_plan->closure_matrix_terms.size()
            ? coordinate_of(reference_plan->closure_matrix_terms[matrix_position])
            : std::pair(std::numeric_limits<std::uint32_t>::max(),
                        std::numeric_limits<std::uint32_t>::max());
    const auto top_lp_coordinate =
        top_lp_position < reference_plan->closure_top_lp_matrix_terms.size()
            ? coordinate_of(
                  reference_plan->closure_top_lp_matrix_terms[top_lp_position])
            : std::pair(std::numeric_limits<std::uint32_t>::max(),
                        std::numeric_limits<std::uint32_t>::max());
    const auto coordinate = std::min(matrix_coordinate, top_lp_coordinate);
    const std::size_t matrix_begin = matrix_position;
    while (matrix_position < reference_plan->closure_matrix_terms.size() &&
           coordinate_of(reference_plan->closure_matrix_terms[matrix_position]) ==
               coordinate) {
      ++matrix_position;
    }
    const std::size_t top_lp_begin = top_lp_position;
    while (
        top_lp_position < reference_plan->closure_top_lp_matrix_terms.size() &&
        coordinate_of(reference_plan->closure_top_lp_matrix_terms[top_lp_position]) ==
            coordinate) {
      ++top_lp_position;
    }
    if (matrix_position > std::numeric_limits<std::uint32_t>::max() ||
        top_lp_position > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("reference evaluator term range exceeds 32-bit ids");
    }
    reference_plan->matrix_coordinates.push_back(
        {coordinate.first, coordinate.second, static_cast<std::uint32_t>(matrix_begin),
         static_cast<std::uint32_t>(matrix_position),
         static_cast<std::uint32_t>(top_lp_begin),
         static_cast<std::uint32_t>(top_lp_position)});
  }

  auto append_rhs = [&](auto& output, std::size_t row, std::size_t target,
                        std::size_t bb_idx, std::int64_t weight) {
    if (weight == 0) return;
    if (row > std::numeric_limits<std::uint32_t>::max() ||
        target > std::numeric_limits<std::uint32_t>::max() ||
        bb_idx > std::numeric_limits<std::uint8_t>::max()) {
      throw std::runtime_error("reference evaluator RHS coordinate overflow");
    }
    output.push_back({static_cast<std::uint32_t>(row),
                      static_cast<std::uint32_t>(target),
                      static_cast<std::uint8_t>(bb_idx), weight});
  };
  auto append_top_lp_rhs = [&](auto& output, std::size_t row, std::size_t target,
                               std::uint32_t expression, std::int64_t weight) {
    if (weight == 0) return;
    if (row > std::numeric_limits<std::uint32_t>::max() ||
        target > std::numeric_limits<std::uint32_t>::max() ||
        expression >= top_lp_expression_programs.size()) {
      throw std::runtime_error("reference evaluator top-LP RHS coordinate overflow");
    }
    output.push_back({static_cast<std::uint32_t>(row),
                      static_cast<std::uint32_t>(target), expression, weight});
  };
  for (std::size_t target = 0; target < indexed_target_cols.size(); ++target) {
    for (const auto& term : indexed_target_cols.column(target)) {
      auto append_weight = [&](std::size_t bb_idx, std::int64_t weight) {
        append_rhs(reference_plan->closure_rhs_terms, term.row, target, bb_idx, weight);
      };
      if (term.coefficient_expression != std::numeric_limits<std::uint32_t>::max()) {
        append_top_lp_rhs(reference_plan->closure_top_lp_rhs_terms, term.row, target,
                          term.coefficient_expression, term.coeff_int);
      } else {
        for_each_bilinear_weight(term, append_weight);
      }
    }
  }
  auto merge_rhs_terms = [](auto& terms) {
    std::ranges::sort(terms, [](const auto& lhs, const auto& rhs) {
      if (lhs.row != rhs.row) return lhs.row < rhs.row;
      if (lhs.target != rhs.target) return lhs.target < rhs.target;
      return lhs.bb_idx < rhs.bb_idx;
    });
    std::vector<ReferenceRhsTerm> merged;
    merged.reserve(terms.size());
    for (const auto& term : terms) {
      if (!merged.empty() && merged.back().row == term.row &&
          merged.back().target == term.target && merged.back().bb_idx == term.bb_idx) {
        merged.back().weight = checked_add_i64(merged.back().weight, term.weight);
      } else {
        merged.push_back(term);
      }
    }
    std::erase_if(merged, [](const auto& term) { return term.weight == 0; });
    terms = std::move(merged);
  };
  merge_rhs_terms(reference_plan->closure_rhs_terms);
  auto merge_top_lp_rhs_terms = [](auto& terms) {
    std::ranges::sort(terms, [](const auto& lhs, const auto& rhs) {
      if (lhs.row != rhs.row) return lhs.row < rhs.row;
      if (lhs.target != rhs.target) return lhs.target < rhs.target;
      return lhs.expression < rhs.expression;
    });
    std::vector<ReferenceTopLpRhsTerm> merged;
    merged.reserve(terms.size());
    for (const auto& term : terms) {
      if (!merged.empty() && merged.back().row == term.row &&
          merged.back().target == term.target &&
          merged.back().expression == term.expression) {
        merged.back().weight = checked_add_i64(merged.back().weight, term.weight);
      } else {
        merged.push_back(term);
      }
    }
    std::erase_if(merged, [](const auto& term) { return term.weight == 0; });
    terms = std::move(merged);
  };
  merge_top_lp_rhs_terms(reference_plan->closure_top_lp_rhs_terms);
  reference_evaluation_plan = std::move(reference_plan);
}
