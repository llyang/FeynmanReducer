#pragma once

#include "reduction/EquationGenerator.hpp"
#include "reduction/ReductionOptions.hpp"

#include <firefly/FFInt.hpp>

#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace reduction::detail {

[[nodiscard]] inline std::int64_t checked_mul_i64(std::int64_t lhs, std::int64_t rhs)
{
  std::int64_t result = 0;
  if (__builtin_mul_overflow(lhs, rhs, &result))
    throw std::overflow_error("symbolic kernel coefficient exceeds int64");
  return result;
}

[[nodiscard]] inline std::int64_t checked_add_i64(std::int64_t lhs, std::int64_t rhs)
{
  std::int64_t result = 0;
  if (__builtin_add_overflow(lhs, rhs, &result))
    throw std::overflow_error("symbolic kernel coefficient exceeds int64");
  return result;
}

[[nodiscard]] inline std::uint32_t positive_power_sum_u32(std::span<const int> powers,
                                                          const char* overflow_message)
{
  std::uint64_t sum = 0;
  for (const int power : powers) {
    if (power <= 0) continue;
    sum += static_cast<std::uint64_t>(power);
    if (sum > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error(overflow_message);
  }
  return static_cast<std::uint32_t>(sum);
}

[[nodiscard]] inline std::span<const PolynomialTerm>
reduction_polynomial_terms(const Config& config)
{
  return config.polynomial_terms;
}

[[nodiscard]] inline std::span<const PolynomialTerm>
reduction_polynomial_terms(const Config& config, NumeratorReductionStrategy strategy)
{
  return strategy == NumeratorReductionStrategy::Direct
             ? std::span<const PolynomialTerm>(config.extended_lp.polynomial_terms)
             : std::span<const PolynomialTerm>(config.polynomial_terms);
}

struct IndexedTerm {
  std::uint32_t row;
  std::uint32_t polynomial_term_index;
  std::uint32_t coefficient_expression;
  bool with_polynomial_coefficient;
  std::int64_t coeff_int;
  std::int64_t coeff_minus_half_d;
};
static_assert(sizeof(IndexedTerm) == 32);

struct IndexedColumns {
  std::vector<IndexedTerm> terms;
  std::vector<std::size_t> offsets{0};

  [[nodiscard]] std::size_t size() const noexcept
  {
    return offsets.size() - 1;
  }

  [[nodiscard]] std::span<const IndexedTerm> column(std::size_t index) const
  {
    return std::span<const IndexedTerm>(terms).subspan(
        offsets[index], offsets[index + 1] - offsets[index]);
  }
};

struct AnsatzColumnMeta {
  std::uint32_t grid_index;
  std::uint32_t seed_sector;
  std::uint16_t derivative_index;
  AnsatzFamily family;
  std::uint32_t pivot_group = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t seed_dot_excess = 0;
};

[[nodiscard]] inline std::uint32_t ansatz_pivot_group(const AnsatzColumnMeta& meta)
{
  return meta.pivot_group == std::numeric_limits<std::uint32_t>::max()
             ? meta.seed_sector
             : meta.pivot_group;
}

struct KernelPublicationInput {
  IndexedColumns basis_columns;
  IndexedColumns target_columns;
  IndexedColumns ansatz_columns;
  std::vector<AnsatzColumnMeta> ansatz_metadata;
  std::vector<std::uint32_t> row_sectors;
  std::vector<std::uint32_t> row_groups;
  std::vector<std::uint32_t> ordered_groups;
  std::vector<std::uint32_t> target_sectors;
  std::vector<std::size_t> ansatz_order;
  std::vector<std::size_t> solution_columns;
  std::vector<std::size_t> elimination_row_map;
  std::vector<firefly::FFInt> polynomial_values;
};

struct CompactPhaseTimings {
  double provisional_build_ms = 0.0;
  double provisional_elimination_ms = 0.0;
  double provisional_relation_elimination_ms = 0.0;
  double provisional_back_substitution_ms = 0.0;
  double provisional_score_refresh_ms = 0.0;
  double provisional_row_elimination_ms = 0.0;
  double support_selection_ms = 0.0;
  double compact_build_ms = 0.0;
  double compact_elimination_ms = 0.0;
  double total_ms = 0.0;
};

struct CompactSelection {
  bool closed = false;
  std::vector<std::size_t> residual_rows;
  std::vector<std::vector<std::size_t>> residual_rhs_support;
  std::vector<std::size_t> ansatz_order;
  std::vector<std::size_t> solution_columns;
  std::vector<std::size_t> elimination_row_map;
  std::size_t provisional_dimension = 0;
  std::size_t cross_group_pivots = 0;
  std::size_t provisional_rhs_columns = 0;
  std::size_t provisional_relation_pivots = 0;
  std::size_t provisional_score_refresh_columns = 0;
  std::size_t provisional_incidence_records_scanned = 0;
  std::size_t provisional_parallel_refresh_batches = 0;
  std::size_t provisional_parallel_refresh_columns = 0;
  std::size_t provisional_stale_choice_pops = 0;
  std::size_t provisional_row_eliminations = 0;
  std::size_t provisional_parallel_row_batches = 0;
  std::size_t provisional_parallel_row_eliminations = 0;
  bool provisional_rhs_fallback = false;
  CompactPhaseTimings timings;
};

[[nodiscard]] CompactSelection plan_compact_kernel(
    const IndexedColumns& basis_columns, const IndexedColumns& target_columns,
    const IndexedColumns& ansatz_columns,
    std::span<const AnsatzColumnMeta> ansatz_metadata,
    std::span<const std::uint32_t> row_sectors,
    std::span<const firefly::FFInt> polynomial_values,
    std::span<const firefly::FFInt> top_lp_coefficients,
    const firefly::FFInt& minus_half_d, std::span<const std::uint32_t> row_groups = {},
    std::span<const std::uint32_t> ordered_groups = {},
    AnsatzDotOrdering dot_ordering = AnsatzDotOrdering::Markowitz,
    std::size_t planning_threads = 1);

template <typename Emit>
void for_each_bilinear_weight(const IndexedTerm& term,
                              std::span<const PolynomialTerm> polynomial_terms,
                              std::size_t parameter_count, Emit&& emit)
{
  if (term.coefficient_expression != std::numeric_limits<std::uint32_t>::max()) {
    throw std::logic_error(
        "top-LP coefficient expression cannot be expanded as a bilinear term");
  }
  if (term.with_polynomial_coefficient) {
    const auto& weights = polynomial_terms[term.polynomial_term_index].weights;
    for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
      emit(parameter, checked_mul_i64(term.coeff_int, weights[parameter]));
      emit(parameter_count + parameter,
           checked_mul_i64(term.coeff_minus_half_d, weights[parameter]));
    }
  } else {
    emit(0, term.coeff_int);
    emit(parameter_count, term.coeff_minus_half_d);
  }
}

} // namespace reduction::detail
