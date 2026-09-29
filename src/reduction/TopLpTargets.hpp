#pragma once

#include "core/Config.hpp"
#include "reduction/Monomial.hpp"

#include <compare>
#include <cstdint>
#include <vector>

struct TopLpCoefficientAtom {
  std::int64_t weight = 0;
  std::uint16_t falling_degree = 0;
  std::vector<std::uint32_t> polynomial_factors;

  auto operator<=>(const TopLpCoefficientAtom&) const = default;
};

struct TopLpCoefficientExpression {
  std::vector<TopLpCoefficientAtom> atoms;

  auto operator<=>(const TopLpCoefficientExpression&) const = default;
};

struct TopLpTargetPlan {
  // Projected columns share one coefficient-expression registry. Basis columns
  // enter the matrix; target columns enter the right-hand side.
  std::vector<std::vector<Monomial>> basis_columns;
  std::vector<std::vector<Monomial>> columns;
  std::vector<TopLpCoefficientExpression> expressions;
  unsigned maximum_g_shift = 0;
  bool projected = false;
};

/// Compiles exact boundary projections for the basis and targets whenever any
/// integral needs the extended LP representation.
[[nodiscard]] TopLpTargetPlan compile_top_lp_target_plan(const Config& config);
