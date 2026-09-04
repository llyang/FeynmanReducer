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
  // Top-LP columns are aligned with Config::targets and used as RHS columns.
  std::vector<std::vector<Monomial>> columns;
  std::vector<TopLpCoefficientExpression> expressions;
  unsigned maximum_g_shift = 0;
};

/// Compiles exact boundary projections for negative-index targets.
[[nodiscard]] TopLpTargetPlan compile_top_lp_target_plan(const Config& config);
