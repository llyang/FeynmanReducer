#pragma once

#include "core/Config.hpp"

#include <span>
#include <string>
#include <vector>

namespace reduction::detail {

// Source terms for derivatives of a set of possible master integrals. Target
// ids refer to the Config used to build the plan and remain valid while new
// targets are only appended.
struct DifferentialEquationSourcePlan {
  std::vector<Integral> differentiated_integrals;
  std::vector<std::string> parameters;
  // parameter-major, then differentiated-integral-major.
  std::vector<std::vector<ReductionRequestTerm>> terms;
};

// Converts legacy one-target-per-output input to explicit requests when
// needed, then appends all unique d+2 source integrals to config.targets.
[[nodiscard]] DifferentialEquationSourcePlan prepare_differential_equation_sources(
    Config& config, std::span<const Integral> differentiated_integrals);

// Appends public derivative requests for final_basis. Every final integral
// must have been included in the source plan.
void append_differential_equation_requests(Config& config,
                                           const DifferentialEquationSourcePlan& source,
                                           std::span<const Integral> final_basis);

// Convenience entry for a basis that is already final.
void materialize_differential_equations(Config& config);

} // namespace reduction::detail
