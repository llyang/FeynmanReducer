#pragma once

#include "core/Config.hpp"
#include "core/FactorizedRational.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

enum class DSeparationStatus { NotChecked, Passed, Failed, Skipped };
struct DSeparationReport {
  DSeparationStatus status = DSeparationStatus::NotChecked;
  std::size_t failed_coefficients = 0;
  std::string witness;
};

struct ReductionResult {
  std::string integral_header = "F";
  std::vector<std::string> parameters;
  std::map<std::string, ExactRationalConstant> numerics;
  std::vector<Integral> basis;
  // Source targets retained for compatibility and diagnostics. Coefficients
  // are indexed by outputs, not by this vector, when combinations are used.
  std::vector<Integral> targets;
  std::vector<ReductionOutput> outputs;
  std::shared_ptr<const FlintRationalContext> context;
  // output-major, then basis-major.
  std::vector<FactorizedRational> coefficients;
  DSeparationReport d_separation;
};
