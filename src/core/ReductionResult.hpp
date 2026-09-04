#pragma once

#include "core/Config.hpp"
#include "core/FlintRational.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct ReductionResult {
  std::string integral_header = "F";
  std::vector<std::string> parameters;
  std::map<std::string, ExactRationalConstant> numerics;
  std::vector<Integral> basis;
  std::vector<Integral> targets;
  std::shared_ptr<const FlintRationalContext> context;
  // target-major, then basis-major.
  std::vector<FlintRational> coefficients;
};
