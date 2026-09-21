#pragma once

#include "core/Config.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace compiler::detail {

struct ParsedCombinationTerm {
  Integral integral;
  std::string coefficient;
};

struct ParsedCombination {
  ReductionOutput output;
  std::vector<ParsedCombinationTerm> terms;
};

[[nodiscard]] std::vector<ParsedCombination>
parse_combinations(const std::filesystem::path& path, const Config& config);

} // namespace compiler::detail
