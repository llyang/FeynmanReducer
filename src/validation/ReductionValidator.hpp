#pragma once

#include "core/Config.hpp"

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace validation {

struct ValidationFiles {
  std::filesystem::path result;
  std::filesystem::path basis;
  std::filesystem::path kira_result;
  std::filesystem::path kira_basis;
  std::map<std::string, ExactRationalConstant> numerics;
};

struct ValidationReport {
  std::size_t target_count = 0;
  std::size_t master_count = 0;
  std::size_t coefficient_count = 0;
  std::size_t skipped_identity_count = 0;
  std::size_t difference_count = 0;
  std::vector<std::string> diagnostics;

  [[nodiscard]] bool passed() const noexcept
  {
    return difference_count == 0;
  }
};

[[nodiscard]] ValidationReport validate_reduction(const ValidationFiles& files,
                                                  std::size_t diagnostic_limit = 20);

} // namespace validation
