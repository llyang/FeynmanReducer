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

// Internal cross-basis experiment. Forward rows express new masters in the old
// basis; reverse rows express old masters in the new basis. Common identity rows
// may be omitted. Both inverse identities and C_old = C_new S are checked exactly.
struct BasisChangeValidationFiles {
  std::filesystem::path old_result, new_result, old_basis, new_basis;
  std::filesystem::path new_in_old, old_in_new;
};
[[nodiscard]] ValidationReport
validate_basis_change(const BasisChangeValidationFiles& files,
                      std::size_t diagnostic_limit = 20);

} // namespace validation
