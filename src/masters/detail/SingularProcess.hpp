#pragma once

#include <string>

namespace masters::detail {

struct SingularProcessOutput {
  int status = 0;
  std::string stdout_text;
  std::string stderr_text;
};

[[nodiscard]] SingularProcessOutput
run_singular_process(const std::string& singular_path, const std::string& script);

} // namespace masters::detail
