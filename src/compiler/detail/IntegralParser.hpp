#pragma once

#include "core/Config.hpp"

#include <filesystem>
#include <vector>

namespace compiler::detail {

[[nodiscard]] std::vector<Integral> parse_integrals(const std::filesystem::path& path,
                                                    unsigned expected_size);

} // namespace compiler::detail
