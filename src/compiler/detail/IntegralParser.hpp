#pragma once

#include "core/Config.hpp"

#include <filesystem>
#include <string_view>
#include <vector>

namespace compiler::detail {

[[nodiscard]] std::vector<Integral> parse_integrals(const std::filesystem::path& path,
                                                    unsigned expected_size);

[[nodiscard]] Integral parse_integral_expression(std::string_view expression,
                                                 unsigned expected_size);

} // namespace compiler::detail
