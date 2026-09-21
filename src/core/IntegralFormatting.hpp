#pragma once

#include "core/Config.hpp"

#include <string>
#include <vector>

[[nodiscard]] std::string format_mathematica_integral(const std::string& header,
                                                      const std::vector<int>& indices);

[[nodiscard]] std::string format_mathematica_integral(const std::string& header,
                                                      const Integral& integral);
