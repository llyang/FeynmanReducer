#pragma once

#include <string>
#include <vector>

[[nodiscard]] std::string format_mathematica_integral(const std::string& header,
                                                      const std::vector<int>& indices);
