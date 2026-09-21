#include "core/IntegralFormatting.hpp"

std::string format_mathematica_integral(const std::string& header,
                                        const std::vector<int>& indices)
{
  std::string result = header + "[";
  for (std::size_t index = 0; index < indices.size(); ++index) {
    if (index != 0) result += ", ";
    result += std::to_string(indices[index]);
  }
  result += "]";
  return result;
}

std::string format_mathematica_integral(const std::string& header,
                                        const Integral& integral)
{
  if (integral.dimension_shift == 0)
    return format_mathematica_integral(header, integral.indices);
  std::string result = header + "[" + std::to_string(integral.dimension_shift) + ",{";
  for (std::size_t index = 0; index < integral.indices.size(); ++index) {
    if (index != 0) result += ',';
    result += std::to_string(integral.indices[index]);
  }
  result += "}]";
  return result;
}
