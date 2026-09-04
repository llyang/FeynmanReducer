#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>

namespace masters::detail {

inline bool monomial_preferred(std::span<const int> lhs, std::span<const int> rhs)
{
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("monomial exponent sizes differ");
  }
  const auto total = [](std::span<const int> powers) {
    return std::accumulate(powers.begin(), powers.end(), std::int64_t{0});
  };
  const auto maximum = [](std::span<const int> powers) {
    return powers.empty() ? 0 : *std::ranges::max_element(powers);
  };
  const auto lhs_total = total(lhs);
  const auto rhs_total = total(rhs);
  if (lhs_total != rhs_total) {
    return lhs_total < rhs_total;
  }
  const int lhs_maximum = maximum(lhs);
  const int rhs_maximum = maximum(rhs);
  if (lhs_maximum != rhs_maximum) {
    return lhs_maximum < rhs_maximum;
  }
  return std::lexicographical_compare(rhs.begin(), rhs.end(), lhs.begin(), lhs.end());
}

} // namespace masters::detail
