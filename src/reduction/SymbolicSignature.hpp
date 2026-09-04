#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <vector>

using SymbolicSignatureTerm = std::array<std::int64_t, 3>;
using SymbolicSignature = std::vector<SymbolicSignatureTerm>;

inline void make_primitive_symbolic_signature(SymbolicSignature& signature)
{
  std::uint64_t common = 0;
  for (const auto& term : signature) {
    const std::int64_t weight = term[2];
    const std::uint64_t magnitude = weight < 0
                                        ? static_cast<std::uint64_t>(-(weight + 1)) + 1
                                        : static_cast<std::uint64_t>(weight);
    common = std::gcd(common, magnitude);
  }
  if (common > 1) {
    constexpr std::uint64_t int64_min_magnitude = std::uint64_t{1} << 63U;
    for (auto& term : signature) {
      if (common == int64_min_magnitude) {
        term[2] = term[2] == std::numeric_limits<std::int64_t>::min() ? -1 : 0;
      } else {
        term[2] /= static_cast<std::int64_t>(common);
      }
    }
  }

  if (!signature.empty() && signature.front()[2] < 0) {
    const bool can_negate = std::ranges::none_of(signature, [](const auto& term) {
      return term[2] == std::numeric_limits<std::int64_t>::min();
    });
    if (can_negate) {
      for (auto& term : signature)
        term[2] = -term[2];
    }
  }
}
