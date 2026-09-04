#pragma once

#include "core/Config.hpp"

#include <cstddef>
#include <cstdint>

/// Returns whether a polynomial monomial remains after all variables outside
/// the sector are set to zero.
[[nodiscard]] inline bool polynomial_term_survives_sector(const PolynomialTerm& term,
                                                          std::uint32_t sector) noexcept
{
  for (std::size_t variable = 0; variable < term.powers.size(); ++variable) {
    if (((sector >> variable) & 1U) == 0 && term.powers[variable] != 0) return false;
  }
  return true;
}
