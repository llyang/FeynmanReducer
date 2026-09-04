#pragma once

// fmpz.h only declares GMP conversion functions if gmp.h was seen first.
// clang-format off
#include <gmp.h>
#include <flint/fmpz.h>
// clang-format on

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace compiler::detail {

inline std::int64_t checked_int64(const fmpz_t value, const std::string& label)
{
  if (fmpz_fits_si(value) == 0) {
    throw std::runtime_error(label + " exceeds the signed machine range");
  }
  const slong raw = fmpz_get_si(value);
  if constexpr (sizeof(slong) > sizeof(std::int64_t)) {
    if (raw < std::numeric_limits<std::int64_t>::min() ||
        raw > std::numeric_limits<std::int64_t>::max()) {
      throw std::runtime_error(label + " exceeds int64");
    }
  }
  return static_cast<std::int64_t>(raw);
}

} // namespace compiler::detail
