#pragma once

#include "FiniteField.hpp"
#include "core/Config.hpp"

#include <cstdint>

namespace basis {

struct LpNormalizationData {
  unsigned positive_sum = 0;
  unsigned pinch_count = 0;
};

[[nodiscard]] LpNormalizationData lp_normalization_data(const Config& config,
                                                        const Integral& integral);

// Converts a native conventional coefficient of `basis` in `target` to the
// quotient-normalized LP convention.  Besides \bar kappa_basis/\bar kappa_target,
// this includes the native-to-LP sign (-1)^(A_target-A_basis).
[[nodiscard]] std::uint64_t lp_normalization_and_sign_ratio(const PrimeField& field,
                                                            const Config& config,
                                                            const Integral& target,
                                                            const Integral& basis,
                                                            std::uint64_t dimension);

} // namespace basis

namespace quotient {
using basis::lp_normalization_and_sign_ratio;
using basis::lp_normalization_data;
using basis::LpNormalizationData;
} // namespace quotient
