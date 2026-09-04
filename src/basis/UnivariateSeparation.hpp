#pragma once

#include "FiniteField.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>

namespace basis {

// Reconstructs the lowest-total-degree univariate rational function whose
// denominator is monic in its highest nonzero degree. Holdout samples are not
// used in the solve.
[[nodiscard]] std::optional<RationalFunction>
interpolate_rational_monic(const PrimeField& field,
                           std::span<const std::uint64_t> arguments,
                           std::span<const std::uint64_t> values,
                           std::size_t maximum_total_degree, std::size_t holdout_count);

// Reconstructs a rational function with a prescribed numerator degree and a
// monic denominator of the prescribed degree.  The unused tail is checked as
// an exact holdout certificate.  A failed or non-unique solve returns nullopt.
[[nodiscard]] std::optional<RationalFunction> interpolate_rational_fixed_degrees(
    const PrimeField& field, std::span<const std::uint64_t> arguments,
    std::span<const std::uint64_t> values, std::size_t numerator_degree,
    std::size_t denominator_degree, std::size_t holdout_count);

struct RationalInterpolationDiagnostics {
  std::size_t attempts = 0;
  std::size_t samples_fed = 0;
  std::size_t numerator_degree = 0;
  std::size_t denominator_degree = 0;
  std::string failure;
};

// Fast exact-delta path. The caller must set firefly::FFInt to field.prime()
// before entering this function and must not change that global prime while
// concurrent interpolations are running. Holdout samples are never fed to the
// interpolator and are used only for candidate validation.
[[nodiscard]] std::optional<RationalFunction> interpolate_rational_thiele_monic(
    const PrimeField& field, std::span<const std::uint64_t> arguments,
    std::span<const std::uint64_t> values, std::size_t holdout_count,
    RationalInterpolationDiagnostics* diagnostics = nullptr);

// Returns the largest denominator degree not shared by every nonconstant
// polynomial. Constant polynomials represent specialization cancellations and
// are ignored.
[[nodiscard]] std::size_t
moving_polynomial_degree(const PrimeField& field,
                         std::span<const FieldVector> polynomials);

[[nodiscard]] bool
stable_polynomial_signatures(const PrimeField& field,
                             std::span<const FieldVector> polynomials);

} // namespace basis

namespace quotient {
using basis::interpolate_rational_fixed_degrees;
using basis::interpolate_rational_monic;
using basis::interpolate_rational_thiele_monic;
using basis::moving_polynomial_degree;
using basis::RationalInterpolationDiagnostics;
using basis::stable_polynomial_signatures;
} // namespace quotient
