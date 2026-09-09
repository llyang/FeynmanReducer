#pragma once

#include "core/FactorizedRational.hpp"

#include <utility>

namespace factorized_detail {
// Internal assembly boundary: an unfinished product has no query interface.
// finish() publishes a normalized value and consumes the builder.
class Builder {
public:
  explicit Builder(std::shared_ptr<const FlintRationalContext> context)
      : value_(std::move(context))
  {}
  explicit Builder(FlintRational residual) : value_(residual.context())
  {
    value_.set_residual(std::move(residual));
  }
  explicit Builder(FactorizedRational value) : value_(std::move(value)) {}
  Builder(const Builder&) = delete;
  Builder& operator=(const Builder&) = delete;
  Builder(Builder&&) = default;
  Builder& operator=(Builder&&) = default;

  void set_numerator(FlintRational numerator)
  {
    value_.set_numerator(std::move(numerator));
  }
  void set_scalar(FlintRational scalar)
  {
    value_.set_scalar(std::move(scalar));
  }
  void add_scanned_factor(const FlintRational& factor)
  {
    value_.add_scanned_factor(factor);
  }
  // Trusted input only: FLINT factorization bases, their homogenizations, or a
  // single variable. Arbitrary polynomials must go through add_scanned_factor.
  void add_irreducible_factor(FlintRational polynomial, std::int64_t power)
  {
    value_.add_irreducible_factor(std::move(polynomial), power);
  }
  [[nodiscard]] FactorizedRational finish() &&
  {
    value_.normalize();
    return std::move(value_);
  }

private:
  FactorizedRational value_;
};
} // namespace factorized_detail
