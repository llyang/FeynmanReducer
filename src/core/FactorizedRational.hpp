#pragma once

#include "core/FlintRational.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace factorized_detail {
class Builder;
}

// Completed exact product. Only the residual numerator stays expanded.
// Factors are primitive irreducible polynomials with signed multiplicities;
// equal factors and common numerator/denominator factors have been cancelled.
// Incremental construction is restricted to the internal Builder.
class FactorizedRational {
public:
  struct Factor {
    FlintRational polynomial;
    std::int64_t power;
  };
  explicit FactorizedRational(std::shared_ptr<const FlintRationalContext> context);
  explicit FactorizedRational(FlintRational residual);
  [[nodiscard]] bool is_zero() const
  {
    return numerator_.is_zero();
  }
  [[nodiscard]] const auto& context() const
  {
    return numerator_.context();
  }
  [[nodiscard]] const auto& numerator() const
  {
    return numerator_;
  }
  [[nodiscard]] const auto& scalar() const
  {
    return scalar_;
  }
  [[nodiscard]] const auto& factors() const
  {
    return factors_;
  }
  [[nodiscard]] std::string to_string() const;
  // Explicit, uncached expansion for exact comparisons and compatibility only.
  [[nodiscard]] FlintRational expand() const;
  [[nodiscard]] std::optional<ulong> evaluate_mod(std::span<const ulong> values,
                                                  ulong prime) const;
  [[nodiscard]] std::optional<std::size_t> mixed_denominator_factor() const;

private:
  friend class factorized_detail::Builder;
  void set_residual(FlintRational residual);
  void set_numerator(FlintRational numerator);
  void set_scalar(FlintRational scalar);
  void add_scanned_factor(const FlintRational& factor);
  void add_irreducible_factor(FlintRational polynomial, std::int64_t power);
  void normalize();
  void factor_polynomial(const fmpz_mpoly_struct* polynomial, std::int64_t sign);
  FlintRational numerator_;
  FlintRational scalar_;
  std::vector<Factor> factors_;
};
