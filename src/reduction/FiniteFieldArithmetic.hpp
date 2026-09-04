#pragma once

#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <flint/ulong_extras.h>

namespace finite_field {

__extension__ typedef unsigned __int128 WideProduct;

// Shared planning/replay arithmetic. Values remain in Montgomery form while
// sparse elimination or the matrix/RHS tape is evaluated; encode/decode are
// used only at external and loader boundaries.
class MontgomeryArithmetic {
public:
  explicit MontgomeryArithmetic(std::uint64_t prime)
  {
    reset(prime);
  }

  void reset(std::uint64_t prime)
  {
    if (prime == 0) throw std::runtime_error("finite-field prime must be nonzero");
    if (prime_ == prime) return;
    if (prime < 3 || (prime & 1U) == 0 || prime >= (std::uint64_t{1} << 63U)) {
      throw std::runtime_error(
          "Montgomery replay requires an odd prime smaller than 2^63");
    }
    prime_ = prime;
    negative_prime_inverse_ = 0 - n_binvert(prime);
    one_ = static_cast<std::uint64_t>((static_cast<WideProduct>(1) << 64U) % prime_);
    r_squared_ = standard_multiply(one_, one_);
    r_cubed_ = standard_multiply(r_squared_, one_);
  }

  [[nodiscard]] std::uint64_t prime() const noexcept
  {
    return prime_;
  }

  [[nodiscard]] std::uint64_t one() const noexcept
  {
    return one_;
  }

  [[nodiscard]] std::uint64_t encode(std::uint64_t value) const noexcept
  {
    return montgomery_multiply(value, r_squared_);
  }

  [[nodiscard]] std::uint64_t decode(std::uint64_t value) const noexcept
  {
    return montgomery_multiply(value, 1);
  }

  [[nodiscard]] std::uint64_t add(std::uint64_t lhs, std::uint64_t rhs) const noexcept
  {
    return n_addmod(lhs, rhs, prime_);
  }

  [[nodiscard]] std::uint64_t multiply(std::uint64_t lhs,
                                       std::uint64_t rhs) const noexcept
  {
    if (lhs == 0 || rhs == 0) return 0;
    if (lhs == one_) return rhs;
    if (rhs == one_) return lhs;
    return montgomery_multiply(lhs, rhs);
  }

  [[nodiscard]] std::uint64_t subtract(std::uint64_t lhs,
                                       std::uint64_t rhs) const noexcept
  {
    return n_submod(lhs, rhs, prime_);
  }

  [[nodiscard]] std::uint64_t subtract_multiply(std::uint64_t lhs, std::uint64_t factor,
                                                std::uint64_t rhs) const noexcept
  {
    if (factor == 0 || rhs == 0) return lhs;
    if (factor == one_) return n_submod(lhs, rhs, prime_);
    if (rhs == one_) return n_submod(lhs, factor, prime_);
    return n_submod(lhs, multiply(factor, rhs), prime_);
  }

  [[nodiscard]] bool inverse(std::uint64_t value, std::uint64_t& result) const noexcept
  {
    if (value == 0) return false;
    if (value == one_ || value == prime_ - one_) {
      result = value;
      return true;
    }
    ulong inverse_value = 0;
    if (n_gcdinv(&inverse_value, value, prime_) != 1) return false;
    // value = xR. inv(value) = x^-1 R^-1, so one Montgomery
    // multiplication by R^3 returns the encoded inverse x^-1 R.
    result = montgomery_multiply(inverse_value, r_cubed_);
    return true;
  }

private:
  [[nodiscard]] std::uint64_t standard_multiply(std::uint64_t lhs,
                                                std::uint64_t rhs) const noexcept
  {
    const auto product = static_cast<WideProduct>(lhs) * static_cast<WideProduct>(rhs);
    return static_cast<std::uint64_t>(product % prime_);
  }

  [[nodiscard]] std::uint64_t montgomery_multiply(std::uint64_t lhs,
                                                  std::uint64_t rhs) const noexcept
  {
    const auto product = static_cast<WideProduct>(lhs) * static_cast<WideProduct>(rhs);
    const auto low = static_cast<std::uint64_t>(product);
    const auto high = static_cast<std::uint64_t>(product >> 64U);
    const auto quotient = low * negative_prime_inverse_;
    const auto correction =
        static_cast<WideProduct>(quotient) * static_cast<WideProduct>(prime_);
    const auto correction_low = static_cast<std::uint64_t>(correction);
    const auto correction_high = static_cast<std::uint64_t>(correction >> 64U);
    const auto carry = static_cast<std::uint64_t>(low + correction_low < low);
    std::uint64_t result = high + correction_high + carry;
    if (result >= prime_) result -= prime_;
    return result;
  }

  std::uint64_t prime_ = 0;
  std::uint64_t negative_prime_inverse_ = 0;
  std::uint64_t one_ = 0;
  std::uint64_t r_squared_ = 0;
  std::uint64_t r_cubed_ = 0;
};

// Planning uses the same scalar field as replay, but needs value semantics for
// the generic sparse-elimination templates. Like FireFly's FFInt, this type has
// one process-wide active prime; callers must not change it during elimination.
class MontgomeryFieldElement {
public:
  MontgomeryFieldElement() = default;

  template <std::integral Integer> MontgomeryFieldElement(Integer value)
  {
    std::uint64_t residue = 0;
    if constexpr (std::is_signed_v<Integer>) {
      if (value < 0) {
        const auto magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1;
        const auto reduced = magnitude % arithmetic_.prime();
        residue = reduced == 0 ? 0 : arithmetic_.prime() - reduced;
      } else {
        residue = static_cast<std::uint64_t>(value) % arithmetic_.prime();
      }
    } else {
      residue = static_cast<std::uint64_t>(value) % arithmetic_.prime();
    }
    value_ = arithmetic_.encode(residue);
  }

  static void set_prime(std::uint64_t prime)
  {
    arithmetic_.reset(prime);
  }

  [[nodiscard]] static MontgomeryFieldElement from_residue(std::uint64_t value)
  {
    MontgomeryFieldElement result;
    result.value_ = arithmetic_.encode(value);
    return result;
  }

  [[nodiscard]] std::uint64_t residue() const noexcept
  {
    return arithmetic_.decode(value_);
  }

  friend bool operator==(MontgomeryFieldElement lhs,
                         MontgomeryFieldElement rhs) noexcept
  {
    return lhs.value_ == rhs.value_;
  }

  friend MontgomeryFieldElement operator+(MontgomeryFieldElement lhs,
                                          MontgomeryFieldElement rhs) noexcept
  {
    return from_encoded(arithmetic_.add(lhs.value_, rhs.value_));
  }

  friend MontgomeryFieldElement operator-(MontgomeryFieldElement lhs,
                                          MontgomeryFieldElement rhs) noexcept
  {
    return from_encoded(arithmetic_.subtract(lhs.value_, rhs.value_));
  }

  friend MontgomeryFieldElement operator*(MontgomeryFieldElement lhs,
                                          MontgomeryFieldElement rhs) noexcept
  {
    return from_encoded(arithmetic_.multiply(lhs.value_, rhs.value_));
  }

  friend MontgomeryFieldElement operator/(MontgomeryFieldElement lhs,
                                          MontgomeryFieldElement rhs)
  {
    std::uint64_t inverse = 0;
    if (!arithmetic_.inverse(rhs.value_, inverse))
      throw std::runtime_error("cannot invert zero finite-field element");
    return from_encoded(arithmetic_.multiply(lhs.value_, inverse));
  }

private:
  [[nodiscard]] static MontgomeryFieldElement from_encoded(std::uint64_t value) noexcept
  {
    MontgomeryFieldElement result;
    result.value_ = value;
    return result;
  }

  inline static MontgomeryArithmetic arithmetic_{3};
  std::uint64_t value_ = 0;
};
static_assert(sizeof(MontgomeryFieldElement) == sizeof(std::uint64_t));

} // namespace finite_field
