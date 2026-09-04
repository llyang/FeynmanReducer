#pragma once

#include <flint/fmpq.h>

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace compiler::detail {

class Rational {
public:
  Rational()
  {
    fmpq_init(value_);
  }
  explicit Rational(long value)
  {
    fmpq_init(value_);
    fmpq_set_si(value_, value, 1);
  }
  explicit Rational(const std::string& value)
  {
    fmpq_init(value_);
    if (fmpq_set_str(value_, value.c_str(), 10) != 0) {
      fmpq_clear(value_);
      throw std::runtime_error("invalid rational literal: " + value);
    }
    fmpq_canonicalise(value_);
  }
  Rational(const Rational& other)
  {
    fmpq_init(value_);
    fmpq_set(value_, other.value_);
  }
  Rational(Rational&& other) noexcept
  {
    fmpq_init(value_);
    fmpq_swap(value_, other.value_);
  }
  Rational& operator=(const Rational& other)
  {
    if (this != &other) {
      fmpq_set(value_, other.value_);
    }
    return *this;
  }
  Rational& operator=(Rational&& other) noexcept
  {
    if (this != &other) {
      fmpq_swap(value_, other.value_);
    }
    return *this;
  }
  ~Rational()
  {
    fmpq_clear(value_);
  }

  [[nodiscard]] bool is_zero() const
  {
    return fmpq_is_zero(value_) != 0;
  }
  [[nodiscard]] bool is_one() const
  {
    return fmpq_is_one(value_) != 0;
  }
  [[nodiscard]] bool is_integer() const
  {
    return fmpz_is_one(fmpq_denref(value_)) != 0;
  }
  [[nodiscard]] const fmpq* raw() const
  {
    return value_;
  }

private:
  fmpq_t value_;

  friend Rational operator-(Rational);
  friend Rational operator+(const Rational&, const Rational&);
  friend Rational operator-(const Rational&, const Rational&);
  friend Rational operator*(const Rational&, const Rational&);
  friend Rational operator/(const Rational&, const Rational&);
};

using Exponents = std::vector<int>;

struct Polynomial {
  std::size_t symbol_count = 0;
  std::map<Exponents, Rational> terms;
};

[[nodiscard]] Polynomial zero_poly(std::size_t count);
void add_term(Polynomial& poly, Exponents powers, Rational coefficient);
[[nodiscard]] Polynomial constant_poly(std::size_t count, Rational value);
[[nodiscard]] Polynomial one_poly(std::size_t count);
[[nodiscard]] Polynomial symbol_poly(std::size_t count, std::size_t index);
[[nodiscard]] Polynomial operator+(const Polynomial&, const Polynomial&);
[[nodiscard]] Polynomial operator-(const Polynomial&);
[[nodiscard]] Polynomial operator-(const Polynomial&, const Polynomial&);
[[nodiscard]] Polynomial operator*(const Polynomial&, const Polynomial&);
[[nodiscard]] Polynomial scale(const Polynomial&, const Rational&);
[[nodiscard]] Polynomial power(Polynomial base, int exponent);
[[nodiscard]] bool is_constant(const Polynomial&);
[[nodiscard]] Rational constant_value(const Polynomial&);
[[nodiscard]] Polynomial derivative(const Polynomial&, std::size_t variable);
[[nodiscard]] Polynomial
substitute(const Polynomial&, const std::map<std::size_t, Polynomial>& replacements);
[[nodiscard]] Polynomial
determinant(const std::vector<std::vector<Polynomial>>& matrix);
[[nodiscard]] std::vector<std::vector<Polynomial>>
adjugate(const std::vector<std::vector<Polynomial>>& matrix);
[[nodiscard]] Polynomial
quadratic_form(const std::vector<Polynomial>& vector,
               const std::vector<std::vector<Polynomial>>& matrix);

} // namespace compiler::detail
