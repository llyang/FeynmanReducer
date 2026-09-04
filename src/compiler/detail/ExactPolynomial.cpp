#include "compiler/detail/ExactPolynomial.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace compiler::detail {

Rational operator-(Rational value)
{
  fmpq_neg(value.value_, value.value_);
  return value;
}

Rational operator+(const Rational& lhs, const Rational& rhs)
{
  Rational result;
  fmpq_add(result.value_, lhs.value_, rhs.value_);
  return result;
}

Rational operator-(const Rational& lhs, const Rational& rhs)
{
  Rational result;
  fmpq_sub(result.value_, lhs.value_, rhs.value_);
  return result;
}

Rational operator*(const Rational& lhs, const Rational& rhs)
{
  Rational result;
  fmpq_mul(result.value_, lhs.value_, rhs.value_);
  return result;
}

Rational operator/(const Rational& lhs, const Rational& rhs)
{
  if (rhs.is_zero()) {
    throw std::runtime_error("division by zero");
  }
  Rational result;
  fmpq_div(result.value_, lhs.value_, rhs.value_);
  return result;
}

Polynomial zero_poly(std::size_t count)
{
  return Polynomial{count, {}};
}

// Passing coefficient by value lets rvalue callers transfer the FLINT value.
void add_term(Polynomial& poly, Exponents powers,
              Rational coefficient) // NOLINT(performance-unnecessary-value-param)
{
  if (coefficient.is_zero()) {
    return;
  }
  auto [found, inserted] =
      poly.terms.try_emplace(std::move(powers), std::move(coefficient));
  if (!inserted) {
    found->second = found->second + coefficient;
    if (found->second.is_zero()) {
      poly.terms.erase(found);
    }
  }
}

Polynomial constant_poly(std::size_t count, Rational value)
{
  Polynomial result = zero_poly(count);
  add_term(result, Exponents(count, 0), std::move(value));
  return result;
}

Polynomial one_poly(std::size_t count)
{
  return constant_poly(count, Rational(1));
}

Polynomial
symbol_poly(std::size_t count, // NOLINT(bugprone-easily-swappable-parameters)
            std::size_t index)
{
  Exponents powers(count, 0);
  powers.at(index) = 1;
  Polynomial result = zero_poly(count);
  result.terms.emplace(std::move(powers), Rational(1));
  return result;
}

void require_same_context(const Polynomial& lhs, const Polynomial& rhs)
{
  if (lhs.symbol_count != rhs.symbol_count) {
    throw std::runtime_error("internal polynomial context mismatch");
  }
}

Polynomial operator+(const Polynomial& lhs, const Polynomial& rhs)
{
  require_same_context(lhs, rhs);
  Polynomial result = lhs;
  for (const auto& [powers, coefficient] : rhs.terms) {
    add_term(result, powers, coefficient);
  }
  return result;
}

Polynomial operator-(const Polynomial& value)
{
  Polynomial result = zero_poly(value.symbol_count);
  for (const auto& [powers, coefficient] : value.terms) {
    result.terms.emplace(powers, -coefficient);
  }
  return result;
}

Polynomial operator-(const Polynomial& lhs, const Polynomial& rhs)
{
  return lhs + (-rhs);
}

Polynomial operator*(const Polynomial& lhs, const Polynomial& rhs)
{
  require_same_context(lhs, rhs);
  Polynomial result = zero_poly(lhs.symbol_count);
  for (const auto& [lhs_powers, lhs_coefficient] : lhs.terms) {
    for (const auto& [rhs_powers, rhs_coefficient] : rhs.terms) {
      Exponents powers(lhs.symbol_count, 0);
      for (std::size_t index = 0; index < powers.size(); ++index) {
        powers[index] = lhs_powers[index] + rhs_powers[index];
      }
      add_term(result, std::move(powers), lhs_coefficient * rhs_coefficient);
    }
  }
  return result;
}

Polynomial scale(const Polynomial& value, const Rational& scalar)
{
  Polynomial result = zero_poly(value.symbol_count);
  for (const auto& [powers, coefficient] : value.terms) {
    add_term(result, powers, coefficient * scalar);
  }
  return result;
}

Polynomial power(Polynomial base, int exponent)
{
  if (exponent < 0) {
    throw std::runtime_error("negative polynomial powers are unsupported");
  }
  Polynomial result = one_poly(base.symbol_count);
  while (exponent > 0) {
    if ((exponent & 1) != 0) {
      result = result * base;
    }
    exponent >>= 1;
    if (exponent != 0) {
      base = base * base;
    }
  }
  return result;
}

bool is_constant(const Polynomial& value)
{
  return value.terms.empty() ||
         (value.terms.size() == 1 &&
          value.terms.begin()->first == Exponents(value.symbol_count, 0));
}

Rational constant_value(const Polynomial& value)
{
  if (!is_constant(value)) {
    throw std::runtime_error("polynomial division is unsupported");
  }
  return value.terms.empty() ? Rational(0) : value.terms.begin()->second;
}

Polynomial derivative(const Polynomial& value, std::size_t variable)
{
  Polynomial result = zero_poly(value.symbol_count);
  for (const auto& [powers, coefficient] : value.terms) {
    if (powers.at(variable) == 0) {
      continue;
    }
    Exponents derived = powers;
    const int factor = derived.at(variable);
    --derived.at(variable);
    add_term(result, std::move(derived), coefficient * Rational(factor));
  }
  return result;
}

Polynomial substitute(const Polynomial& value,
                      const std::map<std::size_t, Polynomial>& replacements)
{
  Polynomial result = zero_poly(value.symbol_count);
  for (const auto& [powers, coefficient] : value.terms) {
    Polynomial term = constant_poly(value.symbol_count, coefficient);
    for (std::size_t variable = 0; variable < powers.size(); ++variable) {
      if (powers[variable] == 0) {
        continue;
      }
      const auto found = replacements.find(variable);
      const Polynomial factor = found == replacements.end()
                                    ? symbol_poly(value.symbol_count, variable)
                                    : found->second;
      term = term * power(factor, powers[variable]);
    }
    result = result + term;
  }
  return result;
}

Polynomial determinant(const std::vector<std::vector<Polynomial>>& matrix)
{
  const std::size_t size = matrix.size();
  if (size == 0) {
    throw std::runtime_error("cannot compute an empty determinant");
  }
  if (size == 1) {
    return matrix[0][0];
  }
  Polynomial result = zero_poly(matrix[0][0].symbol_count);
  for (std::size_t column = 0; column < size; ++column) {
    std::vector<std::vector<Polynomial>> minor;
    for (std::size_t row = 1; row < size; ++row) {
      std::vector<Polynomial> minor_row;
      for (std::size_t inner = 0; inner < size; ++inner) {
        if (inner != column) {
          minor_row.push_back(matrix[row][inner]);
        }
      }
      minor.push_back(std::move(minor_row));
    }
    const Polynomial cofactor = matrix[0][column] * determinant(minor);
    result = column % 2 == 0 ? result + cofactor : result - cofactor;
  }
  return result;
}

std::vector<std::vector<Polynomial>>
adjugate(const std::vector<std::vector<Polynomial>>& matrix)
{
  const std::size_t size = matrix.size();
  std::vector result(size, std::vector(size, zero_poly(matrix[0][0].symbol_count)));
  if (size == 1) {
    result[0][0] = one_poly(matrix[0][0].symbol_count);
    return result;
  }
  for (std::size_t row = 0; row < size; ++row) {
    for (std::size_t column = 0; column < size; ++column) {
      std::vector<std::vector<Polynomial>> minor;
      for (std::size_t r = 0; r < size; ++r) {
        if (r == row) {
          continue;
        }
        std::vector<Polynomial> minor_row;
        for (std::size_t c = 0; c < size; ++c) {
          if (c != column) {
            minor_row.push_back(matrix[r][c]);
          }
        }
        minor.push_back(std::move(minor_row));
      }
      Polynomial cofactor = determinant(minor);
      result[column][row] = (row + column) % 2 == 0 ? cofactor : -cofactor;
    }
  }
  return result;
}

Polynomial quadratic_form(const std::vector<Polynomial>& vector,
                          const std::vector<std::vector<Polynomial>>& matrix)
{
  Polynomial result = zero_poly(vector[0].symbol_count);
  for (std::size_t row = 0; row < vector.size(); ++row) {
    for (std::size_t column = 0; column < vector.size(); ++column) {
      result = result + vector[row] * matrix[row][column] * vector[column];
    }
  }
  return result;
}

} // namespace compiler::detail
