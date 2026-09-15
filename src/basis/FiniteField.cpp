#include "FiniteField.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>

#include <flint/ulong_extras.h>

namespace basis {

namespace {

__extension__ typedef unsigned __int128 WideProduct;

} // namespace

PrimeField::PrimeField(std::uint64_t prime) : prime_(prime)
{
  if (prime < 3) throw std::invalid_argument("finite-field prime is too small");
}

std::uint64_t PrimeField::normalize(std::int64_t value) const noexcept
{
  if (value >= 0) {
    const auto residue = static_cast<std::uint64_t>(value);
    return residue < prime_ ? residue : residue % prime_;
  }
  const auto magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1;
  const auto residue = magnitude < prime_ ? magnitude : magnitude % prime_;
  return residue == 0 ? 0 : prime_ - residue;
}

std::uint64_t PrimeField::add(std::uint64_t lhs, std::uint64_t rhs) const noexcept
{
  if (lhs >= prime_) lhs %= prime_;
  if (rhs >= prime_) rhs %= prime_;
  return lhs >= prime_ - rhs ? lhs - (prime_ - rhs) : lhs + rhs;
}

std::uint64_t PrimeField::subtract(std::uint64_t lhs, std::uint64_t rhs) const noexcept
{
  if (lhs >= prime_) lhs %= prime_;
  if (rhs >= prime_) rhs %= prime_;
  return lhs >= rhs ? lhs - rhs : prime_ - (rhs - lhs);
}

std::uint64_t PrimeField::multiply(std::uint64_t lhs, std::uint64_t rhs) const noexcept
{
  if (lhs >= prime_) lhs %= prime_;
  if (rhs >= prime_) rhs %= prime_;
  if (lhs == 0 || rhs == 0) return 0;
  if (lhs == 1) return rhs;
  if (rhs == 1) return lhs;
  const auto product = static_cast<WideProduct>(lhs) * static_cast<WideProduct>(rhs);
  return static_cast<std::uint64_t>(product % prime_);
}

std::uint64_t PrimeField::power(std::uint64_t value,
                                std::uint64_t exponent) const noexcept
{
  std::uint64_t result = 1;
  if (value >= prime_) value %= prime_;
  while (exponent != 0) {
    if ((exponent & 1U) != 0) result = multiply(result, value);
    value = multiply(value, value);
    exponent >>= 1U;
  }
  return result;
}

std::uint64_t PrimeField::inverse(std::uint64_t value) const
{
  if (value >= prime_) value %= prime_;
  if (value == 0) throw std::domain_error("division by zero in finite field");
  if (value == 1 || value == prime_ - 1) return value;
  ulong result = 0;
  if (n_gcdinv(&result, value, prime_) != 1)
    throw std::domain_error("element is not invertible in finite field");
  return result;
}

std::uint64_t PrimeField::divide(std::uint64_t lhs, std::uint64_t rhs) const
{
  return multiply(lhs, inverse(rhs));
}

LinearSolveResult solve_linear_system(const PrimeField& field, FieldMatrix matrix,
                                      FieldVector rhs)
{
  if (matrix.size() != rhs.size())
    throw std::invalid_argument("linear-system row count mismatch");
  const std::size_t columns = matrix.empty() ? 0 : matrix.front().size();
  for (const auto& row : matrix) {
    if (row.size() != columns)
      throw std::invalid_argument("ragged finite-field matrix");
  }

  std::vector<std::size_t> pivot_columns;
  std::size_t pivot_row = 0;
  for (std::size_t column = 0; column < columns && pivot_row < matrix.size();
       ++column) {
    std::size_t selected = pivot_row;
    while (selected < matrix.size() && matrix[selected][column] == 0)
      ++selected;
    if (selected == matrix.size()) continue;
    std::swap(matrix[pivot_row], matrix[selected]);
    std::swap(rhs[pivot_row], rhs[selected]);
    const auto inverse = field.inverse(matrix[pivot_row][column]);
    for (std::size_t current = column; current < columns; ++current)
      matrix[pivot_row][current] = field.multiply(matrix[pivot_row][current], inverse);
    rhs[pivot_row] = field.multiply(rhs[pivot_row], inverse);
    for (std::size_t row = 0; row < matrix.size(); ++row) {
      if (row == pivot_row || matrix[row][column] == 0) continue;
      const auto factor = matrix[row][column];
      for (std::size_t current = column; current < columns; ++current) {
        matrix[row][current] = field.subtract(
            matrix[row][current], field.multiply(factor, matrix[pivot_row][current]));
      }
      rhs[row] = field.subtract(rhs[row], field.multiply(factor, rhs[pivot_row]));
    }
    pivot_columns.push_back(column);
    ++pivot_row;
  }

  for (std::size_t row = pivot_row; row < matrix.size(); ++row) {
    const bool zero = std::ranges::all_of(
        matrix[row], [](std::uint64_t value) { return value == 0; });
    if (zero && rhs[row] != 0)
      return {.consistent = false,
              .unique = false,
              .rank = pivot_columns.size(),
              .solution = {}};
  }

  FieldVector solution(columns, 0);
  for (std::size_t row = 0; row < pivot_columns.size(); ++row)
    solution[pivot_columns[row]] = rhs[row];
  return {.consistent = true,
          .unique = pivot_columns.size() == columns,
          .rank = pivot_columns.size(),
          .solution = std::move(solution)};
}

std::size_t matrix_rank(const PrimeField& field, FieldMatrix matrix)
{
  const auto rows = matrix.size();
  return solve_linear_system(field, std::move(matrix), FieldVector(rows, 0)).rank;
}

FieldVector multiply(const PrimeField& field, const FieldMatrix& matrix,
                     std::span<const std::uint64_t> vector)
{
  FieldVector result(matrix.size(), 0);
  for (std::size_t row = 0; row < matrix.size(); ++row) {
    if (matrix[row].size() != vector.size())
      throw std::invalid_argument("matrix-vector dimension mismatch");
    for (std::size_t column = 0; column < vector.size(); ++column) {
      result[row] =
          field.add(result[row], field.multiply(matrix[row][column], vector[column]));
    }
  }
  return result;
}

std::uint64_t evaluate(const PrimeField& field, const RationalFunction& function,
                       std::uint64_t argument)
{
  if (argument >= field.prime()) argument %= field.prime();
  const auto evaluate_polynomial =
      [&field, argument](std::span<const std::uint64_t> coefficients) {
        std::uint64_t result = 0;
        for (auto coefficient = coefficients.rbegin();
             coefficient != coefficients.rend(); ++coefficient) {
          result = field.add(field.multiply(result, argument), *coefficient);
        }
        return result;
      };
  const auto numerator = evaluate_polynomial(function.numerator);
  const auto denominator = evaluate_polynomial(function.denominator);
  return field.divide(numerator, denominator);
}

} // namespace basis
