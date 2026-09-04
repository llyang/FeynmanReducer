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

namespace {

std::vector<std::size_t> select_independent_rows(const PrimeField& field,
                                                 const FieldMatrix& matrix,
                                                 std::size_t columns)
{
  FieldMatrix echelon;
  std::vector<std::size_t> pivot_columns;
  std::vector<std::size_t> selected_rows;
  echelon.reserve(columns);
  pivot_columns.reserve(columns);
  selected_rows.reserve(columns);
  for (std::size_t input_row = 0; input_row < matrix.size(); ++input_row) {
    auto row = matrix[input_row];
    for (std::size_t pivot = 0; pivot < echelon.size(); ++pivot) {
      const auto column = pivot_columns[pivot];
      const auto factor = row[column];
      if (factor == 0) continue;
      for (std::size_t current = column; current < columns; ++current) {
        row[current] = field.subtract(row[current],
                                      field.multiply(factor, echelon[pivot][current]));
      }
    }
    const auto first =
        std::ranges::find_if(row, [](std::uint64_t value) { return value != 0; });
    if (first == row.end()) continue;
    const auto column = static_cast<std::size_t>(first - row.begin());
    const auto inverse = field.inverse(row[column]);
    for (std::size_t current = column; current < columns; ++current)
      row[current] = field.multiply(row[current], inverse);
    echelon.push_back(std::move(row));
    pivot_columns.push_back(column);
    selected_rows.push_back(input_row);
    if (selected_rows.size() == columns) break;
  }
  return selected_rows;
}

FieldMatrix invert_square_matrix(const PrimeField& field, FieldMatrix matrix)
{
  const auto size = matrix.size();
  for (const auto& row : matrix) {
    if (row.size() != size)
      throw std::invalid_argument("finite-field inverse requires a square matrix");
  }
  FieldMatrix inverse(size, FieldVector(size, 0));
  for (std::size_t index = 0; index < size; ++index)
    inverse[index][index] = 1;
  for (std::size_t column = 0; column < size; ++column) {
    std::size_t selected = column;
    while (selected < size && matrix[selected][column] == 0)
      ++selected;
    if (selected == size)
      throw std::invalid_argument("finite-field factorization is singular");
    std::swap(matrix[column], matrix[selected]);
    std::swap(inverse[column], inverse[selected]);
    const auto scale = field.inverse(matrix[column][column]);
    for (std::size_t current = 0; current < size; ++current) {
      matrix[column][current] = field.multiply(matrix[column][current], scale);
      inverse[column][current] = field.multiply(inverse[column][current], scale);
    }
    for (std::size_t row = 0; row < size; ++row) {
      if (row == column || matrix[row][column] == 0) continue;
      const auto factor = matrix[row][column];
      for (std::size_t current = 0; current < size; ++current) {
        matrix[row][current] = field.subtract(
            matrix[row][current], field.multiply(factor, matrix[column][current]));
        inverse[row][current] = field.subtract(
            inverse[row][current], field.multiply(factor, inverse[column][current]));
      }
    }
  }
  return inverse;
}

void validate_series_shape(std::span<const std::uint64_t> lhs,
                           std::span<const std::uint64_t> rhs)
{
  if (lhs.empty() || lhs.size() != rhs.size())
    throw std::invalid_argument("truncated-series shape mismatch");
}

} // namespace

ReusableLinearFactorization::ReusableLinearFactorization(
    PrimeField field, FieldMatrix matrix, std::vector<std::size_t> selected_rows,
    FieldMatrix inverse)
    : field_(field), matrix_(std::move(matrix)),
      selected_rows_(std::move(selected_rows)), inverse_(std::move(inverse))
{}

ReusableLinearFactorization
ReusableLinearFactorization::factorize_full_column_rank(const PrimeField& field,
                                                        const FieldMatrix& matrix)
{
  const auto columns = matrix.empty() ? 0 : matrix.front().size();
  if (columns == 0)
    throw std::invalid_argument("cannot factorize an empty finite-field matrix");
  for (const auto& row : matrix) {
    if (row.size() != columns)
      throw std::invalid_argument("ragged finite-field matrix");
  }
  auto selected_rows = select_independent_rows(field, matrix, columns);
  if (selected_rows.size() != columns)
    throw std::invalid_argument("finite-field matrix is not full column rank");
  FieldMatrix minor;
  minor.reserve(columns);
  for (const auto row : selected_rows)
    minor.push_back(matrix[row]);
  return ReusableLinearFactorization(field, matrix, std::move(selected_rows),
                                     invert_square_matrix(field, std::move(minor)));
}

FieldVector ReusableLinearFactorization::solve(std::span<const std::uint64_t> rhs) const
{
  if (rhs.size() != matrix_.size())
    throw std::invalid_argument("reusable solve row count mismatch");
  FieldVector selected_rhs(selected_rows_.size(), 0);
  for (std::size_t row = 0; row < selected_rows_.size(); ++row)
    selected_rhs[row] = rhs[selected_rows_[row]];
  auto solution = multiply(field_, inverse_, selected_rhs);
  if (multiply(field_, matrix_, solution) != FieldVector(rhs.begin(), rhs.end()))
    throw std::runtime_error("reusable solve is inconsistent on unselected rows");
  return solution;
}

FieldMatrix ReusableLinearFactorization::solve_many(const FieldMatrix& rhs) const
{
  if (rhs.size() != matrix_.size())
    throw std::invalid_argument("multi-RHS solve row count mismatch");
  const auto rhs_columns = rhs.empty() ? 0 : rhs.front().size();
  for (const auto& row : rhs) {
    if (row.size() != rhs_columns)
      throw std::invalid_argument("ragged finite-field multi-RHS matrix");
  }
  FieldMatrix result(column_count(), FieldVector(rhs_columns, 0));
  for (std::size_t column = 0; column < rhs_columns; ++column) {
    FieldVector vector(rhs.size(), 0);
    for (std::size_t row = 0; row < rhs.size(); ++row)
      vector[row] = rhs[row][column];
    const auto solution = solve(vector);
    for (std::size_t row = 0; row < solution.size(); ++row)
      result[row][column] = solution[row];
  }
  return result;
}

FieldMatrix ReusableLinearFactorization::solve_rows(const FieldMatrix& rows) const
{
  if (rows.empty()) return {};
  FieldMatrix right_hand_sides(row_count(), FieldVector(rows.size(), 0));
  for (std::size_t vector = 0; vector < rows.size(); ++vector) {
    if (rows[vector].size() != row_count())
      throw std::invalid_argument("finite-field row vector has the wrong dimension");
    for (std::size_t row = 0; row < row_count(); ++row)
      right_hand_sides[row][vector] = rows[vector][row];
  }
  const auto column_solutions = solve_many(right_hand_sides);
  FieldMatrix result(rows.size(), FieldVector(column_count(), 0));
  for (std::size_t coordinate = 0; coordinate < column_solutions.size(); ++coordinate) {
    for (std::size_t vector = 0; vector < rows.size(); ++vector)
      result[vector][coordinate] = column_solutions[coordinate][vector];
  }
  return result;
}

TruncatedFieldSeries series_add(const PrimeField& field,
                                std::span<const std::uint64_t> lhs,
                                std::span<const std::uint64_t> rhs)
{
  validate_series_shape(lhs, rhs);
  TruncatedFieldSeries result(lhs.size(), 0);
  for (std::size_t order = 0; order < lhs.size(); ++order)
    result[order] = field.add(lhs[order], rhs[order]);
  return result;
}

TruncatedFieldSeries series_multiply(const PrimeField& field,
                                     std::span<const std::uint64_t> lhs,
                                     std::span<const std::uint64_t> rhs)
{
  validate_series_shape(lhs, rhs);
  TruncatedFieldSeries result(lhs.size(), 0);
  for (std::size_t order = 0; order < lhs.size(); ++order) {
    for (std::size_t left = 0; left <= order; ++left) {
      result[order] =
          field.add(result[order], field.multiply(lhs[left], rhs[order - left]));
    }
  }
  return result;
}

TruncatedFieldSeries series_scale(const PrimeField& field,
                                  std::span<const std::uint64_t> series,
                                  std::uint64_t scalar)
{
  if (series.empty()) throw std::invalid_argument("truncated series is empty");
  TruncatedFieldSeries result(series.size(), 0);
  for (std::size_t order = 0; order < series.size(); ++order)
    result[order] = field.multiply(series[order], scalar);
  return result;
}

namespace {

FieldVector powers(const PrimeField& field, std::uint64_t value, std::size_t degree)
{
  FieldVector result(degree + 1, 1);
  for (std::size_t index = 1; index <= degree; ++index)
    result[index] = field.multiply(result[index - 1], value);
  return result;
}

} // namespace

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

std::optional<RationalFunction>
interpolate_rational(const PrimeField& field, std::span<const std::uint64_t> arguments,
                     std::span<const std::uint64_t> values,
                     std::size_t maximum_total_degree, std::size_t holdout_count)
{
  if (arguments.size() != values.size() || holdout_count == 0 ||
      arguments.size() <= holdout_count)
    throw std::invalid_argument("invalid rational interpolation samples");
  const std::size_t training_count = arguments.size() - holdout_count;
  for (std::size_t total = 0; total <= maximum_total_degree; ++total) {
    for (std::size_t denominator_degree = 0; denominator_degree <= total;
         ++denominator_degree) {
      const std::size_t numerator_degree = total - denominator_degree;
      const std::size_t unknowns = numerator_degree + 1 + denominator_degree;
      if (training_count < unknowns) continue;
      FieldMatrix matrix(training_count, FieldVector(unknowns, 0));
      FieldVector rhs(training_count, 0);
      for (std::size_t sample = 0; sample < training_count; ++sample) {
        const auto sample_powers = powers(field, arguments[sample], total);
        for (std::size_t degree = 0; degree <= numerator_degree; ++degree)
          matrix[sample][degree] = sample_powers[degree];
        for (std::size_t degree = 1; degree <= denominator_degree; ++degree) {
          matrix[sample][numerator_degree + degree] =
              field.subtract(0, field.multiply(values[sample], sample_powers[degree]));
        }
        rhs[sample] = values[sample];
      }
      const auto solved = solve_linear_system(field, std::move(matrix), std::move(rhs));
      if (!solved.consistent || !solved.unique) continue;
      RationalFunction candidate;
      const auto numerator_end =
          static_cast<FieldVector::difference_type>(numerator_degree + 1);
      candidate.numerator.assign(solved.solution.begin(),
                                 solved.solution.begin() + numerator_end);
      candidate.denominator.assign(denominator_degree + 1, 0);
      candidate.denominator[0] = 1;
      for (std::size_t degree = 1; degree <= denominator_degree; ++degree) {
        candidate.denominator[degree] = solved.solution[numerator_degree + degree];
      }
      bool valid = true;
      for (std::size_t sample = training_count; sample < arguments.size(); ++sample) {
        try {
          if (evaluate(field, candidate, arguments[sample]) != values[sample]) {
            valid = false;
            break;
          }
        } catch (const std::domain_error&) {
          valid = false;
          break;
        }
      }
      if (valid) return candidate;
    }
  }
  return std::nullopt;
}

FieldVector taylor_coefficients(const PrimeField& field,
                                const RationalFunction& function, std::size_t order)
{
  if (function.denominator.empty() || function.denominator.front() == 0)
    throw std::domain_error("rational function is singular at the origin");
  const auto inverse_constant = field.inverse(function.denominator.front());
  FieldVector result(order + 1, 0);
  for (std::size_t degree = 0; degree <= order; ++degree) {
    std::uint64_t value =
        degree < function.numerator.size() ? function.numerator[degree] : 0;
    for (std::size_t denominator_degree = 1;
         denominator_degree <= degree &&
         denominator_degree < function.denominator.size();
         ++denominator_degree) {
      value =
          field.subtract(value, field.multiply(function.denominator[denominator_degree],
                                               result[degree - denominator_degree]));
    }
    result[degree] = field.multiply(value, inverse_constant);
  }
  return result;
}

} // namespace basis
