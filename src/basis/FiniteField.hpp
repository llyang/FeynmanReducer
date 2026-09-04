#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace basis {

class PrimeField {
public:
  explicit PrimeField(std::uint64_t prime);

  [[nodiscard]] std::uint64_t prime() const noexcept
  {
    return prime_;
  }
  [[nodiscard]] std::uint64_t normalize(std::int64_t value) const noexcept;
  [[nodiscard]] std::uint64_t add(std::uint64_t lhs, std::uint64_t rhs) const noexcept;
  [[nodiscard]] std::uint64_t subtract(std::uint64_t lhs,
                                       std::uint64_t rhs) const noexcept;
  [[nodiscard]] std::uint64_t multiply(std::uint64_t lhs,
                                       std::uint64_t rhs) const noexcept;
  [[nodiscard]] std::uint64_t power(std::uint64_t value,
                                    std::uint64_t exponent) const noexcept;
  [[nodiscard]] std::uint64_t inverse(std::uint64_t value) const;
  [[nodiscard]] std::uint64_t divide(std::uint64_t lhs, std::uint64_t rhs) const;

private:
  std::uint64_t prime_;
};

using FieldVector = std::vector<std::uint64_t>;
using FieldMatrix = std::vector<FieldVector>;

struct LinearSolveResult {
  bool consistent = false;
  bool unique = false;
  std::size_t rank = 0;
  FieldVector solution;
};

[[nodiscard]] LinearSolveResult
solve_linear_system(const PrimeField& field, FieldMatrix matrix, FieldVector rhs);

[[nodiscard]] std::size_t matrix_rank(const PrimeField& field, FieldMatrix matrix);

[[nodiscard]] FieldVector multiply(const PrimeField& field, const FieldMatrix& matrix,
                                   std::span<const std::uint64_t> vector);

class ReusableLinearFactorization {
public:
  [[nodiscard]] static ReusableLinearFactorization
  factorize_full_column_rank(const PrimeField& field, const FieldMatrix& matrix);

  [[nodiscard]] FieldVector solve(std::span<const std::uint64_t> rhs) const;
  // rhs is stored as row -> right-hand-side column.
  [[nodiscard]] FieldMatrix solve_many(const FieldMatrix& rhs) const;
  // Solves a batch stored as right-hand-side row vectors. Each input row has
  // row_count() entries; each output row has column_count() coordinates.
  [[nodiscard]] FieldMatrix solve_rows(const FieldMatrix& rows) const;

  [[nodiscard]] std::size_t row_count() const noexcept
  {
    return matrix_.size();
  }
  [[nodiscard]] std::size_t column_count() const noexcept
  {
    return inverse_.size();
  }
  [[nodiscard]] std::span<const std::size_t> selected_rows() const noexcept
  {
    return selected_rows_;
  }

private:
  ReusableLinearFactorization(PrimeField field, FieldMatrix matrix,
                              std::vector<std::size_t> selected_rows,
                              FieldMatrix inverse);

  PrimeField field_;
  FieldMatrix matrix_;
  std::vector<std::size_t> selected_rows_;
  FieldMatrix inverse_;
};

using TruncatedFieldSeries = FieldVector;

[[nodiscard]] TruncatedFieldSeries series_add(const PrimeField& field,
                                              std::span<const std::uint64_t> lhs,
                                              std::span<const std::uint64_t> rhs);
[[nodiscard]] TruncatedFieldSeries series_multiply(const PrimeField& field,
                                                   std::span<const std::uint64_t> lhs,
                                                   std::span<const std::uint64_t> rhs);
[[nodiscard]] TruncatedFieldSeries series_scale(const PrimeField& field,
                                                std::span<const std::uint64_t> series,
                                                std::uint64_t scalar);

struct RationalFunction {
  FieldVector numerator;
  FieldVector denominator;
};

[[nodiscard]] std::optional<RationalFunction>
interpolate_rational(const PrimeField& field, std::span<const std::uint64_t> arguments,
                     std::span<const std::uint64_t> values,
                     std::size_t maximum_total_degree, std::size_t holdout_count);

[[nodiscard]] FieldVector taylor_coefficients(const PrimeField& field,
                                              const RationalFunction& function,
                                              std::size_t order);

[[nodiscard]] std::uint64_t evaluate(const PrimeField& field,
                                     const RationalFunction& function,
                                     std::uint64_t argument);

} // namespace basis

// Compatibility surface for the isolated Quotient research project. Production
// code uses the basis namespace; remove these aliases only together with that
// project's include-site migration.
namespace quotient {
using basis::evaluate;
using basis::FieldMatrix;
using basis::FieldVector;
using basis::interpolate_rational;
using basis::LinearSolveResult;
using basis::matrix_rank;
using basis::multiply;
using basis::PrimeField;
using basis::RationalFunction;
using basis::ReusableLinearFactorization;
using basis::series_add;
using basis::series_multiply;
using basis::series_scale;
using basis::solve_linear_system;
using basis::taylor_coefficients;
using basis::TruncatedFieldSeries;
} // namespace quotient
