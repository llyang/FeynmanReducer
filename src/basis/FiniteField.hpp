#pragma once

#include <cstddef>
#include <cstdint>
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

struct RationalFunction {
  FieldVector numerator;
  FieldVector denominator;
};

[[nodiscard]] std::uint64_t evaluate(const PrimeField& field,
                                     const RationalFunction& function,
                                     std::uint64_t argument);

} // namespace basis
