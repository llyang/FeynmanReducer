#pragma once

#include "reduction/FiniteFieldArithmetic.hpp"
#include "reduction/SparseMarkowitz.hpp"

#include <firefly/FFInt.hpp>
#include <optional>

namespace reduction::detail {

// Preparation-only numerical data. All stored field values are canonical residues;
// the recorded operators are valid only at this prime and full parameter point.
struct ProvisionalCheckpoint {
  using T = finite_field::MontgomeryFieldElement;
  std::uint64_t prime = 0;
  bool master_independence_checked = false;
  std::vector<std::uint64_t> parameters;
  std::size_t basis_columns = 0, relation_columns = 0;
  std::vector<linalg::NumericalRowOperation<std::uint64_t>> operations;
  std::vector<std::size_t> row_map, solution_columns;
  linalg::SparseMatrix<std::uint64_t> upper;

  void capture(const linalg::SparseMatrix<T>& matrix,
               const linalg::EliminationResult& elimination,
               std::span<const linalg::NumericalRowOperation<T>> trace,
               std::size_t basis_count, std::size_t relation_count,
               std::uint64_t modulus)
  {
    prime = modulus;
    basis_columns = basis_count;
    relation_columns = relation_count;
    row_map = elimination.row_map;
    solution_columns = elimination.solution_cols;
    operations.reserve(trace.size());
    for (const auto& op : trace)
      operations.push_back({op.destination, op.source, op.factor.residue()});
    std::vector<std::size_t> position(basis_count + relation_count, matrix.size());
    for (std::size_t i = 0; i < solution_columns.size(); ++i)
      position.at(solution_columns[i]) = i;
    upper.resize(solution_columns.size());
    for (std::size_t i = 0; i < upper.size(); ++i) {
      for (const auto& entry : matrix.at(row_map.at(i))) {
        const auto j = position.at(entry.column);
        if (j > i && j < upper.size()) upper[i].push_back({j, entry.value.residue()});
      }
    }
  }

  [[nodiscard]] std::size_t bytes() const
  {
    std::size_t result =
        sizeof(*this) + parameters.capacity() * sizeof(std::uint64_t) +
        operations.capacity() * sizeof(operations[0]) +
        (row_map.capacity() + solution_columns.capacity()) * sizeof(std::size_t) +
        upper.capacity() * sizeof(upper[0]);
    for (const auto& row : upper)
      result += row.capacity() * sizeof(row[0]);
    return result;
  }

  [[nodiscard]] bool matches(std::uint64_t modulus,
                             std::span<const firefly::FFInt> point) const
  {
    if (prime != modulus || parameters.size() != point.size()) return false;
    for (std::size_t i = 0; i < point.size(); ++i)
      if (parameters[i] != point[i].n) return false;
    return true;
  }

  // Solve only requested RHS columns in the recorded independent column frame.
  // A nonzero residual means this frame cannot represent the new RHS.
  [[nodiscard]] std::optional<std::vector<T>>
  solve(std::vector<T> rhs, std::size_t width, std::uint64_t modulus) const
  {
    if (modulus != prime) throw std::logic_error("checkpoint prime mismatch");
    if (width == 0 || rhs.size() != row_map.size() * width ||
        upper.size() != solution_columns.size() || upper.size() > row_map.size())
      throw std::logic_error("checkpoint shape mismatch");
    std::vector<bool> used(row_map.size(), false);
    for (auto row : row_map) {
      if (row >= used.size() || used[row])
        throw std::logic_error("checkpoint row map is not a permutation");
      used[row] = true;
    }
    std::vector<bool> columns(basis_columns + relation_columns, false);
    for (auto column : solution_columns) {
      if (column >= columns.size() || columns[column])
        throw std::logic_error("checkpoint pivot map is invalid");
      columns[column] = true;
    }
    for (const auto& op : operations) {
      if (op.factor >= prime)
        throw std::logic_error("checkpoint factor is not canonical");
      if (op.destination >= row_map.size() || op.source >= row_map.size())
        throw std::logic_error("checkpoint operation row out of range");
      const T factor = T::from_residue(op.factor);
      for (std::size_t j = 0; j < width; ++j) {
        auto& dst = rhs[op.destination * width + j];
        if (op.destination == op.source)
          dst = dst * factor;
        else
          dst = dst - factor * rhs[op.source * width + j];
      }
    }
    for (std::size_t i = upper.size(); i < row_map.size(); ++i)
      for (std::size_t j = 0; j < width; ++j)
        if (rhs.at(row_map[i] * width + j) != T(0)) return std::nullopt;
    std::vector<T> solution(upper.size() * width, T(0));
    for (std::size_t i = upper.size(); i-- > 0;) {
      for (std::size_t j = 0; j < width; ++j)
        solution[i * width + j] = rhs.at(row_map[i] * width + j);
      for (const auto& entry : upper[i]) {
        if (entry.column <= i || entry.column >= upper.size())
          throw std::logic_error("checkpoint upper column out of range");
        const T factor = T::from_residue(entry.value);
        for (std::size_t j = 0; j < width; ++j)
          solution[i * width + j] =
              solution[i * width + j] - factor * solution[entry.column * width + j];
      }
    }
    return solution;
  }
};

} // namespace reduction::detail
