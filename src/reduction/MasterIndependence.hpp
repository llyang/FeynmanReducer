#pragma once

#include "reduction/EliminationTape.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <limits>
#include <stdexcept>
#include <vector>

namespace reduction::detail {

class MasterBasisDependenceError : public std::runtime_error {
public:
  explicit MasterBasisDependenceError(std::size_t direction)
      : std::runtime_error(std::format("isolated master basis is linearly dependent in "
                                       "the current relation system "
                                       "(master direction {})",
                                       direction + 1))
  {}
};

struct MasterIndependenceTimings {
  double completion_ms = 0.0;
  double check_ms = 0.0;
};

// Complete only the unpivoted part of the existing echelon. Physical pivot rows
// are immutable; moving the residual rows avoids copying the large system.
template <class T>
void complete_residual(linalg::SparseMatrix<T>& matrix,
                       linalg::EliminationResult& result, std::size_t num_cols)
{
  const auto prefix = result.solution_cols.size();
  std::vector<std::size_t> rows, columns;
  for (std::size_t i = prefix; i < result.row_map.size(); ++i) {
    const auto row = result.row_map[i];
    if (matrix[row].empty()) continue;
    rows.push_back(row);
    for (const auto& e : matrix[row])
      columns.push_back(e.column);
  }
  std::sort(columns.begin(), columns.end());
  columns.erase(std::unique(columns.begin(), columns.end()), columns.end());
  if (!rows.empty()) {
    std::vector<std::size_t> local(num_cols, std::numeric_limits<std::size_t>::max());
    for (std::size_t i = 0; i < columns.size(); ++i)
      local[columns[i]] = i;
    linalg::SparseMatrix<T> remainder;
    remainder.reserve(rows.size());
    for (const auto row : rows) {
      auto r = std::move(matrix[row]);
      for (auto& e : r)
        e.column = static_cast<std::uint32_t>(local[e.column]);
      remainder.push_back(std::move(r));
    }
    std::vector<T> no_rhs;
    const auto tail = linalg::sparse_gaussian_elimination(remainder, columns.size(),
                                                          no_rhs, 0, columns.size());
    for (auto c : tail.solution_cols)
      result.solution_cols.push_back(columns[c]);
    std::vector<unsigned char> tail_used(matrix.size(), 0);
    std::vector<std::size_t> row_map(result.row_map.begin(),
                                     result.row_map.begin() +
                                         static_cast<std::ptrdiff_t>(prefix));
    for (std::size_t i = 0; i < tail.solution_cols.size(); ++i) {
      const auto row = rows[tail.row_map[i]];
      row_map.push_back(row);
      tail_used[row] = 1;
    }
    for (std::size_t i = prefix; i < result.row_map.size(); ++i)
      if (!tail_used[result.row_map[i]]) row_map.push_back(result.row_map[i]);
    result.row_map = std::move(row_map);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      for (auto& e : remainder[i])
        e.column = static_cast<std::uint32_t>(columns[e.column]);
      matrix[rows[i]] = std::move(remainder[i]);
    }
  }
  if (!result.closed && result.solution_cols.size() != prefix)
    throw std::logic_error("unclosed provisional system has unprocessed pivots");
}

// Unit covectors for the master coordinates must annihilate every relation
// column. Propagating them through the triangular pivot system tests
// rank([relations,basis])-rank(relations)==basis.size() exactly in this field.
// One column-sized scratch vector is reused across all master directions.
template <class T>
void check_master_directions(const linalg::SparseMatrix<T>& matrix,
                             const linalg::EliminationResult& result,
                             std::size_t num_cols, std::size_t basis_cols)
{
  std::vector<unsigned char> pivot(num_cols, 0);
  for (auto c : result.solution_cols)
    pivot[c] = 1;
  std::vector<T> weights(num_cols, T(0));
  for (std::size_t master = 0; master < basis_cols; ++master) {
    if (!pivot[master]) throw MasterBasisDependenceError(master);
    std::fill(weights.begin(), weights.end(), T(0));
    weights[master] = T(1);
    for (std::size_t i = 0; i < result.solution_cols.size(); ++i) {
      const auto col = result.solution_cols[i];
      const T factor = weights[col];
      if (factor == T(0)) continue;
      const auto& row = matrix[result.row_map[i]];
      for (const auto& entry : row) {
        if (entry.column != col)
          weights[entry.column] = weights[entry.column] - factor * entry.value;
      }
      weights[col] = T(0);
    }
    for (std::size_t col = basis_cols; col < num_cols; ++col) {
      if (!pivot[col] && weights[col] != T(0)) {
        throw MasterBasisDependenceError(master);
      }
    }
  }
}

// Keep the physical pivot metadata unchanged so the audit cannot enlarge the
// target-selected compact support. Only unused matrix rows are modified.
template <class T>
MasterIndependenceTimings
audit_master_independence(linalg::SparseMatrix<T>& matrix,
                          const linalg::EliminationResult& physical,
                          std::size_t num_cols, std::size_t basis_cols)
{
  if (basis_cols == 0) return {};
  const auto start = std::chrono::steady_clock::now();
  auto complete = physical;
  complete_residual(matrix, complete, num_cols);
  const auto completed = std::chrono::steady_clock::now();
  check_master_directions(matrix, complete, num_cols, basis_cols);
  const auto checked = std::chrono::steady_clock::now();
  return {std::chrono::duration<double, std::milli>(completed - start).count(),
          std::chrono::duration<double, std::milli>(checked - completed).count()};
}

} // namespace reduction::detail
