#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace linalg {

// =========================================================================
// 零开销矩阵视图
//
// 包装裸指针 + 列数，提供 (row, col) 二维访问。
// 编译器会完全内联，不产生运行时开销。
// =========================================================================
template <typename T> struct MatrixView {
  T* data;
  size_t cols;
  T& operator()(size_t r, size_t c) noexcept
  {
    return data[r * cols + c];
  }
  const T& operator()(size_t r, size_t c) const noexcept
  {
    return data[r * cols + c];
  }
};

// =========================================================================
// 指令操作码
//
// 用于描述高斯消元过程中的每一步操作。
// 整个消元过程被录制为一条指令序列（Tape），之后可对不同的数值输入
// 进行高速回放，从而避免重复分析矩阵的稀疏结构和主元选择。
// =========================================================================
enum class OpCode : uint8_t {
  Inv = 0,   // 求主元逆元:      reg = 1 / M[src]
  MulM = 1,  // 矩阵元缩放:      M[dst] *= reg
  MulB = 2,  // 右端列缩放:      B[dst] *= reg
  LoadF = 3, // 加载消元因子:    reg = M[src]
  FmaM = 4,  // 矩阵行消元:      M[dst] -= reg * M[src]
  FmaB = 5   // 右端列消元:      B[dst] -= reg * B[src]
};

// =========================================================================
// 紧凑指令（8 字节）
//
// 两个 30 位槽位和 4 位操作码共享一个 64 位字。槽位上限对应至少
// 8 GiB 的 uint64_t 工作区，超过时直接拒绝，不再保留宽指令回退。
// =========================================================================
struct Instruction {
  std::uint64_t bits;

  static constexpr std::uint64_t OP_MASK = (std::uint64_t{1} << 30) - 1;

  OpCode opcode() const noexcept
  {
    return static_cast<OpCode>(bits >> 60);
  }
  std::uint32_t offset() const noexcept
  {
    return static_cast<std::uint32_t>((bits >> 30) & OP_MASK);
  }
  std::uint32_t source() const noexcept
  {
    return static_cast<std::uint32_t>(bits & OP_MASK);
  }

  static Instruction pack(OpCode opcode, std::uint64_t destination,
                          std::uint64_t source)
  {
    if (destination > OP_MASK || source > OP_MASK)
      throw std::runtime_error("tape slot exceeds 30-bit encoding");
    return {(static_cast<std::uint64_t>(opcode) << 60) | (destination << 30) | source};
  }
};

static_assert(sizeof(Instruction) == 8, "Instruction should be 8 bytes");

/// 高斯消元的返回结果
struct EliminationResult {
  std::vector<size_t> solution_cols; ///< 线性无关列索引
  bool closed;                       ///< 方程组是否相容
  std::vector<size_t> row_map;       ///< 行置换序列
  size_t cross_group_pivots = 0;     ///< 未使用首选行组的主元数量
};

template <typename T> struct SparseEntry {
  static constexpr std::uint32_t INVALID_SLOT =
      std::numeric_limits<std::uint32_t>::max();

  std::uint32_t column;
  std::uint32_t slot;
  T value;

  SparseEntry(size_t column_value, T field_value)
      : column(checked_index(column_value)), slot(INVALID_SLOT),
        value(std::move(field_value))
  {}

  SparseEntry(size_t column_value, std::uint32_t slot_value, T field_value)
      : column(checked_index(column_value)), slot(slot_value),
        value(std::move(field_value))
  {}

private:
  static std::uint32_t checked_index(size_t value)
  {
    if (value > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("sparse column exceeds 32-bit encoding");
    return static_cast<std::uint32_t>(value);
  }
};

template <typename T> using SparseRow = std::vector<SparseEntry<T>>;

template <typename T> using SparseMatrix = std::vector<SparseRow<T>>;

template <typename T>
[[nodiscard]] inline auto sparse_lower_bound(SparseRow<T>& row, size_t column)
{
  return std::lower_bound(
      row.begin(), row.end(), column,
      [](const SparseEntry<T>& entry, size_t value) { return entry.column < value; });
}

template <typename T>
[[nodiscard]] inline auto sparse_lower_bound(const SparseRow<T>& row, size_t column)
{
  return std::lower_bound(
      row.begin(), row.end(), column,
      [](const SparseEntry<T>& entry, size_t value) { return entry.column < value; });
}

template <typename T>
[[nodiscard]] inline const T* sparse_find(const SparseRow<T>& row, size_t column)
{
  const auto found = sparse_lower_bound(row, column);
  if (found == row.end() || found->column != column) return nullptr;
  return &found->value;
}

template <typename T>
[[nodiscard]] inline SparseEntry<T>* sparse_find_entry(SparseRow<T>& row, size_t column)
{
  const auto found = sparse_lower_bound(row, column);
  if (found == row.end() || found->column != column) return nullptr;
  return &*found;
}

template <typename T>
[[nodiscard]] inline const SparseEntry<T>* sparse_find_entry(const SparseRow<T>& row,
                                                             size_t column)
{
  const auto found = sparse_lower_bound(row, column);
  if (found == row.end() || found->column != column) return nullptr;
  return &*found;
}

template <typename T, typename NewNonzero>
inline void sparse_eliminate_row(SparseRow<T>& target, const SparseRow<T>& pivot,
                                 size_t pivot_column, const T& factor,
                                 NewNonzero&& on_new_nonzero)
{
  SparseRow<T> merged;
  merged.reserve(target.size() + pivot.size());

  auto target_it = target.begin();
  auto pivot_it = sparse_lower_bound(pivot, pivot_column + 1);
  while (target_it != target.end() || pivot_it != pivot.end()) {
    if (target_it != target.end() && target_it->column == pivot_column) {
      ++target_it;
      continue;
    }

    if (pivot_it == pivot.end() ||
        (target_it != target.end() && target_it->column < pivot_it->column)) {
      merged.push_back(*target_it);
      ++target_it;
      continue;
    }

    if (target_it == target.end() || pivot_it->column < target_it->column) {
      const T value = T(0) - factor * pivot_it->value;
      if (value != T(0)) {
        merged.push_back({pivot_it->column, value});
        on_new_nonzero(pivot_it->column);
      }
      ++pivot_it;
      continue;
    }

    const T value = target_it->value - factor * pivot_it->value;
    if (value != T(0)) merged.push_back({target_it->column, value});
    ++target_it;
    ++pivot_it;
  }
  target = std::move(merged);
}

/// Eliminate an arbitrary pivot column while retaining only columns that are
/// still active. Unlike sparse_eliminate_row, this helper does not assume that
/// the numeric column id is also the elimination order. New and retained
/// incidence callbacks receive both the column and its final position.
template <typename T, typename NewNonzero, typename RemovedNonzero,
          typename RetainedNonzero>
inline void sparse_eliminate_active_row(SparseRow<T>& target, const SparseRow<T>& pivot,
                                        size_t pivot_column,
                                        const std::vector<bool>& active_columns,
                                        const T& factor, NewNonzero&& on_new_nonzero,
                                        RemovedNonzero&& on_removed_nonzero,
                                        RetainedNonzero&& on_retained_nonzero)
{
  SparseRow<T> merged;
  merged.reserve(target.size() + pivot.size());

  auto target_it = target.begin();
  auto pivot_it = pivot.begin();
  const auto skip_inactive = [&](auto& iterator, const auto& end) {
    while (iterator != end &&
           (iterator->column == pivot_column || !active_columns[iterator->column])) {
      ++iterator;
    }
  };
  skip_inactive(target_it, target.end());
  skip_inactive(pivot_it, pivot.end());
  while (target_it != target.end() || pivot_it != pivot.end()) {
    if (pivot_it == pivot.end() ||
        (target_it != target.end() && target_it->column < pivot_it->column)) {
      on_retained_nonzero(target_it->column, merged.size());
      merged.push_back(*target_it++);
      skip_inactive(target_it, target.end());
      continue;
    }
    if (target_it == target.end() || pivot_it->column < target_it->column) {
      const T value = T(0) - factor * pivot_it->value;
      if (value != T(0)) {
        merged.push_back({pivot_it->column, value});
        on_new_nonzero(pivot_it->column, merged.size() - 1);
      }
      ++pivot_it;
      skip_inactive(pivot_it, pivot.end());
      continue;
    }

    const T value = target_it->value - factor * pivot_it->value;
    if (value != T(0)) {
      on_retained_nonzero(target_it->column, merged.size());
      merged.push_back({target_it->column, value});
    } else {
      on_removed_nonzero(target_it->column);
    }
    ++target_it;
    ++pivot_it;
    skip_inactive(target_it, target.end());
    skip_inactive(pivot_it, pivot.end());
  }
  target = std::move(merged);
}

/// Probe-only sparse Gaussian elimination. Columns retain their original
/// priority; within a column the sparsest available row is selected.
template <typename T, bool ReleaseCompletedColumns = false>
[[nodiscard]] EliminationResult
sparse_gaussian_elimination(SparseMatrix<T>& matrix, size_t num_cols,
                            std::vector<T>& right_hand_side, size_t rhs_cols,
                            size_t required_prefix_cols = 0,
                            std::span<const std::uint32_t> row_groups = {},
                            std::span<const std::uint32_t> column_groups = {})
{
  const size_t num_rows = matrix.size();
  if (rhs_cols > 0 && right_hand_side.size() != num_rows * rhs_cols)
    throw std::runtime_error("sparse RHS shape mismatch");
  if (required_prefix_cols > num_cols)
    throw std::runtime_error("required column prefix exceeds matrix width");
  const bool prefer_groups = !row_groups.empty() || !column_groups.empty();
  if (prefer_groups &&
      (row_groups.size() != num_rows || column_groups.size() != num_cols)) {
    throw std::runtime_error("sparse elimination group metadata shape mismatch");
  }
  if (num_rows == 0) return {{}, true, {}};

  MatrixView<T> rhs{right_hand_side.data(), rhs_cols};
  auto rhs_row_nonzero = [&](size_t row) {
    for (size_t column = 0; column < rhs_cols; ++column) {
      if (rhs(row, column) != T(0)) return true;
    }
    return false;
  };
  size_t residual_nonzero_rows = 0;
  for (size_t row = 0; row < num_rows; ++row) {
    if (rhs_row_nonzero(row)) ++residual_nonzero_rows;
  }
  std::vector<std::vector<size_t>> column_rows(num_cols);
  for (size_t row = 0; row < num_rows; ++row) {
    for (const auto& entry : matrix[row]) {
      if (entry.column >= num_cols)
        throw std::runtime_error("sparse matrix column out of range");
      column_rows[entry.column].push_back(row);
    }
  }

  std::vector<bool> used(num_rows, false);
  std::vector<size_t> seen(num_rows, std::numeric_limits<size_t>::max());
  std::vector<size_t> row_map;
  row_map.reserve(num_rows);
  std::vector<size_t> solution_cols;
  std::vector<size_t> candidates;
  size_t cross_group_pivots = 0;

  for (size_t column = 0; column < num_cols; ++column) {
    const size_t generation = column;
    size_t best_row = num_rows;
    size_t best_nonzeros = std::numeric_limits<size_t>::max();
    auto find_best_row = [&](bool require_matching_group) {
      for (const size_t row : column_rows[column]) {
        if (seen[row] == generation) continue;
        if (require_matching_group && row_groups[row] != column_groups[column]) {
          continue;
        }
        seen[row] = generation;
        const T* value = sparse_find(matrix[row], column);
        if (used[row] || value == nullptr || *value == T(0)) continue;
        if (matrix[row].size() < best_nonzeros ||
            (matrix[row].size() == best_nonzeros && row < best_row)) {
          best_row = row;
          best_nonzeros = matrix[row].size();
        }
      }
    };
    if (prefer_groups) find_best_row(true);
    if (best_row == num_rows) {
      find_best_row(false);
      if (prefer_groups && best_row != num_rows &&
          row_groups[best_row] != column_groups[column]) {
        ++cross_group_pivots;
      }
    }
    if (best_row == num_rows) {
      if constexpr (ReleaseCompletedColumns)
        std::vector<size_t>().swap(column_rows[column]);
      if (column + 1 >= required_prefix_cols && residual_nonzero_rows == 0) break;
      continue;
    }

    if (rhs_row_nonzero(best_row)) --residual_nonzero_rows;
    used[best_row] = true;
    row_map.push_back(best_row);
    solution_cols.push_back(column);

    auto& pivot_row = matrix[best_row];
    const T* pivot_value = sparse_find(pivot_row, column);
    const T inverse = T(1) / *pivot_value;
    for (auto& entry : pivot_row) {
      if (entry.column >= column) entry.value = entry.value * inverse;
    }
    for (size_t rhs_column = 0; rhs_column < rhs_cols; ++rhs_column) {
      if (rhs(best_row, rhs_column) != T(0))
        rhs(best_row, rhs_column) = rhs(best_row, rhs_column) * inverse;
    }

    candidates.clear();
    candidates.reserve(column_rows[column].size());
    for (const size_t row : column_rows[column]) {
      if (seen[row] == generation + num_cols) continue;
      seen[row] = generation + num_cols;
      if (!used[row] && sparse_find(matrix[row], column) != nullptr)
        candidates.push_back(row);
    }
    // Rank-only callers can release completed incidence lists without changing
    // the allocation schedule of fixed-basis kernel planning. Candidate row ids
    // are already copied, and the echelon remains available for back substitution.
    if constexpr (ReleaseCompletedColumns)
      std::vector<size_t>().swap(column_rows[column]);
    for (const size_t row : candidates) {
      const T* factor_ptr = sparse_find(matrix[row], column);
      if (factor_ptr == nullptr || *factor_ptr == T(0)) continue;
      const T factor = *factor_ptr;
      const bool rhs_was_nonzero = rhs_row_nonzero(row);
      sparse_eliminate_row(
          matrix[row], pivot_row, column, factor,
          [&](size_t new_column) { column_rows[new_column].push_back(row); });
      for (size_t rhs_column = 0; rhs_column < rhs_cols; ++rhs_column) {
        if (rhs(best_row, rhs_column) != T(0)) {
          rhs(row, rhs_column) =
              rhs(row, rhs_column) - factor * rhs(best_row, rhs_column);
        }
      }
      const bool rhs_is_nonzero = rhs_row_nonzero(row);
      if (rhs_was_nonzero != rhs_is_nonzero) {
        if (rhs_is_nonzero)
          ++residual_nonzero_rows;
        else
          --residual_nonzero_rows;
      }
    }
    if (column + 1 >= required_prefix_cols && residual_nonzero_rows == 0) break;
  }

  for (size_t row = 0; row < num_rows; ++row) {
    if (!used[row]) row_map.push_back(row);
  }

  return {std::move(solution_cols), residual_nonzero_rows == 0, std::move(row_map),
          cross_group_pivots};
}

/// Back-substitute the pivot variables produced by sparse_gaussian_elimination.
/// The returned array is pivot-major and then RHS-major.
template <typename T>
[[nodiscard]] std::vector<T>
back_substitute_sparse_echelon(const SparseMatrix<T>& matrix,
                               const std::vector<T>& right_hand_side, size_t rhs_cols,
                               const std::vector<size_t>& solution_cols,
                               const std::vector<size_t>& row_map)
{
  const size_t num_rows = matrix.size();
  if (rhs_cols > 0 && right_hand_side.size() != num_rows * rhs_cols)
    throw std::runtime_error("sparse echelon RHS shape mismatch");
  if (row_map.size() != num_rows || solution_cols.size() > num_rows)
    throw std::runtime_error("sparse echelon pivot metadata is inconsistent");
  if (solution_cols.empty()) return {};

  const size_t invalid = std::numeric_limits<size_t>::max();
  const size_t max_column = *std::ranges::max_element(solution_cols);
  std::vector<size_t> pivot_position(max_column + 1, invalid);
  for (size_t position = 0; position < solution_cols.size(); ++position) {
    const size_t column = solution_cols[position];
    if (pivot_position[column] != invalid)
      throw std::runtime_error("sparse echelon contains a duplicate pivot column");
    pivot_position[column] = position;
  }

  std::vector<T> solution(solution_cols.size() * rhs_cols, T(0));
  for (size_t reverse = solution_cols.size(); reverse > 0; --reverse) {
    const size_t position = reverse - 1;
    const size_t row = row_map[position];
    if (row >= num_rows)
      throw std::runtime_error("sparse echelon pivot row is out of range");
    for (size_t rhs_column = 0; rhs_column < rhs_cols; ++rhs_column) {
      T value = right_hand_side[row * rhs_cols + rhs_column];
      for (const auto& entry : matrix[row]) {
        if (entry.column >= pivot_position.size()) continue;
        const size_t dependency = pivot_position[entry.column];
        if (dependency != invalid && dependency > position) {
          value = value - entry.value * solution[dependency * rhs_cols + rhs_column];
        }
      }
      solution[position * rhs_cols + rhs_column] = value;
    }
  }
  return solution;
}

/// Tape 录制器的返回结果
struct TapeResult {
  std::vector<Instruction> tape; ///< 指令序列
  std::vector<size_t> perm;      ///< 行置换映射
  std::uint32_t matrix_slot_count = 0;
  std::uint32_t rhs_slot_count = 0;
  struct RowOperation {
    std::uint32_t destination = 0;
    std::uint32_t source = 0;
    bool scale = false;
  };
  std::vector<RowOperation> row_operations;
};

/// Solve a fixed sparse square system directly without recording a Tape.
/// The returned permutation maps logical solution columns to physical RHS rows.
template <typename T>
[[nodiscard]] std::vector<size_t> solve_sparse_system(SparseMatrix<T>& matrix,
                                                      std::vector<T>& right_hand_side,
                                                      size_t rhs_cols)
{
  const size_t dim = matrix.size();
  if (rhs_cols > 0 && right_hand_side.size() != dim * rhs_cols)
    throw std::runtime_error("sparse-solve RHS shape mismatch");

  MatrixView<T> rhs{right_hand_side.data(), rhs_cols};
  std::vector<std::vector<size_t>> column_rows(dim);
  for (size_t row = 0; row < dim; ++row) {
    for (const auto& entry : matrix[row]) {
      if (entry.column >= dim)
        throw std::runtime_error("sparse-solve column out of range");
      column_rows[entry.column].push_back(row);
    }
  }

  std::vector<size_t> permutation(dim);
  std::vector<size_t> position(dim);
  std::iota(permutation.begin(), permutation.end(), size_t{0});
  std::iota(position.begin(), position.end(), size_t{0});
  std::vector<size_t> seen(dim, std::numeric_limits<size_t>::max());
  std::vector<size_t> candidates;

  for (size_t column = 0; column < dim; ++column) {
    size_t best_position = dim;
    size_t best_nonzeros = std::numeric_limits<size_t>::max();
    for (const size_t row : column_rows[column]) {
      const size_t logical = position[row];
      if (logical < column) continue;
      const T* value = sparse_find(matrix[row], column);
      if (value != nullptr && *value != T(0) &&
          (matrix[row].size() < best_nonzeros ||
           (matrix[row].size() == best_nonzeros && logical < best_position))) {
        best_position = logical;
        best_nonzeros = matrix[row].size();
      }
    }
    if (best_position == dim) {
      throw std::runtime_error("sparse solve encountered a zero pivot");
    }

    std::swap(permutation[column], permutation[best_position]);
    position[permutation[column]] = column;
    position[permutation[best_position]] = best_position;
    const size_t pivot_row_id = permutation[column];
    auto& pivot_row = matrix[pivot_row_id];
    const T inverse = T(1) / *sparse_find(pivot_row, column);
    for (auto& entry : pivot_row) {
      if (entry.column > column) entry.value = entry.value * inverse;
    }
    for (size_t rhs_column = 0; rhs_column < rhs_cols; ++rhs_column) {
      rhs(pivot_row_id, rhs_column) = rhs(pivot_row_id, rhs_column) * inverse;
    }

    const size_t generation = column;
    candidates.clear();
    candidates.reserve(column_rows[column].size());
    for (const size_t row : column_rows[column]) {
      if (seen[row] == generation) continue;
      seen[row] = generation;
      if (position[row] > column && sparse_find(matrix[row], column) != nullptr) {
        candidates.push_back(row);
      }
    }
    for (const size_t row : candidates) {
      const T* factor_pointer = sparse_find(matrix[row], column);
      if (factor_pointer == nullptr || *factor_pointer == T(0)) continue;
      const T factor = *factor_pointer;
      sparse_eliminate_row(
          matrix[row], pivot_row, column, factor,
          [&](size_t new_column) { column_rows[new_column].push_back(row); });
      for (size_t rhs_column = 0; rhs_column < rhs_cols; ++rhs_column) {
        rhs(row, rhs_column) =
            rhs(row, rhs_column) - factor * rhs(pivot_row_id, rhs_column);
      }
    }
  }

  for (size_t logical = dim; logical-- > 0;) {
    const size_t pivot_row_id = permutation[logical];
    const size_t generation = dim + logical;
    for (const size_t row : column_rows[logical]) {
      if (seen[row] == generation) continue;
      seen[row] = generation;
      if (position[row] >= logical) continue;
      const T* factor_pointer = sparse_find(matrix[row], logical);
      if (factor_pointer == nullptr || *factor_pointer == T(0)) continue;
      const T factor = *factor_pointer;
      for (size_t rhs_column = 0; rhs_column < rhs_cols; ++rhs_column) {
        rhs(row, rhs_column) =
            rhs(row, rhs_column) - factor * rhs(pivot_row_id, rhs_column);
      }
    }
  }
  return permutation;
}

/// Sparse counterpart of record_tape. Matrix entries carry stable slots;
/// fill-in receives new slots as it is created. RHS uses its compact dense
/// coordinate because dim * rhs_cols is small.
template <typename T>
[[nodiscard]] TapeResult
record_sparse_tape(SparseMatrix<T>& matrix, std::vector<T>& right_hand_side,
                   size_t rhs_cols, std::span<std::uint8_t> rhs_support = {},
                   bool evaluate_rhs = true)
{
  using enum OpCode;
  const size_t dim = matrix.size();
  if (rhs_cols > 0 && right_hand_side.size() != dim * rhs_cols)
    throw std::runtime_error("sparse tape RHS shape mismatch");
  if (!rhs_support.empty() && rhs_support.size() != right_hand_side.size())
    throw std::runtime_error("sparse tape RHS support shape mismatch");
  const auto rhs_is_live = [&](size_t slot) {
    return rhs_support.empty() ? right_hand_side[slot] != T(0) : rhs_support[slot] != 0;
  };
  constexpr std::uint64_t max_slot = Instruction::OP_MASK;
  if (rhs_cols > 0 && dim > max_slot / rhs_cols)
    throw std::runtime_error("sparse tape RHS offset exceeds limit");

  MatrixView<T> rhs{right_hand_side.data(), rhs_cols};
  std::vector<std::vector<size_t>> column_rows(dim);
  std::uint64_t next_matrix_slot = 0;
  for (const auto& row : matrix) {
    for (const auto& entry : row) {
      if (entry.slot != SparseEntry<T>::INVALID_SLOT) {
        next_matrix_slot =
            std::max(next_matrix_slot, static_cast<std::uint64_t>(entry.slot) + 1);
      }
    }
  }
  for (size_t row = 0; row < dim; ++row) {
    for (auto& entry : matrix[row]) {
      if (entry.column >= dim)
        throw std::runtime_error("sparse tape column out of range");
      if (entry.slot == SparseEntry<T>::INVALID_SLOT) {
        if (next_matrix_slot > max_slot)
          throw std::runtime_error("sparse tape matrix slot exceeds limit");
        entry.slot = static_cast<std::uint32_t>(next_matrix_slot++);
      }
      column_rows[entry.column].push_back(row);
    }
  }
  auto allocate_matrix_slot = [&]() {
    if (next_matrix_slot > max_slot)
      throw std::runtime_error("sparse tape matrix slot exceeds limit");
    return static_cast<std::uint32_t>(next_matrix_slot++);
  };

  std::vector<Instruction> tape;
  std::vector<TapeResult::RowOperation> recorded_row_operations;
  std::vector<size_t> permutation(dim);
  std::vector<size_t> position(dim);
  std::iota(permutation.begin(), permutation.end(), size_t{0});
  std::iota(position.begin(), position.end(), size_t{0});
  std::vector<size_t> seen(dim, std::numeric_limits<size_t>::max());
  std::vector<size_t> candidates;

  for (size_t column = 0; column < dim; ++column) {
    size_t best_position = dim;
    size_t best_nonzeros = std::numeric_limits<size_t>::max();
    for (const size_t row : column_rows[column]) {
      const size_t logical = position[row];
      if (logical < column) continue;
      const T* value = sparse_find(matrix[row], column);
      if (value != nullptr && *value != T(0) &&
          (matrix[row].size() < best_nonzeros ||
           (matrix[row].size() == best_nonzeros && logical < best_position))) {
        best_position = logical;
        best_nonzeros = matrix[row].size();
      }
    }
    if (best_position == dim)
      throw std::runtime_error("sparse tape recording encountered a zero pivot");

    std::swap(permutation[column], permutation[best_position]);
    position[permutation[column]] = column;
    position[permutation[best_position]] = best_position;
    const size_t pivot_row_id = permutation[column];
    auto& pivot_row = matrix[pivot_row_id];
    const size_t pivot_rhs = pivot_row_id * rhs_cols;
    const auto* pivot_entry = sparse_find_entry(pivot_row, column);
    tape.push_back(Instruction::pack(Inv, 0, pivot_entry->slot));
    recorded_row_operations.push_back({static_cast<std::uint32_t>(pivot_row_id),
                                       static_cast<std::uint32_t>(pivot_row_id), true});
    const T inverse = T(1) / pivot_entry->value;

    for (auto& entry : pivot_row) {
      if (entry.column <= column) continue;
      tape.push_back(Instruction::pack(MulM, entry.slot, 0));
      entry.value = entry.value * inverse;
    }
    for (size_t rhs_column = 0; rhs_column < rhs_cols; ++rhs_column) {
      if (rhs_is_live(pivot_rhs + rhs_column)) {
        tape.push_back(Instruction::pack(MulB, pivot_rhs + rhs_column, 0));
        if (evaluate_rhs)
          rhs(pivot_row_id, rhs_column) = rhs(pivot_row_id, rhs_column) * inverse;
      }
    }

    const size_t generation = column;
    candidates.clear();
    candidates.reserve(column_rows[column].size());
    for (const size_t row : column_rows[column]) {
      if (seen[row] == generation) continue;
      seen[row] = generation;
      if (position[row] > column && sparse_find(matrix[row], column) != nullptr) {
        candidates.push_back(row);
      }
    }
    for (const size_t row : candidates) {
      const T* factor_ptr = sparse_find(matrix[row], column);
      if (factor_ptr == nullptr || *factor_ptr == T(0)) continue;
      const T factor = *factor_ptr;
      const size_t row_rhs = row * rhs_cols;
      const auto* factor_entry = sparse_find_entry(matrix[row], column);
      tape.push_back(Instruction::pack(LoadF, 0, factor_entry->slot));
      recorded_row_operations.push_back({static_cast<std::uint32_t>(row),
                                         static_cast<std::uint32_t>(pivot_row_id),
                                         false});

      auto& target_row = matrix[row];
      SparseRow<T> merged;
      merged.reserve(target_row.size() + pivot_row.size());
      auto target_it = target_row.begin();
      auto pivot_it = sparse_lower_bound(pivot_row, column + 1);
      while (target_it != target_row.end() || pivot_it != pivot_row.end()) {
        if (target_it != target_row.end() && target_it->column == column) {
          ++target_it;
          continue;
        }
        if (pivot_it == pivot_row.end() ||
            (target_it != target_row.end() && target_it->column < pivot_it->column)) {
          merged.push_back(*target_it++);
          continue;
        }
        if (target_it == target_row.end() || pivot_it->column < target_it->column) {
          const T value = T(0) - factor * pivot_it->value;
          if (value != T(0)) {
            const auto slot = allocate_matrix_slot();
            tape.push_back(Instruction::pack(FmaM, slot, pivot_it->slot));
            merged.emplace_back(pivot_it->column, slot, value);
            column_rows[pivot_it->column].push_back(row);
          }
          ++pivot_it;
          continue;
        }
        tape.push_back(Instruction::pack(FmaM, target_it->slot, pivot_it->slot));
        const T value = target_it->value - factor * pivot_it->value;
        if (value != T(0)) {
          merged.emplace_back(target_it->column, target_it->slot, value);
        }
        ++target_it;
        ++pivot_it;
      }
      target_row = std::move(merged);
      // A row update changes an RHS coordinate only when its source is
      // structurally live; destination-only support does not require an FMA.
      for (size_t rhs_column = 0; rhs_column < rhs_cols; ++rhs_column) {
        const size_t target_slot = row_rhs + rhs_column;
        const size_t source_slot = pivot_rhs + rhs_column;
        if (rhs_is_live(source_slot)) {
          tape.push_back(Instruction::pack(FmaB, target_slot, source_slot));
          if (evaluate_rhs) {
            rhs(row, rhs_column) =
                rhs(row, rhs_column) - factor * rhs(pivot_row_id, rhs_column);
          }
          if (!rhs_support.empty()) rhs_support[target_slot] = 1;
        }
      }
    }
  }

  for (size_t logical = dim; logical-- > 0;) {
    const size_t pivot_row_id = permutation[logical];
    const size_t pivot_rhs = pivot_row_id * rhs_cols;
    const size_t generation = dim + logical;
    for (const size_t row : column_rows[logical]) {
      if (seen[row] == generation) continue;
      seen[row] = generation;
      if (position[row] >= logical) continue;
      const T* factor_ptr = sparse_find(matrix[row], logical);
      if (factor_ptr == nullptr || *factor_ptr == T(0)) continue;
      const T factor = *factor_ptr;
      const size_t row_rhs = row * rhs_cols;
      const auto* factor_entry = sparse_find_entry(matrix[row], logical);
      tape.push_back(Instruction::pack(LoadF, 0, factor_entry->slot));
      recorded_row_operations.push_back({static_cast<std::uint32_t>(row),
                                         static_cast<std::uint32_t>(pivot_row_id),
                                         false});
      for (size_t rhs_column = 0; rhs_column < rhs_cols; ++rhs_column) {
        const size_t target_slot = row_rhs + rhs_column;
        const size_t source_slot = pivot_rhs + rhs_column;
        if (rhs_is_live(source_slot)) {
          tape.push_back(Instruction::pack(FmaB, target_slot, source_slot));
          if (evaluate_rhs) {
            rhs(row, rhs_column) =
                rhs(row, rhs_column) - factor * rhs(pivot_row_id, rhs_column);
          }
          if (!rhs_support.empty()) rhs_support[target_slot] = 1;
        }
      }
    }
  }
  return {std::move(tape), std::move(permutation),
          static_cast<std::uint32_t>(next_matrix_slot),
          static_cast<std::uint32_t>(right_hand_side.size()),
          std::move(recorded_row_operations)};
}

} // namespace linalg
