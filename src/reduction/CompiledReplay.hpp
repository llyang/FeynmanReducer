#pragma once

#include "reduction/EliminationTape.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace linalg {

enum class TapeGroupKind : std::uint32_t { Scale, Eliminate };

struct TapeGroup {
  std::uint32_t source = 0;
  std::uint32_t matrix_operations = 0;
  std::uint32_t rhs_operations = 0;
  std::uint32_t rhs_ranges = 0;
  TapeGroupKind kind = TapeGroupKind::Scale;

  bool operator==(const TapeGroup&) const = default;
};
static_assert(sizeof(TapeGroup) == 20);

struct TapeOperation {
  std::uint32_t destination = 0;
  std::uint32_t source = 0;

  bool operator==(const TapeOperation&) const = default;
};
static_assert(sizeof(TapeOperation) == 8);

struct TapeRhsRange {
  std::uint32_t destination = 0;
  std::uint32_t source = 0;
  std::uint32_t count = 0;

  bool operator==(const TapeRhsRange&) const = default;
};
static_assert(sizeof(TapeRhsRange) == 12);

struct CompiledTape {
  std::vector<TapeGroup> groups;
  std::vector<TapeOperation> operations;
  std::vector<TapeRhsRange> rhs_ranges;
  std::size_t logical_rhs_operations = 0;

  bool operator==(const CompiledTape&) const = default;
};

[[nodiscard]] CompiledTape compile_tape(std::span<const Instruction> tape);

template <typename Arithmetic>
[[nodiscard]] bool execute_matrix_tape_capture(std::vector<std::uint64_t>& matrix,
                                               const CompiledTape& tape,
                                               std::vector<std::uint64_t>& factors,
                                               const Arithmetic& arithmetic)
{
  if (!tape.rhs_ranges.empty() || tape.logical_rhs_operations != 0)
    throw std::logic_error("matrix factor tape contains RHS operations");
  if (factors.size() < tape.groups.size()) factors.resize(tape.groups.size());
  std::size_t operation_index = 0;
  for (std::size_t group_index = 0; group_index < tape.groups.size(); ++group_index) {
    const TapeGroup& group = tape.groups[group_index];
    std::uint64_t factor = 0;
    if (group.kind == TapeGroupKind::Scale) {
      if (!arithmetic.inverse(matrix[group.source], factor)) return false;
      factors[group_index] = factor;
      const std::size_t end = operation_index + group.matrix_operations;
      while (operation_index != end) {
        const auto operation = tape.operations[operation_index++];
        matrix[operation.destination] =
            arithmetic.multiply(matrix[operation.destination], factor);
      }
    } else {
      factor = matrix[group.source];
      factors[group_index] = factor;
      const std::size_t end = operation_index + group.matrix_operations;
      while (operation_index != end) {
        const auto operation = tape.operations[operation_index++];
        matrix[operation.destination] = arithmetic.subtract_multiply(
            matrix[operation.destination], factor, matrix[operation.source]);
      }
    }
  }
  if (operation_index != tape.operations.size())
    throw std::logic_error("matrix factor tape does not cover its operands");
  return true;
}

// Replays typed groups without per-instruction opcode dispatch. A false result
// means that the finite-field point produced a zero pivot.
template <typename Arithmetic>
[[nodiscard]] bool execute_tape_raw(std::vector<std::uint64_t>& matrix,
                                    std::vector<std::uint64_t>& right_hand_side,
                                    const CompiledTape& tape,
                                    const Arithmetic& arithmetic)
{
  std::uint64_t reg = 0;
  const TapeOperation* operation = tape.operations.data();
  const TapeOperation* const operation_end = operation + tape.operations.size();
  const TapeRhsRange* rhs_range = tape.rhs_ranges.data();
  const TapeRhsRange* const rhs_range_end = rhs_range + tape.rhs_ranges.size();
  for (const TapeGroup& group : tape.groups) {
    if (group.kind == TapeGroupKind::Scale) {
      if (!arithmetic.inverse(matrix[group.source], reg)) [[unlikely]]
        return false;
      const TapeOperation* const matrix_operation_end =
          operation + group.matrix_operations;
      while (operation != matrix_operation_end) {
        matrix[operation->destination] =
            arithmetic.multiply(matrix[operation->destination], reg);
        ++operation;
      }
      const TapeOperation* const rhs_operation_end = operation + group.rhs_operations;
      if (group.rhs_ranges == 0) {
        while (operation != rhs_operation_end) {
          right_hand_side[operation->destination] =
              arithmetic.multiply(right_hand_side[operation->destination], reg);
          ++operation;
        }
        continue;
      }
      operation = rhs_operation_end;
      const TapeRhsRange* const group_rhs_range_end = rhs_range + group.rhs_ranges;
      while (rhs_range != group_rhs_range_end) {
        auto* destination = right_hand_side.data() + rhs_range->destination;
        auto* const destination_end = destination + rhs_range->count;
        while (destination != destination_end) {
          *destination = arithmetic.multiply(*destination, reg);
          ++destination;
        }
        ++rhs_range;
      }
      continue;
    }

    reg = matrix[group.source];
    const TapeOperation* const matrix_operation_end =
        operation + group.matrix_operations;
    while (operation != matrix_operation_end) {
      matrix[operation->destination] = arithmetic.subtract_multiply(
          matrix[operation->destination], reg, matrix[operation->source]);
      ++operation;
    }
    const TapeOperation* const rhs_operation_end = operation + group.rhs_operations;
    if (group.rhs_ranges == 0) {
      while (operation != rhs_operation_end) {
        right_hand_side[operation->destination] =
            arithmetic.subtract_multiply(right_hand_side[operation->destination], reg,
                                         right_hand_side[operation->source]);
        ++operation;
      }
      continue;
    }
    operation = rhs_operation_end;
    const TapeRhsRange* const group_rhs_range_end = rhs_range + group.rhs_ranges;
    while (rhs_range != group_rhs_range_end) {
      auto* destination = right_hand_side.data() + rhs_range->destination;
      const auto* source = right_hand_side.data() + rhs_range->source;
      auto* const destination_end = destination + rhs_range->count;
      while (destination != destination_end) {
        *destination = arithmetic.subtract_multiply(*destination, reg, *source);
        ++destination;
        ++source;
      }
      ++rhs_range;
    }
  }
  if (operation != operation_end || rhs_range != rhs_range_end) [[unlikely]] {
    throw std::logic_error("compiled replay metadata does not cover its operands");
  }
  return true;
}

struct CompactedTapeLayout {
  static constexpr std::uint32_t INVALID_SLOT =
      std::numeric_limits<std::uint32_t>::max();

  std::vector<std::uint32_t> matrix_remap;
  std::vector<std::uint32_t> rhs_remap;
  std::uint32_t matrix_size = 0;
  std::uint32_t rhs_size = 0;
};

struct CompactedMatrixTapeLayout {
  std::vector<std::uint32_t> matrix_remap;
  std::vector<std::uint32_t> group_remap;
  std::uint32_t matrix_size = 0;
};

[[nodiscard]] CompactedMatrixTapeLayout
prune_matrix_factor_tape(std::vector<Instruction>& tape,
                         std::uint32_t matrix_slot_count,
                         std::span<const std::uint8_t> required_factor_groups);

[[nodiscard]] CompactedTapeLayout
prune_and_compact_tape(std::vector<Instruction>& tape, std::uint32_t matrix_slot_count,
                       std::uint32_t rhs_slot_count,
                       const std::vector<std::size_t>& required_rhs_outputs);

} // namespace linalg
