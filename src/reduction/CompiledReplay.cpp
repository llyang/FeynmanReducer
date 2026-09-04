#include "reduction/CompiledReplay.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace linalg {

CompiledTape compile_tape(std::span<const Instruction> tape)
{
  using enum OpCode;
  CompiledTape compiled;
  compiled.groups.reserve(tape.size() / 4 + 1);
  compiled.operations.reserve(tape.size());
  compiled.rhs_ranges.reserve(tape.size() / 8 + 1);
  std::size_t position = 0;
  while (position < tape.size()) {
    const Instruction header = tape[position++];
    OpCode matrix_opcode;
    OpCode rhs_opcode;
    TapeGroupKind kind;
    if (header.opcode() == Inv) {
      matrix_opcode = MulM;
      rhs_opcode = MulB;
      kind = TapeGroupKind::Scale;
    } else if (header.opcode() == LoadF) {
      matrix_opcode = FmaM;
      rhs_opcode = FmaB;
      kind = TapeGroupKind::Eliminate;
    } else {
      throw std::logic_error("replay tape group has an invalid leading opcode");
    }

    const std::size_t matrix_begin = position;
    while (position < tape.size() && tape[position].opcode() == matrix_opcode) {
      compiled.operations.push_back({tape[position].offset(), tape[position].source()});
      ++position;
    }
    const std::size_t rhs_begin = position;
    const std::size_t compiled_rhs_begin = compiled.operations.size();
    while (position < tape.size() && tape[position].opcode() == rhs_opcode) {
      compiled.operations.push_back({tape[position].offset(), tape[position].source()});
      ++position;
    }
    const std::size_t matrix_count = rhs_begin - matrix_begin;
    const std::size_t rhs_count = position - rhs_begin;
    compiled.logical_rhs_operations += rhs_count;
    if (matrix_count > std::numeric_limits<std::uint32_t>::max() ||
        rhs_count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("replay tape group exceeds 32-bit operation counts");
    }

    const std::size_t range_begin = compiled.rhs_ranges.size();
    for (std::size_t index = rhs_begin; index < position;) {
      const std::uint32_t destination = tape[index].offset();
      const std::uint32_t source = tape[index].source();
      std::size_t range_end = index + 1;
      while (range_end < position &&
             tape[range_end].offset() == destination + (range_end - index) &&
             (rhs_opcode == MulB ||
              tape[range_end].source() == source + (range_end - index))) {
        ++range_end;
      }
      const std::size_t count = range_end - index;
      if (count > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("replay RHS range exceeds 32-bit counts");
      compiled.rhs_ranges.push_back(
          {destination, source, static_cast<std::uint32_t>(count)});
      index = range_end;
    }
    const std::size_t range_count = compiled.rhs_ranges.size() - range_begin;
    if (range_count > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("replay tape group has too many RHS ranges");
    constexpr std::size_t kMinimumAverageRhsRange = 8;
    const bool use_ranges =
        range_count != 0 && range_count <= rhs_count / kMinimumAverageRhsRange;
    if (!use_ranges) {
      compiled.rhs_ranges.resize(range_begin);
    } else {
      compiled.operations.resize(compiled_rhs_begin);
    }
    compiled.groups.push_back(
        {header.source(), static_cast<std::uint32_t>(matrix_count),
         use_ranges ? 0U : static_cast<std::uint32_t>(rhs_count),
         use_ranges ? static_cast<std::uint32_t>(range_count) : 0, kind});
  }
  return compiled;
}

CompactedMatrixTapeLayout
prune_matrix_factor_tape(std::vector<Instruction>& tape,
                         std::uint32_t matrix_slot_count,
                         std::span<const std::uint8_t> required_factor_groups)
{
  using enum OpCode;
  struct GroupRange {
    std::size_t begin;
    std::size_t end;
  };
  std::vector<GroupRange> groups;
  for (std::size_t position = 0; position < tape.size();) {
    const std::size_t begin = position++;
    const auto header = tape[begin].opcode();
    if (header != Inv && header != LoadF)
      throw std::logic_error("matrix factor tape has an invalid group header");
    const auto operation = header == Inv ? MulM : FmaM;
    while (position < tape.size() && tape[position].opcode() == operation)
      ++position;
    if (position < tape.size() &&
        (tape[position].opcode() == MulB || tape[position].opcode() == FmaB)) {
      throw std::logic_error("matrix factor tape contains RHS operations");
    }
    groups.push_back({begin, position});
  }
  if (groups.size() != required_factor_groups.size())
    throw std::logic_error("matrix factor support shape is inconsistent");

  std::vector<std::uint8_t> live_matrix(matrix_slot_count, 0);
  std::vector<std::uint8_t> keep(tape.size(), 0);
  for (std::size_t reverse = groups.size(); reverse > 0; --reverse) {
    const std::size_t group = reverse - 1;
    const auto range = groups[group];
    bool factor_needed = required_factor_groups[group] != 0;
    for (std::size_t position = range.end; position-- > range.begin + 1;) {
      const auto instruction = tape[position];
      if (instruction.offset() >= live_matrix.size() ||
          (instruction.opcode() == FmaM &&
           instruction.source() >= live_matrix.size())) {
        throw std::logic_error("matrix factor tape slot is out of range");
      }
      if (live_matrix[instruction.offset()] == 0) continue;
      keep[position] = 1;
      factor_needed = true;
      if (instruction.opcode() == FmaM) live_matrix[instruction.source()] = 1;
    }
    if (!factor_needed) continue;
    const auto source = tape[range.begin].source();
    if (source >= live_matrix.size())
      throw std::logic_error("matrix factor source is out of range");
    keep[range.begin] = 1;
    live_matrix[source] = 1;
  }

  CompactedMatrixTapeLayout layout;
  layout.matrix_remap.assign(matrix_slot_count, CompactedTapeLayout::INVALID_SLOT);
  layout.group_remap.assign(groups.size(), CompactedTapeLayout::INVALID_SLOT);
  for (std::size_t slot = 0; slot < live_matrix.size(); ++slot) {
    if (live_matrix[slot] != 0) layout.matrix_remap[slot] = layout.matrix_size++;
  }
  std::uint32_t next_group = 0;
  std::size_t write = 0;
  for (std::size_t group = 0; group < groups.size(); ++group) {
    const auto range = groups[group];
    if (keep[range.begin] == 0) continue;
    layout.group_remap[group] = next_group++;
    for (std::size_t position = range.begin; position < range.end; ++position) {
      if (keep[position] == 0) continue;
      auto instruction = tape[position];
      const auto mapped = [&](std::uint32_t slot) {
        if (slot >= layout.matrix_remap.size() ||
            layout.matrix_remap[slot] == CompactedTapeLayout::INVALID_SLOT) {
          throw std::logic_error("matrix factor tape references a dead slot");
        }
        return layout.matrix_remap[slot];
      };
      if (instruction.opcode() == Inv || instruction.opcode() == LoadF) {
        instruction =
            Instruction::pack(instruction.opcode(), 0, mapped(instruction.source()));
      } else if (instruction.opcode() == MulM) {
        instruction = Instruction::pack(MulM, mapped(instruction.offset()), 0);
      } else if (instruction.opcode() == FmaM) {
        instruction = Instruction::pack(FmaM, mapped(instruction.offset()),
                                        mapped(instruction.source()));
      } else {
        throw std::logic_error("matrix factor tape has an invalid operation");
      }
      tape[write++] = instruction;
    }
  }
  tape.resize(write);
  return layout;
}

CompactedTapeLayout
prune_and_compact_tape(std::vector<Instruction>& tape, std::uint32_t matrix_slot_count,
                       std::uint32_t rhs_slot_count,
                       const std::vector<std::size_t>& required_rhs_outputs)
{
  using enum OpCode;
  std::vector<bool> alive_matrix(matrix_slot_count, false);
  std::vector<bool> alive_rhs(rhs_slot_count, false);
  for (const std::size_t slot : required_rhs_outputs) {
    if (slot >= rhs_slot_count)
      throw std::runtime_error("required RHS slot is out of range");
    alive_rhs[slot] = true;
  }

  bool register_alive = false;
  std::size_t write = tape.size();
  for (std::size_t read = tape.size(); read-- > 0;) {
    const Instruction instruction = tape[read];
    const auto opcode = instruction.opcode();
    const std::size_t destination = instruction.offset();
    const std::size_t source = instruction.source();
    bool keep = false;
    switch (opcode) {
    case FmaB:
      if (alive_rhs[destination]) {
        alive_rhs[source] = true;
        register_alive = true;
        keep = true;
      }
      break;
    case FmaM:
      if (alive_matrix[destination]) {
        alive_matrix[source] = true;
        register_alive = true;
        keep = true;
      }
      break;
    case MulB:
      if (alive_rhs[destination]) {
        register_alive = true;
        keep = true;
      }
      break;
    case MulM:
      if (alive_matrix[destination]) {
        register_alive = true;
        keep = true;
      }
      break;
    case Inv:
    case LoadF:
      if (register_alive) {
        alive_matrix[source] = true;
        register_alive = false;
        keep = true;
      }
      break;
    }
    if (keep) tape[--write] = instruction;
  }
  std::move(tape.begin() + static_cast<std::ptrdiff_t>(write), tape.end(),
            tape.begin());
  tape.resize(tape.size() - write);

  CompactedTapeLayout layout;
  layout.matrix_remap.assign(matrix_slot_count, CompactedTapeLayout::INVALID_SLOT);
  layout.rhs_remap.assign(rhs_slot_count, CompactedTapeLayout::INVALID_SLOT);
  for (std::size_t slot = 0; slot < alive_matrix.size(); ++slot) {
    if (alive_matrix[slot]) layout.matrix_remap[slot] = layout.matrix_size++;
  }
  for (std::size_t slot = 0; slot < alive_rhs.size(); ++slot) {
    if (alive_rhs[slot]) layout.rhs_remap[slot] = layout.rhs_size++;
  }

  const auto mapped = [](const auto& remap, std::size_t slot) {
    if (slot >= remap.size() || remap[slot] == CompactedTapeLayout::INVALID_SLOT)
      throw std::runtime_error("missing compact tape slot");
    return remap[slot];
  };
  for (auto& instruction : tape) {
    const auto opcode = instruction.opcode();
    switch (opcode) {
    case Inv:
    case LoadF:
      instruction = Instruction::pack(
          opcode, 0, mapped(layout.matrix_remap, instruction.source()));
      break;
    case MulM:
      instruction = Instruction::pack(
          opcode, mapped(layout.matrix_remap, instruction.offset()), 0);
      break;
    case FmaM:
      instruction =
          Instruction::pack(opcode, mapped(layout.matrix_remap, instruction.offset()),
                            mapped(layout.matrix_remap, instruction.source()));
      break;
    case MulB:
      instruction =
          Instruction::pack(opcode, mapped(layout.rhs_remap, instruction.offset()), 0);
      break;
    case FmaB:
      instruction =
          Instruction::pack(opcode, mapped(layout.rhs_remap, instruction.offset()),
                            mapped(layout.rhs_remap, instruction.source()));
      break;
    }
  }
  return layout;
}

} // namespace linalg
