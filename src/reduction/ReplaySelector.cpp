#include "reduction/BlackBoxFeynman.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

void BlackBoxFeynman::complete_coefficient_selection(
    CoefficientSelection& selection) const
{
  const auto sort_unique = [](auto& values) {
    std::ranges::sort(values);
    values.erase(std::unique(values.begin(), values.end()), values.end());
  };
  sort_unique(selection.top_lp_expressions);
  sort_unique(selection.pooled_expressions);
  sort_unique(selection.targets);
  sort_unique(selection.basis);

  selection.lp_programs.clear();
  selection.lp_reciprocals.clear();
  selection.top_lp_product_nodes.clear();
  selection.extended_polynomial_factors.clear();
  selection.maximum_falling_degree = 0;
  selection.maximum_positive_lp_delta = 0;
  selection.maximum_negative_lp_delta = 0;
  for (const std::uint32_t target : selection.targets) {
    if (target >= target_lp_program_ids.size())
      throw std::logic_error("selected target normalization is out of range");
    const std::size_t program = target_lp_program_ids[target];
    selection.lp_programs.push_back(program);
    selection.lp_reciprocals.push_back(
        {program, CoefficientSelection::LpReciprocalSide::TargetDenominator});
  }
  for (const std::uint32_t master : selection.basis) {
    if (master >= basis_lp_program_ids.size())
      throw std::logic_error("selected basis normalization is out of range");
    const std::size_t program = basis_lp_program_ids[master];
    selection.lp_programs.push_back(program);
    selection.lp_reciprocals.push_back(
        {program, CoefficientSelection::LpReciprocalSide::BasisNumerator});
  }
  sort_unique(selection.lp_programs);
  sort_unique(selection.lp_reciprocals);
  for (const std::size_t program_index : selection.lp_programs) {
    if (program_index >= lp_programs.size())
      throw std::logic_error("selected LP normalization program is out of range");
    const auto delta = lp_programs[program_index].delta;
    if (delta > 0) {
      selection.maximum_positive_lp_delta = std::max(
          selection.maximum_positive_lp_delta, static_cast<std::size_t>(delta));
    } else if (delta < 0) {
      if (delta == std::numeric_limits<std::int64_t>::min())
        throw std::overflow_error("negative LP delta exceeds size_t");
      selection.maximum_negative_lp_delta = std::max(
          selection.maximum_negative_lp_delta, static_cast<std::size_t>(-delta));
    }
  }

  for (const std::uint32_t expression : selection.top_lp_expressions) {
    if (expression >= top_lp_expression_programs.size())
      throw std::logic_error("selected top-LP expression is out of range");
    const auto& program = top_lp_expression_programs[expression];
    const std::size_t end =
        static_cast<std::size_t>(program.term_begin) + program.term_count;
    for (std::size_t term = program.term_begin; term < end; ++term) {
      const auto& atom = top_lp_expression_terms[term];
      selection.maximum_falling_degree =
          std::max(selection.maximum_falling_degree, atom.falling_degree);
      std::uint32_t product = atom.product;
      while (product != 0) {
        const std::uint32_t node = product - 1;
        if (node >= top_lp_product_nodes.size())
          throw std::logic_error("selected top-LP product is out of range");
        selection.top_lp_product_nodes.push_back(node);
        selection.extended_polynomial_factors.push_back(
            top_lp_product_nodes[node].polynomial_factor);
        product = top_lp_product_nodes[node].parent;
      }
    }
  }
  sort_unique(selection.top_lp_product_nodes);
  sort_unique(selection.extended_polynomial_factors);
}

std::shared_ptr<const BlackBoxFeynman::ReplayVariant>
BlackBoxFeynman::selected_replay_variant(
    std::span<const std::uint32_t> active_outputs) const
{
  auto current = replay_variant.load(std::memory_order_acquire);
  if (current != nullptr &&
      std::ranges::equal(current->active_outputs, active_outputs)) {
    return current;
  }

  // A cache miss must not keep an obsolete variant alive while waiting for
  // another builder. Reload the published variant after acquiring the lock.
  current.reset();
  std::lock_guard lock(replay_variant_mutex);
  current = replay_variant.load(std::memory_order_relaxed);
  if (current != nullptr &&
      std::ranges::equal(current->active_outputs, active_outputs)) {
    return current;
  }

  auto variant = build_replay_variant(active_outputs);
  replay_variant.store(variant, std::memory_order_release);
  return variant;
}

std::shared_ptr<BlackBoxFeynman::ReplayVariant> BlackBoxFeynman::build_replay_variant(
    std::span<const std::uint32_t> active_outputs) const
{
  auto variant = std::make_shared<ReplayVariant>();
  variant->active_outputs.assign(active_outputs.begin(), active_outputs.end());
  variant->outputs.reserve(active_outputs.size());
  for (const std::uint32_t output : active_outputs)
    variant->outputs.push_back(replay_outputs[output]);

  if (replay_orientation == ReplayOrientation::Master) {
    variant->matrix_workspace_size = matrix_workspace_size;
    variant->rhs_workspace_size = rhs_workspace_size;
    for (const auto& output : variant->outputs) {
      const auto found = std::ranges::lower_bound(master_rhs_columns, output.basis);
      if (found == master_rhs_columns.end() || *found != output.basis)
        throw std::logic_error("selected master output basis is unavailable");
      variant->master_columns.push_back(
          static_cast<std::uint32_t>(found - master_rhs_columns.begin()));
    }
    std::ranges::sort(variant->master_columns);
    variant->master_columns.erase(
        std::unique(variant->master_columns.begin(), variant->master_columns.end()),
        variant->master_columns.end());

    // Keep the published dense program untouched when every master column is
    // active. Sparse variants intersect its ranges with the selected columns
    // once, so the hot replay loop has no per-column mask branch.
    if (variant->master_columns.size() == master_rhs_columns.size()) {
      variant->coefficients = replay_coefficients;
      // The dense algebra program may stay shared, but LP normalization is an
      // output dependency: evaluating an inactive target's reciprocal can
      // spuriously reject an otherwise regular selected probe.
      variant->coefficients.targets.clear();
      variant->coefficients.basis.clear();
      for (const auto& output : variant->outputs) {
        variant->coefficients.targets.push_back(output.target);
        variant->coefficients.basis.push_back(output.basis);
      }
      complete_coefficient_selection(variant->coefficients);
      return variant;
    }

    const auto merge_clear_runs = [](std::vector<ContiguousSlotRun> runs) {
      std::ranges::sort(runs, {}, &ContiguousSlotRun::begin);
      std::vector<ContiguousSlotRun> merged;
      merged.reserve(runs.size());
      for (const auto run : runs) {
        if (merged.empty() ||
            static_cast<std::uint64_t>(merged.back().begin) + merged.back().count <
                run.begin) {
          merged.push_back(run);
          continue;
        }
        const std::uint64_t end = std::max(
            static_cast<std::uint64_t>(merged.back().begin) + merged.back().count,
            static_cast<std::uint64_t>(run.begin) + run.count);
        if (end > std::numeric_limits<std::uint32_t>::max())
          throw std::logic_error("selected master clear range exceeds 32-bit slots");
        merged.back().count = static_cast<std::uint32_t>(end - merged.back().begin);
      }
      return merged;
    };
    const auto append_clear_run = [](std::vector<ContiguousSlotRun>& runs,
                                     std::uint32_t begin, std::uint32_t count) {
      if (count != 0) runs.push_back({begin, count});
    };

    auto& selected = variant->selected_master.emplace();
    std::vector<ContiguousSlotRun> output_rhs_clear_runs;
    std::vector<ContiguousSlotRun> target_clear_runs;
    std::vector<std::uint8_t> active_target_slots(target_rhs_workspace_size, 0);
    for (const auto& output : variant->outputs) {
      if (output.contraction_begin > output.contraction_end ||
          output.contraction_end > master_contractions.size()) {
        throw std::logic_error("selected master contraction range is invalid");
      }
      for (std::size_t position = output.contraction_begin;
           position < output.contraction_end; ++position) {
        const auto contraction = master_contractions[position];
        if (contraction.master_slot >= rhs_workspace_size ||
            contraction.target_slot >= target_rhs_workspace_size) {
          throw std::logic_error("selected master contraction slot is out of range");
        }
        append_clear_run(output_rhs_clear_runs, contraction.master_slot, 1);
        append_clear_run(target_clear_runs, contraction.target_slot, 1);
        active_target_slots[contraction.target_slot] = 1;
      }
    }

    std::erase_if(selected_master_structure_cache,
                  [](const auto& structure) { return structure.expired(); });
    std::shared_ptr<const SelectedMasterStructure> cached_structure;
    bool live_structures_are_comparable = true;
    for (const auto& weak_structure : selected_master_structure_cache) {
      auto structure = weak_structure.lock();
      if (structure == nullptr) continue;
      if (std::ranges::equal(structure->master_columns, variant->master_columns))
        cached_structure = structure;
      if (!std::ranges::includes(structure->master_columns, variant->master_columns) &&
          !std::ranges::includes(variant->master_columns, structure->master_columns)) {
        live_structures_are_comparable = false;
      }
    }
    if (cached_structure == nullptr) {
      if (!live_structures_are_comparable) {
        selected_master_structure_cache.clear();
      }

      auto structure = std::make_shared<SelectedMasterStructure>();
      structure->master_columns = variant->master_columns;
      structure->blocks.reserve(replay_blocks.size());
      std::vector<ContiguousSlotRun> structure_clear_runs;
      for (const std::uint32_t column : structure->master_columns) {
        if (column >= master_seed_slots.size())
          throw std::logic_error("selected master seed column is out of range");
        const std::uint32_t slot = master_seed_slots[column];
        if (slot == linalg::CompactedTapeLayout::INVALID_SLOT) continue;
        if (slot >= rhs_workspace_size)
          throw std::logic_error("selected master seed slot is out of range");
        append_clear_run(structure_clear_runs, slot, 1);
      }

      const auto append_active_ranges = [&](const MasterReplayProgram& source,
                                            std::uint32_t range_begin,
                                            std::uint32_t range_end,
                                            std::vector<MasterSlotRange>& ranges) {
        if (range_begin > range_end || range_end > source.slot_ranges.size()) {
          throw std::logic_error("selected master slot range is invalid");
        }
        if (ranges.size() > std::numeric_limits<std::uint32_t>::max()) {
          throw std::logic_error("selected master range count exceeds 32-bit slots");
        }
        const auto selected_begin = static_cast<std::uint32_t>(ranges.size());
        for (std::size_t range_index = range_begin; range_index < range_end;
             ++range_index) {
          const auto source_range = source.slot_ranges[range_index];
          const std::size_t column_end =
              static_cast<std::size_t>(source_range.column_begin) + source_range.count;
          auto active = std::ranges::lower_bound(structure->master_columns,
                                                 source_range.column_begin);
          while (active != structure->master_columns.end() && *active < column_end) {
            const std::uint32_t run_column = *active;
            std::uint32_t run_count = 1;
            std::uint32_t previous = *active;
            ++active;
            while (active != structure->master_columns.end() && *active < column_end &&
                   *active == previous + 1) {
              previous = *active;
              ++run_count;
              ++active;
            }
            const std::uint32_t offset = run_column - source_range.column_begin;
            const std::uint64_t destination =
                static_cast<std::uint64_t>(source_range.destination) + offset;
            const std::uint64_t source_slot =
                static_cast<std::uint64_t>(source_range.source) + offset;
            if (destination + run_count > rhs_workspace_size ||
                source_slot + run_count > rhs_workspace_size ||
                ranges.size() >= std::numeric_limits<std::uint32_t>::max()) {
              throw std::logic_error("selected master active run is out of range");
            }
            ranges.push_back({static_cast<std::uint32_t>(destination),
                              static_cast<std::uint32_t>(source_slot), run_column,
                              run_count});
            append_clear_run(structure_clear_runs,
                             static_cast<std::uint32_t>(destination), run_count);
            append_clear_run(structure_clear_runs,
                             static_cast<std::uint32_t>(source_slot), run_count);
          }
        }
        return std::pair{selected_begin, static_cast<std::uint32_t>(ranges.size())};
      };

      for (const auto& source_block : replay_blocks) {
        if (source_block.master == nullptr)
          throw std::logic_error("selected master replay program is unavailable");
        const auto& source = *source_block.master;
        SelectedMasterBlock block;
        block.couplings.resize(source.couplings.size());
        for (std::size_t coupling = 0; coupling < source.couplings.size(); ++coupling) {
          const auto original = source.couplings[coupling];
          const auto ranges = append_active_ranges(
              source, original.range_begin, original.range_end, block.slot_ranges);
          block.couplings[coupling] = {ranges.first, ranges.second};
        }
        block.operations.reserve(source.operations.size());
        for (const auto operation : source.operations) {
          const auto ranges = append_active_ranges(
              source, operation.range_begin, operation.range_end, block.slot_ranges);
          if (ranges.first == ranges.second) continue;
          block.operations.push_back(
              {operation.factor_group, ranges.first, ranges.second, operation.kind});
        }

        const auto coupling_is_active = [&](std::size_t coupling) {
          if (coupling >= block.couplings.size())
            throw std::logic_error("selected master coupling is out of range");
          return block.couplings[coupling].range_begin !=
                 block.couplings[coupling].range_end;
        };
        block.loader_couplings.reserve(source_block.loader_couplings.size());
        for (std::size_t begin = 0; begin < source_block.loader_couplings.size();) {
          const std::size_t group_size =
              source_block.loader_couplings[begin].group_size;
          const std::size_t end = begin + group_size;
          if (group_size == 0 || end > source_block.loader_couplings.size()) {
            throw std::logic_error(
                "selected master encountered an invalid coupling loader group");
          }
          if (coupling_is_active(source_block.loader_couplings[begin].flat_idx)) {
            block.loader_couplings.insert(block.loader_couplings.end(),
                                          source_block.loader_couplings.begin() +
                                              static_cast<std::ptrdiff_t>(begin),
                                          source_block.loader_couplings.begin() +
                                              static_cast<std::ptrdiff_t>(end));
          }
          begin = end;
        }
        for (const auto instruction : source_block.pooled_loader_couplings) {
          if (!coupling_is_active(instruction.flat_idx)) continue;
          block.pooled_loader_couplings.push_back(instruction);
          structure->coefficient_seed.pooled_expressions.push_back(
              instruction.expression);
        }
        for (const auto instruction : source_block.top_lp_loader_couplings) {
          if (!coupling_is_active(instruction.flat_idx)) continue;
          block.top_lp_loader_couplings.push_back(instruction);
          structure->coefficient_seed.top_lp_expressions.push_back(
              instruction.expression);
        }
        for (const auto instruction : source_block.pooled_loader_M)
          structure->coefficient_seed.pooled_expressions.push_back(
              instruction.expression);
        for (const auto instruction : source_block.top_lp_loader_M)
          structure->coefficient_seed.top_lp_expressions.push_back(
              instruction.expression);
        structure->blocks.push_back(std::move(block));
      }
      structure->rhs_clear_runs = merge_clear_runs(std::move(structure_clear_runs));
      selected_master_structure_cache.push_back(structure);
      cached_structure = std::move(structure);
    }
    selected.structure = std::move(cached_structure);
    variant->coefficients = selected.structure->coefficient_seed;

    variant->loader_B.reserve(loader_B.size());
    for (std::size_t begin = 0; begin < loader_B.size();) {
      const std::size_t group_size = loader_B[begin].group_size;
      const std::size_t end = begin + group_size;
      if (group_size == 0 || end > loader_B.size()) {
        throw std::logic_error(
            "selected master encountered an invalid RHS loader group");
      }
      const std::size_t slot = loader_B[begin].flat_idx;
      if (slot >= active_target_slots.size())
        throw std::logic_error("selected master RHS loader is out of range");
      if (active_target_slots[slot] != 0) {
        variant->loader_B.insert(variant->loader_B.end(),
                                 loader_B.begin() + static_cast<std::ptrdiff_t>(begin),
                                 loader_B.begin() + static_cast<std::ptrdiff_t>(end));
      }
      begin = end;
    }
    for (const auto instruction : pooled_loader_B) {
      if (instruction.flat_idx >= active_target_slots.size())
        throw std::logic_error("selected master pooled RHS is out of range");
      if (active_target_slots[instruction.flat_idx] == 0) continue;
      variant->pooled_loader_B.push_back(instruction);
      variant->coefficients.pooled_expressions.push_back(instruction.expression);
    }
    for (const auto instruction : top_lp_loader_B) {
      if (instruction.flat_idx >= active_target_slots.size())
        throw std::logic_error("selected master top-LP RHS is out of range");
      if (active_target_slots[instruction.flat_idx] == 0) continue;
      variant->top_lp_loader_B.push_back(instruction);
      variant->coefficients.top_lp_expressions.push_back(instruction.expression);
    }

    selected.rhs_clear_runs = merge_clear_runs(std::move(output_rhs_clear_runs));
    selected.target_clear_runs = merge_clear_runs(std::move(target_clear_runs));
    for (const auto& output : variant->outputs) {
      variant->coefficients.targets.push_back(output.target);
      variant->coefficients.basis.push_back(output.basis);
    }
    complete_coefficient_selection(variant->coefficients);
    return variant;
  }

  if (rhs_workspace_size > std::numeric_limits<std::uint32_t>::max())
    throw std::logic_error("selected replay RHS workspace exceeds 32-bit slots");
  std::vector<std::uint8_t> live_global_rhs(rhs_workspace_size, 0);
  variant->blocks.reserve(replay_blocks.size());

  auto filter_matrix_loader = [](const std::vector<LoadInstruction>& source,
                                 const auto& remap,
                                 std::vector<LoadInstruction>& result) {
    result.reserve(source.size());
    for (std::size_t begin = 0; begin < source.size();) {
      const std::size_t group_size = source[begin].group_size;
      const std::size_t end = begin + group_size;
      if (group_size == 0 || end > source.size())
        throw std::logic_error("selected replay encountered an invalid loader group");
      const std::size_t slot = source[begin].flat_idx;
      if (slot < remap.size() &&
          remap[slot] != linalg::CompactedTapeLayout::INVALID_SLOT) {
        for (std::size_t index = begin; index < end; ++index) {
          auto instruction = source[index];
          instruction.flat_idx = remap[slot];
          result.push_back(instruction);
        }
      }
      begin = end;
    }
  };
  auto filter_pooled_matrix_loader = [](const auto& source, const auto& remap) {
    std::vector<PooledCoefficientLoad> result;
    result.reserve(source.size());
    for (auto instruction : source) {
      if (instruction.flat_idx >= remap.size() ||
          remap[instruction.flat_idx] == linalg::CompactedTapeLayout::INVALID_SLOT) {
        continue;
      }
      instruction.flat_idx = remap[instruction.flat_idx];
      result.push_back(instruction);
    }
    return result;
  };
  auto filter_top_lp_matrix_loader = [](const auto& source, const auto& remap) {
    std::vector<TopLpLoadInstruction> result;
    result.reserve(source.size());
    for (auto instruction : source) {
      if (instruction.flat_idx >= remap.size() ||
          remap[instruction.flat_idx] == linalg::CompactedTapeLayout::INVALID_SLOT) {
        continue;
      }
      instruction.flat_idx = remap[instruction.flat_idx];
      result.push_back(instruction);
    }
    return result;
  };
  auto filter_coupling_loader = [](const std::vector<LoadInstruction>& source,
                                   const auto& couplings,
                                   std::vector<LoadInstruction>& result) {
    result.reserve(source.size());
    for (std::size_t begin = 0; begin < source.size();) {
      const std::size_t group_size = source[begin].group_size;
      const std::size_t end = begin + group_size;
      if (group_size == 0 || end > source.size())
        throw std::logic_error("selected replay encountered an invalid coupling group");
      const std::size_t coupling = source[begin].flat_idx;
      if (coupling >= couplings.size())
        throw std::logic_error("selected replay coupling loader is out of range");
      if (couplings[coupling].target_begin != couplings[coupling].target_end) {
        result.insert(result.end(), source.begin() + static_cast<ptrdiff_t>(begin),
                      source.begin() + static_cast<ptrdiff_t>(end));
      }
      begin = end;
    }
  };
  auto filter_pooled_coupling_loader = [](const auto& source, const auto& couplings) {
    std::vector<PooledCoefficientLoad> result;
    result.reserve(source.size());
    for (const auto& instruction : source) {
      if (instruction.flat_idx >= couplings.size())
        throw std::logic_error("selected replay pooled coupling is out of range");
      const auto& coupling = couplings[instruction.flat_idx];
      if (coupling.target_begin != coupling.target_end) result.push_back(instruction);
    }
    return result;
  };
  auto filter_top_lp_coupling_loader = [](const auto& source, const auto& couplings) {
    std::vector<TopLpLoadInstruction> result;
    result.reserve(source.size());
    for (const auto& instruction : source) {
      if (instruction.flat_idx >= couplings.size())
        throw std::logic_error("selected replay transfer coupling is out of range");
      const auto& coupling = couplings[instruction.flat_idx];
      if (coupling.target_begin != coupling.target_end) result.push_back(instruction);
    }
    return result;
  };

  for (std::size_t block_index = 0; block_index < replay_blocks.size(); ++block_index) {
    const auto& source = replay_blocks[block_index];
    BlockReplayProgram selected;
    selected.row_begin = source.row_begin;
    selected.dimension = source.dimension;
    selected.couplings = source.couplings;

    std::vector<std::size_t> required;
    if (block_index == 0) {
      required.reserve(active_outputs.size());
      for (const std::uint32_t output : active_outputs)
        required.push_back(replay_outputs[output].rhs_slot);
    } else {
      for (std::size_t coupling_index = 0; coupling_index < source.couplings.size();
           ++coupling_index) {
        const auto& original = source.couplings[coupling_index];
        auto& filtered = selected.couplings[coupling_index];
        filtered.target_begin =
            static_cast<std::uint32_t>(selected.coupling_operations.size());
        for (std::size_t position = original.target_begin;
             position < original.target_end; ++position) {
          const auto operation = source.coupling_operations[position];
          const std::size_t destination = operation.destination;
          if (destination >= live_global_rhs.size())
            throw std::logic_error("selected replay destination RHS is out of range");
          if (live_global_rhs[destination] == 0) continue;
          if (selected.coupling_operations.size() >=
              std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error(
                "selected replay coupling operation count is too large");
          }
          selected.coupling_operations.push_back(operation);
          required.push_back(operation.source);
        }
        filtered.target_end =
            static_cast<std::uint32_t>(selected.coupling_operations.size());
      }
    }
    std::ranges::sort(required);
    required.erase(std::unique(required.begin(), required.end()), required.end());

    selected.tape = source.tape;
    auto layout = linalg::prune_and_compact_tape(
        selected.tape, source.matrix_workspace_size,
        static_cast<std::uint32_t>(rhs_workspace_size), required);
    selected.matrix_workspace_size = layout.matrix_size;
    variant->matrix_workspace_size = std::max(
        variant->matrix_workspace_size, static_cast<std::size_t>(layout.matrix_size));

    std::vector<std::uint32_t> original_rhs(layout.rhs_size);
    for (std::size_t slot = 0; slot < layout.rhs_remap.size(); ++slot) {
      const std::uint32_t compact = layout.rhs_remap[slot];
      if (compact == linalg::CompactedTapeLayout::INVALID_SLOT) continue;
      original_rhs[compact] = static_cast<std::uint32_t>(slot);
      live_global_rhs[slot] = 1;
    }
    for (auto& instruction : selected.tape) {
      using enum linalg::OpCode;
      if (instruction.opcode() == MulB) {
        instruction =
            linalg::Instruction::pack(MulB, original_rhs[instruction.offset()], 0);
      } else if (instruction.opcode() == FmaB) {
        instruction =
            linalg::Instruction::pack(FmaB, original_rhs[instruction.offset()],
                                      original_rhs[instruction.source()]);
      }
    }
    filter_matrix_loader(source.loader_M, layout.matrix_remap, selected.loader_M);
    selected.pooled_loader_M =
        filter_pooled_matrix_loader(source.pooled_loader_M, layout.matrix_remap);
    selected.top_lp_loader_M =
        filter_top_lp_matrix_loader(source.top_lp_loader_M, layout.matrix_remap);
    filter_coupling_loader(source.loader_couplings, selected.couplings,
                           selected.loader_couplings);
    selected.pooled_loader_couplings = filter_pooled_coupling_loader(
        source.pooled_loader_couplings, selected.couplings);
    selected.top_lp_loader_couplings = filter_top_lp_coupling_loader(
        source.top_lp_loader_couplings, selected.couplings);
    variant->blocks.push_back(std::move(selected));
  }

  auto live_rhs = [&live_global_rhs](std::uint32_t slot) {
    return slot < live_global_rhs.size() && live_global_rhs[slot] != 0;
  };
  variant->loader_B.reserve(loader_B.size());
  for (std::size_t begin = 0; begin < loader_B.size();) {
    const std::size_t group_size = loader_B[begin].group_size;
    const std::size_t end = begin + group_size;
    if (group_size == 0 || end > loader_B.size())
      throw std::logic_error("selected replay encountered an invalid RHS loader group");
    if (live_rhs(loader_B[begin].flat_idx)) {
      variant->loader_B.insert(variant->loader_B.end(),
                               loader_B.begin() + static_cast<ptrdiff_t>(begin),
                               loader_B.begin() + static_cast<ptrdiff_t>(end));
    }
    begin = end;
  }
  for (const auto& instruction : pooled_loader_B) {
    if (live_rhs(instruction.flat_idx)) variant->pooled_loader_B.push_back(instruction);
  }
  variant->top_lp_loader_B.reserve(top_lp_loader_B.size());
  for (std::size_t index = 0; index < top_lp_loader_B.size(); ++index) {
    const auto& instruction = top_lp_loader_B[index];
    if (live_rhs(instruction.flat_idx)) {
      variant->top_lp_loader_B.push_back(instruction);
      variant->coefficients.top_lp_expressions.push_back(instruction.expression);
    }
  }

  std::vector<std::uint32_t> rhs_remap(live_global_rhs.size(),
                                       linalg::CompactedTapeLayout::INVALID_SLOT);
  std::uint32_t live_rhs_count = 0;
  for (std::size_t slot = 0; slot < live_global_rhs.size(); ++slot) {
    if (live_global_rhs[slot] == 0) continue;
    if (live_rhs_count == std::numeric_limits<std::uint32_t>::max())
      throw std::logic_error("selected replay live RHS count exceeds 32-bit slots");
    rhs_remap[slot] = live_rhs_count++;
  }
  const auto compact_rhs_slot = [&](std::size_t slot) {
    if (slot >= rhs_remap.size() ||
        rhs_remap[slot] == linalg::CompactedTapeLayout::INVALID_SLOT) {
      throw std::logic_error("selected replay references a dead RHS slot");
    }
    return rhs_remap[slot];
  };
  const auto remap_rhs_loader = [&](auto& loader) {
    for (auto& instruction : loader)
      instruction.flat_idx = compact_rhs_slot(instruction.flat_idx);
  };
  remap_rhs_loader(variant->loader_B);
  remap_rhs_loader(variant->pooled_loader_B);
  remap_rhs_loader(variant->top_lp_loader_B);
  for (auto& output : variant->outputs)
    output.rhs_slot = compact_rhs_slot(output.rhs_slot);
  for (auto& block : variant->blocks) {
    for (auto& instruction : block.tape) {
      using enum linalg::OpCode;
      const auto opcode = instruction.opcode();
      if (opcode == MulB) {
        instruction = linalg::Instruction::pack(
            opcode, compact_rhs_slot(instruction.offset()), 0);
      } else if (opcode == FmaB) {
        instruction =
            linalg::Instruction::pack(opcode, compact_rhs_slot(instruction.offset()),
                                      compact_rhs_slot(instruction.source()));
      }
    }
    for (auto& operation : block.coupling_operations) {
      operation.destination = compact_rhs_slot(operation.destination);
      operation.source = compact_rhs_slot(operation.source);
    }
    block.compiled_tape = linalg::compile_tape(block.tape);
    std::vector<linalg::Instruction>().swap(block.tape);
  }
  variant->rhs_workspace_size = live_rhs_count;

  for (const auto& instruction : variant->pooled_loader_B)
    variant->coefficients.pooled_expressions.push_back(instruction.expression);
  for (const auto& block : variant->blocks) {
    for (const auto& instruction : block.pooled_loader_M)
      variant->coefficients.pooled_expressions.push_back(instruction.expression);
    for (const auto& instruction : block.pooled_loader_couplings)
      variant->coefficients.pooled_expressions.push_back(instruction.expression);
    for (const auto& instruction : block.top_lp_loader_M)
      variant->coefficients.top_lp_expressions.push_back(instruction.expression);
    for (const auto& instruction : block.top_lp_loader_couplings)
      variant->coefficients.top_lp_expressions.push_back(instruction.expression);
  }
  for (const auto& output : variant->outputs) {
    variant->coefficients.targets.push_back(output.target);
    variant->coefficients.basis.push_back(output.basis);
  }
  complete_coefficient_selection(variant->coefficients);

  return variant;
}
