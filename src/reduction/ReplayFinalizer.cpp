#include "reduction/BlackBoxFeynman.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

void BlackBoxFeynman::finalize_replay(const ReductionProgressCallback& progress)
{
  if (!output_support_initialized)
    throw std::logic_error("reduction output support is not initialized");
  if (pending_block_replay == nullptr)
    throw std::logic_error("pending block replay is unavailable");
  if (!replay_blocks.empty() || coefficient_pool_initialized)
    throw std::logic_error("block replay is already finalized");

  const std::size_t num_targets = cfg.targets.size();
  const std::size_t num_basis = cfg.basis.size();
  auto pending = std::move(pending_block_replay);
  auto& programs = pending->programs;
  auto& recorded_blocks = pending->recorded_blocks;
  auto& operation_tapes = pending->operation_tapes;
  const auto& solution_row_by_column = pending->solution_row_by_column;
  const auto& block_of_position = pending->block_of_position;
  if (programs.empty() || programs.size() != recorded_blocks.size() ||
      programs.size() != operation_tapes.size())
    throw std::logic_error("pending block replay shape is inconsistent");
  if (solution_row_by_column.size() != square_dim ||
      block_of_position.size() != square_dim) {
    throw std::logic_error("pending block replay coordinates are inconsistent");
  }

  std::vector<std::uint8_t> live_targets(num_targets, 0);
  std::vector<std::uint8_t> live_basis(num_basis, 0);
  for (const std::uint32_t output : reconstructed_output_positions) {
    if (output >= total_output_count())
      throw std::logic_error("reconstructed output position is out of range");
    live_targets[output / num_basis] = 1;
    live_basis[output % num_basis] = 1;
  }
  const std::size_t live_basis_count =
      static_cast<std::size_t>(std::ranges::count(live_basis, 1));

  std::size_t estimated_target_operations = 0;
  std::vector<std::vector<std::uint8_t>> target_live_rhs(programs.size());
  for (std::size_t block = 0; block < programs.size(); ++block) {
    const auto& program = programs[block];
    const auto& recorded = recorded_blocks[block];
    std::vector<std::size_t> required;
    if (block == 0) {
      required.reserve(reconstructed_output_positions.size());
      for (const std::uint32_t output : reconstructed_output_positions) {
        const std::size_t target = output / num_basis;
        const std::size_t basis = output % num_basis;
        const std::size_t solution_row = solution_row_by_column[basis];
        if (solution_row < program.row_begin ||
            solution_row >=
                static_cast<std::size_t>(program.row_begin) + program.dimension) {
          throw std::logic_error("basis output is outside the first replay block");
        }
        required.push_back((solution_row - program.row_begin) * num_targets + target);
      }
    } else {
      for (const auto coupling : program.couplings) {
        const std::size_t destination_block =
            block_of_position[coupling.destination_row];
        const auto& destination_program = programs[destination_block];
        const auto& destination_live = target_live_rhs[destination_block];
        const std::size_t destination_local_row =
            coupling.destination_row - destination_program.row_begin;
        const std::size_t source_local_row = coupling.source_row - program.row_begin;
        for (std::size_t target = 0; target < num_targets; ++target) {
          if (destination_live[destination_local_row * num_targets + target] == 0)
            continue;
          required.push_back(source_local_row * num_targets + target);
          ++estimated_target_operations;
        }
      }
    }
    auto& live_rhs = target_live_rhs[block];
    live_rhs.assign(recorded.rhs_slot_count, 0);
    for (const std::size_t slot : required) {
      if (slot >= live_rhs.size())
        throw std::logic_error("target estimate RHS slot is out of range");
      live_rhs[slot] = 1;
    }
    for (std::size_t reverse = operation_tapes[block].size(); reverse > 0; --reverse) {
      const auto instruction = operation_tapes[block][reverse - 1];
      if (instruction.opcode() == linalg::OpCode::MulB) {
        if (live_rhs[instruction.offset()] != 0) ++estimated_target_operations;
      } else if (instruction.opcode() == linalg::OpCode::FmaB &&
                 live_rhs[instruction.offset()] != 0) {
        live_rhs[instruction.source()] = 1;
        ++estimated_target_operations;
      }
    }
  }

  std::vector<std::vector<std::uint32_t>> target_coordinates_by_target(num_targets);
  const auto append_target_coordinate = [&](std::uint32_t flat_idx) {
    target_coordinates_by_target[flat_idx % num_targets].push_back(flat_idx);
  };
  for (const auto& entry : pending->rhs_skeleton)
    append_target_coordinate(entry.flat_idx);
  for (const auto& entry : pending->top_lp_rhs_skeleton)
    append_target_coordinate(entry.flat_idx);
  for (auto& coordinates : target_coordinates_by_target) {
    std::ranges::sort(coordinates);
    coordinates.erase(std::unique(coordinates.begin(), coordinates.end()),
                      coordinates.end());
  }

  std::size_t estimated_master_operations = 0;
  if (live_basis_count != 0) {
    std::vector<std::uint32_t> basis_to_estimate_column(
        num_basis, linalg::CompactedTapeLayout::INVALID_SLOT);
    std::size_t estimate_column = 0;
    for (std::size_t basis = 0; basis < num_basis; ++basis) {
      if (live_basis[basis] == 0) continue;
      basis_to_estimate_column[basis] = static_cast<std::uint32_t>(estimate_column++);
    }
    const std::size_t dense_slots = square_dim * live_basis_count;
    const std::size_t mask_words = (live_basis_count + 63U) / 64U;
    std::vector<std::vector<std::uint64_t>> operation_masks(programs.size());
    std::vector<std::vector<std::uint64_t>> coupling_masks(programs.size());
    for (std::size_t block = 0; block < programs.size(); ++block) {
      operation_masks[block].assign(
          recorded_blocks[block].row_operations.size() * mask_words, 0);
      coupling_masks[block].assign(programs[block].couplings.size() * mask_words, 0);
    }
    std::vector<std::uint8_t> reachable(dense_slots, 0);
    for (std::size_t basis = 0; basis < num_basis; ++basis) {
      const std::uint32_t column = basis_to_estimate_column[basis];
      if (column == linalg::CompactedTapeLayout::INVALID_SLOT) continue;
      reachable[static_cast<std::size_t>(solution_row_by_column[basis]) *
                    live_basis_count +
                column] = 1;
    }
    const auto capture_reachable = [&](std::size_t source_row,
                                       std::size_t destination_row, bool propagate,
                                       std::span<std::uint64_t> mask) {
      const std::size_t source = source_row * live_basis_count;
      const std::size_t destination = destination_row * live_basis_count;
      for (std::size_t column = 0; column < live_basis_count; ++column) {
        if (reachable[source + column] == 0) continue;
        mask[column / 64U] |= std::uint64_t{1} << (column % 64U);
        if (propagate) reachable[destination + column] = 1;
      }
    };
    for (std::size_t block = 0; block < programs.size(); ++block) {
      const auto& program = programs[block];
      for (std::size_t coupling_index = 0; coupling_index < program.couplings.size();
           ++coupling_index) {
        const auto coupling = program.couplings[coupling_index];
        capture_reachable(coupling.destination_row, coupling.source_row, true,
                          std::span(coupling_masks[block])
                              .subspan(coupling_index * mask_words, mask_words));
      }
      const auto& operations = recorded_blocks[block].row_operations;
      for (std::size_t reverse = operations.size(); reverse > 0; --reverse) {
        const std::size_t operation_index = reverse - 1;
        const auto operation = operations[operation_index];
        const std::size_t destination =
            static_cast<std::size_t>(program.row_begin) + operation.destination;
        const std::size_t source =
            static_cast<std::size_t>(program.row_begin) + operation.source;
        capture_reachable(destination, operation.scale ? destination : source,
                          !operation.scale,
                          std::span(operation_masks[block])
                              .subspan(operation_index * mask_words, mask_words));
      }
    }

    std::vector<std::uint8_t> required(dense_slots, 0);
    for (const std::uint32_t output : reconstructed_output_positions) {
      const std::size_t target = output / num_basis;
      const std::size_t basis = output % num_basis;
      const std::size_t column = basis_to_estimate_column[basis];
      for (const std::uint32_t coordinate : target_coordinates_by_target[target]) {
        const std::size_t row = coordinate / num_targets;
        const std::size_t slot = row * live_basis_count + column;
        if (reachable[slot] == 0) continue;
        required[slot] = 1;
        ++estimated_master_operations;
      }
    }
    const auto propagate_required = [&](std::size_t destination_row,
                                        std::size_t source_row,
                                        std::span<const std::uint64_t> mask) {
      const std::size_t destination = destination_row * live_basis_count;
      const std::size_t source = source_row * live_basis_count;
      std::size_t operations = 0;
      for (std::size_t column = 0; column < live_basis_count; ++column) {
        if ((mask[column / 64U] & (std::uint64_t{1} << (column % 64U))) == 0 ||
            required[destination + column] == 0) {
          continue;
        }
        required[source + column] = 1;
        ++operations;
      }
      return operations;
    };
    for (std::size_t reverse_block = programs.size(); reverse_block > 0;
         --reverse_block) {
      const std::size_t block = reverse_block - 1;
      const auto& program = programs[block];
      const auto& operations = recorded_blocks[block].row_operations;
      for (std::size_t operation_index = 0; operation_index < operations.size();
           ++operation_index) {
        const auto operation = operations[operation_index];
        const std::size_t destination =
            static_cast<std::size_t>(program.row_begin) + operation.destination;
        const std::size_t source =
            static_cast<std::size_t>(program.row_begin) + operation.source;
        estimated_master_operations +=
            operation.scale
                ? propagate_required(
                      destination, destination,
                      std::span(operation_masks[block])
                          .subspan(operation_index * mask_words, mask_words))
                : propagate_required(
                      source, destination,
                      std::span(operation_masks[block])
                          .subspan(operation_index * mask_words, mask_words));
      }
      for (std::size_t reverse = program.couplings.size(); reverse > 0; --reverse) {
        const std::size_t coupling_index = reverse - 1;
        const auto coupling = program.couplings[coupling_index];
        estimated_master_operations +=
            propagate_required(coupling.source_row, coupling.destination_row,
                               std::span(coupling_masks[block])
                                   .subspan(coupling_index * mask_words, mask_words));
      }
    }
  }
  kernel_statistics_.estimated_target_operations = estimated_target_operations;
  kernel_statistics_.estimated_master_operations = estimated_master_operations;

  // The transposed state stream has less contiguous access than target replay.
  // Forced-orientation benchmarks require about a 1.25x arithmetic advantage to
  // offset that overhead, so close structural scores deliberately stay target-first.
  const bool automatically_use_master =
      live_basis_count != 0 &&
      estimated_master_operations < estimated_target_operations &&
      estimated_target_operations - estimated_master_operations >=
          estimated_target_operations / 5U;
  const bool use_master =
      live_basis_count != 0 &&
      (replay_orientation_preference == ReplayOrientationPreference::Master ||
       (replay_orientation_preference == ReplayOrientationPreference::Auto &&
        automatically_use_master));

  // Both orientations use the same selected square, pivots, factor tape, and SCC
  // blocks. Auto mode compares their structural boundary-operation estimates.
  if (use_master) {
    replay_orientation = ReplayOrientation::Master;
    kernel_statistics_.replay_orientation = "master";
    std::vector<std::uint32_t> basis_to_master(
        num_basis, linalg::CompactedTapeLayout::INVALID_SLOT);
    for (std::size_t basis = 0; basis < num_basis; ++basis) {
      if (live_basis[basis] == 0) continue;
      basis_to_master[basis] = static_cast<std::uint32_t>(master_rhs_columns.size());
      master_rhs_columns.push_back(static_cast<std::uint32_t>(basis));
    }
    const std::size_t solver_rhs_columns = master_rhs_columns.size();
    if (solver_rhs_columns == 0 ||
        square_dim > std::numeric_limits<std::uint32_t>::max() / solver_rhs_columns) {
      throw std::logic_error("master RHS workspace exceeds tape encoding");
    }
    const std::size_t dense_rhs_slots = square_dim * solver_rhs_columns;
    master_seed_slots.reserve(solver_rhs_columns);
    for (std::size_t column = 0; column < solver_rhs_columns; ++column) {
      const std::size_t basis = master_rhs_columns[column];
      const std::size_t slot =
          static_cast<std::size_t>(solution_row_by_column[basis]) * solver_rhs_columns +
          column;
      master_seed_slots.push_back(static_cast<std::uint32_t>(slot));
    }

    std::vector<std::uint32_t> target_coordinates;
    target_coordinates.reserve(pending->rhs_skeleton.size() +
                               pending->top_lp_rhs_skeleton.size());
    const auto collect_coordinate = [&](std::uint32_t flat_idx) {
      const std::size_t target = flat_idx % num_targets;
      if (live_targets[target] != 0) target_coordinates.push_back(flat_idx);
    };
    for (const auto& entry : pending->rhs_skeleton)
      collect_coordinate(entry.flat_idx);
    for (const auto& entry : pending->top_lp_rhs_skeleton)
      collect_coordinate(entry.flat_idx);
    std::ranges::sort(target_coordinates);
    target_coordinates.erase(
        std::unique(target_coordinates.begin(), target_coordinates.end()),
        target_coordinates.end());
    if (target_coordinates.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::logic_error("master target workspace exceeds 32-bit slots");
    const auto target_slot = [&](std::uint32_t flat_idx) {
      const auto found = std::ranges::lower_bound(target_coordinates, flat_idx);
      if (found == target_coordinates.end() || *found != flat_idx)
        throw std::logic_error("master contraction references a dead target slot");
      return static_cast<std::uint32_t>(found - target_coordinates.begin());
    };
    struct TargetCoordinate {
      std::uint32_t slot;
      std::uint32_t row;
    };
    std::vector<std::vector<TargetCoordinate>> coordinates_by_target(num_targets);
    for (std::size_t slot = 0; slot < target_coordinates.size(); ++slot) {
      const std::size_t coordinate = target_coordinates[slot];
      const std::size_t row = coordinate / num_targets;
      const std::size_t target = coordinate % num_targets;
      coordinates_by_target[target].push_back(
          {static_cast<std::uint32_t>(slot), static_cast<std::uint32_t>(row)});
    }
    replay_outputs.reserve(reconstructed_output_positions.size());
    for (const std::uint32_t output : reconstructed_output_positions) {
      const std::size_t target = output / num_basis;
      const std::size_t basis = output % num_basis;
      const std::uint32_t master_column = basis_to_master[basis];
      if (master_column == linalg::CompactedTapeLayout::INVALID_SLOT)
        throw std::logic_error("master output basis is not live");
      if (master_contractions.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::logic_error("master contraction count exceeds 32-bit offsets");
      const auto begin = static_cast<std::uint32_t>(master_contractions.size());
      for (const auto coordinate : coordinates_by_target[target]) {
        const std::size_t slot =
            static_cast<std::size_t>(coordinate.row) * solver_rhs_columns +
            master_column;
        master_contractions.push_back(
            {coordinate.slot, static_cast<std::uint32_t>(slot)});
      }
      const auto end = static_cast<std::uint32_t>(master_contractions.size());
      replay_outputs.push_back({0, static_cast<std::uint32_t>(target),
                                static_cast<std::uint32_t>(basis), begin, end});
    }

    // First propagate structural support forward from the master-selector seeds.
    // This is the source side of master liveness; output demand is applied below.
    std::size_t forward_operations = 0;
    std::vector<std::uint8_t> live_master_rhs(dense_rhs_slots, 0);
    for (const std::uint32_t slot : master_seed_slots)
      live_master_rhs[slot] = 1;
    for (std::size_t block = 0; block < programs.size(); ++block) {
      auto& program = programs[block];
      auto& recorded = recorded_blocks[block];
      program.master = std::make_unique<MasterReplayProgram>();
      auto& master = *program.master;
      program.tape.reserve(operation_tapes[block].size());
      for (const auto instruction : operation_tapes[block]) {
        if (instruction.opcode() == linalg::OpCode::MulB ||
            instruction.opcode() == linalg::OpCode::FmaB) {
          continue;
        }
        program.tape.push_back(instruction);
      }
      master.pending_operations.reserve(recorded.row_operations.size());
      master.uses_masks = solver_rhs_columns <= 64;
      for (std::size_t group = 0; group < recorded.row_operations.size(); ++group) {
        const auto operation = recorded.row_operations[group];
        master.pending_operations.push_back({operation.destination, operation.source,
                                             static_cast<std::uint32_t>(group), 0, 0, 0,
                                             operation.scale});
      }
      const auto append_live_ranges = [&](std::size_t source_row,
                                          std::size_t destination_row, bool propagate,
                                          std::uint64_t& mask) {
        if (master.column_ranges.size() > std::numeric_limits<std::uint32_t>::max()) {
          throw std::logic_error("master range count exceeds 32-bit offsets");
        }
        const auto begin = static_cast<std::uint32_t>(master.column_ranges.size());
        const std::size_t source = source_row * solver_rhs_columns;
        const std::size_t destination = destination_row * solver_rhs_columns;
        if (master.uses_masks) {
          mask = 0;
          for (std::size_t column = 0; column < solver_rhs_columns; ++column) {
            if (live_master_rhs[source + column] == 0) continue;
            mask |= std::uint64_t{1} << column;
            if (propagate) live_master_rhs[destination + column] = 1;
          }
          return std::pair{begin, begin};
        }
        for (std::size_t column = 0; column < solver_rhs_columns;) {
          while (column < solver_rhs_columns && live_master_rhs[source + column] == 0) {
            ++column;
          }
          const std::size_t range_begin = column;
          while (column < solver_rhs_columns && live_master_rhs[source + column] != 0) {
            if (propagate) live_master_rhs[destination + column] = 1;
            ++column;
          }
          if (range_begin != column) {
            master.column_ranges.push_back({static_cast<std::uint32_t>(range_begin),
                                            static_cast<std::uint32_t>(column)});
          }
        }
        return std::pair{begin,
                         static_cast<std::uint32_t>(master.column_ranges.size())};
      };
      for (auto& coupling : program.couplings) {
        const auto [begin, end] = append_live_ranges(
            coupling.destination_row, coupling.source_row, true, coupling.master_mask);
        coupling.target_begin = begin;
        coupling.target_end = end;
        forward_operations +=
            master.uses_masks
                ? static_cast<std::size_t>(std::popcount(coupling.master_mask))
                : [&] {
                    std::size_t count = 0;
                    for (std::size_t range = begin; range < end; ++range) {
                      count += master.column_ranges[range].end -
                               master.column_ranges[range].begin;
                    }
                    return count;
                  }();
      }
      for (std::size_t reverse = master.pending_operations.size(); reverse > 0;
           --reverse) {
        auto& operation = master.pending_operations[reverse - 1];
        const std::size_t target_destination =
            static_cast<std::size_t>(program.row_begin) + operation.destination_row;
        const std::size_t target_source =
            static_cast<std::size_t>(program.row_begin) + operation.source_row;
        const auto [begin, end] =
            operation.scale ? append_live_ranges(target_destination, target_destination,
                                                 false, operation.mask)
                            : append_live_ranges(target_destination, target_source,
                                                 true, operation.mask);
        operation.range_begin = begin;
        operation.range_end = end;
        forward_operations +=
            master.uses_masks
                ? static_cast<std::size_t>(std::popcount(operation.mask))
                : [&] {
                    std::size_t count = 0;
                    for (std::size_t range = begin; range < end; ++range) {
                      count += master.column_ranges[range].end -
                               master.column_ranges[range].begin;
                    }
                    return count;
                  }();
      }
    }
    const std::size_t forward_slots =
        static_cast<std::size_t>(std::ranges::count(live_master_rhs, 1));

    // Contractions are the sinks of the master program.  Remove coordinates that
    // are structurally unreachable from their master seed before seeding the reverse
    // liveness pass.
    std::vector<std::uint8_t> required_master_rhs(dense_rhs_slots, 0);
    std::vector<std::uint8_t> live_master_slots(dense_rhs_slots, 0);
    std::vector<MasterContraction> reachable_contractions;
    reachable_contractions.reserve(master_contractions.size());
    for (auto& output : replay_outputs) {
      const auto begin = static_cast<std::uint32_t>(reachable_contractions.size());
      for (std::size_t position = output.contraction_begin;
           position < output.contraction_end; ++position) {
        const auto contraction = master_contractions[position];
        if (live_master_rhs[contraction.master_slot] == 0) continue;
        required_master_rhs[contraction.master_slot] = 1;
        live_master_slots[contraction.master_slot] = 1;
        reachable_contractions.push_back(contraction);
      }
      output.contraction_begin = begin;
      output.contraction_end =
          static_cast<std::uint32_t>(reachable_contractions.size());
    }
    master_contractions = std::move(reachable_contractions);

    std::size_t row_operations = 0;
    std::size_t coupling_operations = 0;
    for (std::size_t reverse_block = programs.size(); reverse_block > 0;
         --reverse_block) {
      auto& program = programs[reverse_block - 1];
      if (program.master == nullptr)
        throw std::logic_error("master replay planning state is unavailable");
      auto& master = *program.master;
      auto old_ranges = std::move(master.column_ranges);
      std::vector<MasterColumnRange> pruned_ranges;
      pruned_ranges.reserve(old_ranges.size());
      const auto prune_operation = [&](std::size_t destination_row,
                                       std::size_t source_row, std::uint64_t& mask,
                                       std::uint32_t& range_begin,
                                       std::uint32_t& range_end) {
        const std::size_t destination = destination_row * solver_rhs_columns;
        const std::size_t source = source_row * solver_rhs_columns;
        if (master.uses_masks) {
          std::uint64_t kept = 0;
          for (std::size_t column = 0; column < solver_rhs_columns; ++column) {
            const std::uint64_t bit = std::uint64_t{1} << column;
            if ((mask & bit) == 0 || required_master_rhs[destination + column] == 0) {
              continue;
            }
            kept |= bit;
            required_master_rhs[source + column] = 1;
            live_master_slots[destination + column] = 1;
            live_master_slots[source + column] = 1;
          }
          mask = kept;
          range_begin = range_end = 0;
          return static_cast<std::size_t>(std::popcount(kept));
        }

        if (pruned_ranges.size() > std::numeric_limits<std::uint32_t>::max())
          throw std::logic_error("pruned master range count exceeds 32-bit offsets");
        const auto new_begin = static_cast<std::uint32_t>(pruned_ranges.size());
        std::size_t kept_count = 0;
        for (std::size_t range = range_begin; range < range_end; ++range) {
          const auto old = old_ranges[range];
          std::size_t column = old.begin;
          while (column < old.end) {
            while (column < old.end && required_master_rhs[destination + column] == 0) {
              ++column;
            }
            const std::size_t begin = column;
            while (column < old.end && required_master_rhs[destination + column] != 0) {
              required_master_rhs[source + column] = 1;
              live_master_slots[destination + column] = 1;
              live_master_slots[source + column] = 1;
              ++column;
            }
            if (begin != column) {
              pruned_ranges.push_back({static_cast<std::uint32_t>(begin),
                                       static_cast<std::uint32_t>(column)});
              kept_count += column - begin;
            }
          }
        }
        range_begin = new_begin;
        range_end = static_cast<std::uint32_t>(pruned_ranges.size());
        return kept_count;
      };

      // Runtime applies row operations in reverse recorded order, so reverse
      // liveness visits them in their stored order.
      for (auto& operation : master.pending_operations) {
        const std::size_t target_destination =
            static_cast<std::size_t>(program.row_begin) + operation.destination_row;
        const std::size_t target_source =
            static_cast<std::size_t>(program.row_begin) + operation.source_row;
        row_operations +=
            operation.scale
                ? prune_operation(target_destination, target_destination,
                                  operation.mask, operation.range_begin,
                                  operation.range_end)
                : prune_operation(target_source, target_destination, operation.mask,
                                  operation.range_begin, operation.range_end);
      }
      // Transposed cross-block couplings execute before the block-local operations.
      // They commute within a block because they only read lower-block rows.
      for (std::size_t reverse = program.couplings.size(); reverse > 0; --reverse) {
        auto& coupling = program.couplings[reverse - 1];
        coupling_operations += prune_operation(
            coupling.source_row, coupling.destination_row, coupling.master_mask,
            coupling.target_begin, coupling.target_end);
      }
      master.column_ranges = std::move(pruned_ranges);
    }

    for (const std::uint32_t seed : master_seed_slots) {
      if (required_master_rhs[seed] != 0) live_master_slots[seed] = 1;
    }

    // Compact the sparse (row, master-column) state space once.  Published replay
    // operations carry compact slot runs, so the hot loop performs no remap lookup.
    std::vector<std::uint32_t> master_remap(dense_rhs_slots,
                                            linalg::CompactedTapeLayout::INVALID_SLOT);
    std::uint32_t live_slot_count = 0;
    for (std::size_t slot = 0; slot < live_master_slots.size(); ++slot) {
      if (live_master_slots[slot] == 0) continue;
      if (live_slot_count == std::numeric_limits<std::uint32_t>::max())
        throw std::logic_error("compacted master workspace exceeds 32-bit slots");
      master_remap[slot] = live_slot_count++;
    }
    const auto compact_master_slot = [&](std::size_t row, std::size_t column) {
      const std::size_t slot = row * solver_rhs_columns + column;
      if (slot >= master_remap.size() ||
          master_remap[slot] == linalg::CompactedTapeLayout::INVALID_SLOT) {
        throw std::logic_error("master operation references a dead compact slot");
      }
      return master_remap[slot];
    };
    for (std::size_t column = 0; column < master_seed_slots.size(); ++column) {
      const std::uint32_t original = master_seed_slots[column];
      master_seed_slots[column] = required_master_rhs[original] == 0
                                      ? linalg::CompactedTapeLayout::INVALID_SLOT
                                      : master_remap[original];
    }
    for (auto& contraction : master_contractions)
      contraction.master_slot = master_remap[contraction.master_slot];
    rhs_workspace_size = live_slot_count;

    const auto append_compact_ranges = [&](MasterReplayProgram& master,
                                           std::size_t destination_row,
                                           std::size_t source_row, std::uint64_t mask,
                                           std::uint32_t range_begin,
                                           std::uint32_t range_end) {
      if (master.slot_ranges.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("master slot range count exceeds 32-bit offsets");
      }
      const auto begin = static_cast<std::uint32_t>(master.slot_ranges.size());
      const auto append_column = [&](std::size_t column) {
        const std::uint32_t destination = compact_master_slot(destination_row, column);
        const std::uint32_t source = compact_master_slot(source_row, column);
        if (master.slot_ranges.size() > begin) {
          auto& previous = master.slot_ranges.back();
          if (previous.destination + previous.count == destination &&
              previous.source + previous.count == source &&
              previous.column_begin + previous.count == column) {
            ++previous.count;
            return;
          }
        }
        master.slot_ranges.push_back(
            {destination, source, static_cast<std::uint32_t>(column), 1});
      };
      if (master.uses_masks) {
        for (std::size_t column = 0; column < solver_rhs_columns; ++column) {
          if ((mask & (std::uint64_t{1} << column)) != 0) append_column(column);
        }
      } else {
        for (std::size_t range = range_begin; range < range_end; ++range) {
          const auto columns = master.column_ranges[range];
          for (std::size_t column = columns.begin; column < columns.end; ++column)
            append_column(column);
        }
      }
      return std::pair{begin, static_cast<std::uint32_t>(master.slot_ranges.size())};
    };

    std::size_t tape_instructions = 0;
    std::size_t matrix_operations = 0;
    std::size_t factor_groups = 0;
    std::size_t compiled_bytes = 0;
    std::size_t maximum_matrix_workspace = 0;
    std::size_t maximum_factor_workspace = 0;
    std::array<std::size_t, 6> opcode_counts{};
    for (std::size_t block = 0; block < programs.size(); ++block) {
      auto& program = programs[block];
      auto& recorded = recorded_blocks[block];
      if (program.master == nullptr)
        throw std::logic_error("master replay planning state is unavailable");
      auto& master = *program.master;
      for (auto& coupling : program.couplings) {
        const auto ranges = append_compact_ranges(
            master, coupling.source_row, coupling.destination_row, coupling.master_mask,
            coupling.target_begin, coupling.target_end);
        coupling.target_begin = ranges.first;
        coupling.target_end = ranges.second;
      }
      for (auto& operation : master.pending_operations) {
        const std::size_t target_destination =
            static_cast<std::size_t>(program.row_begin) + operation.destination_row;
        const std::size_t target_source =
            static_cast<std::size_t>(program.row_begin) + operation.source_row;
        const auto ranges =
            operation.scale
                ? append_compact_ranges(master, target_destination, target_destination,
                                        operation.mask, operation.range_begin,
                                        operation.range_end)
                : append_compact_ranges(master, target_source, target_destination,
                                        operation.mask, operation.range_begin,
                                        operation.range_end);
        operation.range_begin = ranges.first;
        operation.range_end = ranges.second;
      }

      std::vector<std::uint8_t> required_factors(master.pending_operations.size(), 0);
      for (const auto& operation : master.pending_operations) {
        if (operation.range_begin != operation.range_end)
          required_factors[operation.factor_group] = 1;
      }
      auto matrix_layout = linalg::prune_matrix_factor_tape(
          program.tape, recorded.matrix_slot_count, required_factors);
      std::erase_if(program.skeleton_M, [&](AnsatzEntry& entry) {
        if (entry.flat_idx >= matrix_layout.matrix_remap.size() ||
            matrix_layout.matrix_remap[entry.flat_idx] ==
                linalg::CompactedTapeLayout::INVALID_SLOT) {
          return true;
        }
        entry.flat_idx = matrix_layout.matrix_remap[entry.flat_idx];
        return false;
      });
      std::erase_if(program.top_lp_skeleton_M, [&](TopLpRhsEntry& entry) {
        if (entry.flat_idx >= matrix_layout.matrix_remap.size() ||
            matrix_layout.matrix_remap[entry.flat_idx] ==
                linalg::CompactedTapeLayout::INVALID_SLOT) {
          return true;
        }
        entry.flat_idx = matrix_layout.matrix_remap[entry.flat_idx];
        return false;
      });
      std::erase_if(master.pending_operations, [&](PendingMasterOperation& operation) {
        if (operation.range_begin == operation.range_end) return true;
        const std::uint32_t mapped = matrix_layout.group_remap[operation.factor_group];
        if (mapped == linalg::CompactedTapeLayout::INVALID_SLOT) return true;
        operation.factor_group = mapped;
        return false;
      });
      std::erase_if(program.skeleton_couplings, [&](const auto& entry) {
        return entry.flat_idx >= program.couplings.size() ||
               program.couplings[entry.flat_idx].target_begin ==
                   program.couplings[entry.flat_idx].target_end;
      });
      std::erase_if(program.top_lp_skeleton_couplings, [&](const auto& entry) {
        return entry.flat_idx >= program.couplings.size() ||
               program.couplings[entry.flat_idx].target_begin ==
                   program.couplings[entry.flat_idx].target_end;
      });
      master.operations.reserve(master.pending_operations.size());
      for (auto operation = master.pending_operations.rbegin();
           operation != master.pending_operations.rend(); ++operation) {
        master.operations.push_back(
            {operation->factor_group, operation->range_begin, operation->range_end,
             operation->scale ? MasterOperationKind::Scale
                              : MasterOperationKind::Eliminate});
      }
      master.couplings.reserve(program.couplings.size());
      for (const auto coupling : program.couplings) {
        master.couplings.push_back({coupling.target_begin, coupling.target_end});
      }
      program.compiled_tape = linalg::compile_tape(program.tape);
      master.factor_capture_slots.assign(program.compiled_tape.groups.size(),
                                         linalg::CompactedTapeLayout::INVALID_SLOT);
      for (const auto& operation : master.operations) {
        if (operation.factor_group >= master.factor_capture_slots.size())
          throw std::logic_error("master operation refers to an absent matrix factor");
        master.factor_capture_slots[operation.factor_group] = 0;
      }
      for (auto& slot : master.factor_capture_slots) {
        if (slot == linalg::CompactedTapeLayout::INVALID_SLOT) continue;
        if (master.factor_count == linalg::CompactedTapeLayout::INVALID_SLOT)
          throw std::overflow_error("master factor slots exceed compact encoding");
        slot = master.factor_count++;
      }
      for (auto& operation : master.operations)
        operation.factor_group = master.factor_capture_slots[operation.factor_group];
      for (const auto instruction : program.tape)
        ++opcode_counts[static_cast<std::size_t>(instruction.opcode())];
      program.matrix_workspace_size = matrix_layout.matrix_size;
      maximum_matrix_workspace =
          std::max(maximum_matrix_workspace,
                   static_cast<std::size_t>(program.matrix_workspace_size));
      maximum_factor_workspace = std::max(
          maximum_factor_workspace, static_cast<std::size_t>(master.factor_count));
      tape_instructions += program.tape.size();
      matrix_operations += program.compiled_tape.operations.size();
      factor_groups += program.compiled_tape.groups.size();
      compiled_bytes +=
          program.compiled_tape.groups.size() * sizeof(linalg::TapeGroup) +
          program.compiled_tape.operations.size() * sizeof(linalg::TapeOperation) +
          master.factor_capture_slots.size() * sizeof(std::uint32_t) +
          master.operations.size() * sizeof(CompiledMasterOperation) +
          master.slot_ranges.size() * sizeof(MasterSlotRange);
      std::vector<PendingMasterOperation>().swap(master.pending_operations);
      std::vector<MasterColumnRange>().swap(master.column_ranges);
      std::vector<BlockCoupling>().swap(program.couplings);
      std::vector<std::uint32_t>().swap(program.coupling_targets);
      std::vector<CouplingOperation>().swap(program.coupling_operations);
    }

    // Target coordinates form the other sparse boundary.  Compact them after dead
    // contractions have been removed so unused target coefficients are not loaded.
    std::vector<std::uint8_t> live_target_slots(target_coordinates.size(), 0);
    for (const auto contraction : master_contractions)
      live_target_slots[contraction.target_slot] = 1;
    std::vector<std::uint32_t> target_remap(target_coordinates.size(),
                                            linalg::CompactedTapeLayout::INVALID_SLOT);
    std::uint32_t compact_target_count = 0;
    for (std::size_t slot = 0; slot < live_target_slots.size(); ++slot) {
      if (live_target_slots[slot] != 0) target_remap[slot] = compact_target_count++;
    }
    for (auto& contraction : master_contractions)
      contraction.target_slot = target_remap[contraction.target_slot];
    target_rhs_workspace_size = compact_target_count;
    const auto compact_target_slot = [&](std::uint32_t flat_idx) {
      const std::uint32_t original = target_slot(flat_idx);
      return target_remap[original];
    };
    std::erase_if(pending->rhs_skeleton, [&](AnsatzEntry& entry) {
      if (live_targets[entry.flat_idx % num_targets] == 0) return true;
      const std::uint32_t compact = compact_target_slot(entry.flat_idx);
      if (compact == linalg::CompactedTapeLayout::INVALID_SLOT) return true;
      entry.flat_idx = compact;
      return false;
    });
    std::erase_if(pending->top_lp_rhs_skeleton, [&](TopLpRhsEntry& entry) {
      if (live_targets[entry.flat_idx % num_targets] == 0) return true;
      const std::uint32_t compact = compact_target_slot(entry.flat_idx);
      if (compact == linalg::CompactedTapeLayout::INVALID_SLOT) return true;
      entry.flat_idx = compact;
      return false;
    });

    replay_blocks = std::move(programs);
    skeleton_B = std::move(pending->rhs_skeleton);
    top_lp_rhs_skeleton = std::move(pending->top_lp_rhs_skeleton);
    matrix_workspace_size = maximum_matrix_workspace;
    master_factor_workspace_size = maximum_factor_workspace;
    compile_loaders(firefly::FFInt::p);

    replay_coefficients = {};
    for (const auto& instruction : top_lp_loader_B)
      replay_coefficients.top_lp_expressions.push_back(instruction.expression);
    for (const auto& instruction : pooled_loader_B)
      replay_coefficients.pooled_expressions.push_back(instruction.expression);
    for (const auto& block : replay_blocks) {
      for (const auto& instruction : block.pooled_loader_M)
        replay_coefficients.pooled_expressions.push_back(instruction.expression);
      for (const auto& instruction : block.pooled_loader_couplings)
        replay_coefficients.pooled_expressions.push_back(instruction.expression);
      for (const auto& instruction : block.top_lp_loader_M)
        replay_coefficients.top_lp_expressions.push_back(instruction.expression);
      for (const auto& instruction : block.top_lp_loader_couplings)
        replay_coefficients.top_lp_expressions.push_back(instruction.expression);
    }
    for (const auto& output : replay_outputs) {
      replay_coefficients.targets.push_back(output.target);
      replay_coefficients.basis.push_back(output.basis);
    }
    complete_coefficient_selection(replay_coefficients);

    const std::size_t maximum_block_dimension =
        std::ranges::max(replay_blocks | std::views::transform([](const auto& program) {
                           return static_cast<std::size_t>(program.dimension);
                         }));
    kernel_statistics_.block_count = replay_blocks.size();
    kernel_statistics_.maximum_block_dimension = maximum_block_dimension;
    kernel_statistics_.maximum_block_targets = solver_rhs_columns;
    kernel_statistics_.solver_rhs_columns = solver_rhs_columns;
    kernel_statistics_.master_factor_slots = maximum_factor_workspace;
    kernel_statistics_.master_forward_slots = forward_slots;
    kernel_statistics_.master_live_slots = live_slot_count;
    kernel_statistics_.master_forward_operations = forward_operations;
    kernel_statistics_.master_live_operations = row_operations + coupling_operations;
    kernel_statistics_.contraction_fmas = master_contractions.size();
    kernel_statistics_.tape_instructions = tape_instructions;
    kernel_statistics_.tape_opcode_counts = opcode_counts;
    kernel_statistics_.matrix_slots = matrix_workspace_size;
    kernel_statistics_.rhs_slots = rhs_workspace_size + target_rhs_workspace_size;
    kernel_statistics_.target_rhs_slots = target_rhs_workspace_size;
    kernel_statistics_.coupling_fmas = coupling_operations;
    kernel_statistics_.compiled_groups = factor_groups;
    kernel_statistics_.compiled_matrix_operations = matrix_operations;
    kernel_statistics_.compiled_rhs_operations = row_operations + coupling_operations;
    kernel_statistics_.compiled_tape_bytes =
        compiled_bytes + master_contractions.size() * sizeof(MasterContraction);
    kernel_statistics_.matrix_total_operations = matrix_operations;
    kernel_statistics_.top_lp_replay_expressions =
        replay_coefficients.top_lp_expressions.size();
    kernel_statistics_.top_lp_replay_product_nodes =
        replay_coefficients.top_lp_product_nodes.size();
    kernel_statistics_.top_lp_replay_polynomial_factors =
        replay_coefficients.extended_polynomial_factors.size();
    if (progress) {
      progress(std::format(
                   "Block replay: layout={}, orientation=master, dimension={}, "
                   "blocks={}, max_block={}, solver_rhs={}, matrix_slots={}, "
                   "rhs_slots={}, forward_slots={}, live_slots={}, contraction_fmas={}",
                   kernel_statistics_.block_layout, square_dim, replay_blocks.size(),
                   maximum_block_dimension, solver_rhs_columns, matrix_workspace_size,
                   kernel_statistics_.rhs_slots, forward_slots, live_slot_count,
                   master_contractions.size()),
               ReductionProgressEvent::info);
    }
    return;
  }

  replay_orientation = ReplayOrientation::Target;
  kernel_statistics_.replay_orientation = "target";
  kernel_statistics_.solver_rhs_columns = num_targets;

  std::size_t finalized_matrix_workspace_size = 0;
  std::size_t finalized_tape_instructions = 0;
  std::size_t finalized_coupling_fmas = 0;
  std::size_t finalized_rhs_ranges = 0;
  std::size_t finalized_rhs_ranged_operations = 0;
  std::size_t finalized_compiled_groups = 0;
  std::size_t finalized_compiled_matrix_operations = 0;
  std::size_t finalized_compiled_rhs_operations = 0;
  std::size_t finalized_compiled_tape_bytes = 0;
  std::size_t finalized_matrix_contiguous_operations = 0;
  std::size_t maximum_block_targets = 0;
  std::array<std::size_t, 6> finalized_opcode_counts{};
  std::vector<linalg::CompactedTapeLayout> block_layouts(programs.size());
  std::vector<std::uint8_t> live_global_rhs(square_dim * num_targets, 0);
  for (std::size_t block = 0; block < programs.size(); ++block) {
    auto& program = programs[block];
    auto& recorded = recorded_blocks[block];
    const std::size_t dimension = program.dimension;
    std::vector<std::size_t> required;
    if (block == 0) {
      required.reserve(reconstructed_output_positions.size());
      for (const std::uint32_t output : reconstructed_output_positions) {
        if (output >= total_output_count())
          throw std::logic_error("reconstructed output position is out of range");
        const std::size_t target = output / num_basis;
        const std::size_t basis = output % num_basis;
        const std::size_t solution_row = solution_row_by_column[basis];
        if (solution_row < program.row_begin ||
            solution_row >= static_cast<std::size_t>(program.row_begin) + dimension) {
          throw std::logic_error("basis output is outside the first replay block");
        }
        required.push_back((solution_row - program.row_begin) * num_targets + target);
      }
      std::ranges::sort(required);
      required.erase(std::unique(required.begin(), required.end()), required.end());
    } else {
      program.coupling_targets.clear();
      for (auto& coupling : program.couplings) {
        const std::size_t destination_block =
            block_of_position[coupling.destination_row];
        if (destination_block >= block)
          throw std::logic_error("block coupling destination is not lower");
        const auto& destination_program = programs[destination_block];
        const auto& destination_layout = block_layouts[destination_block];
        const std::size_t destination_local_row =
            coupling.destination_row - destination_program.row_begin;
        const std::size_t source_local_row = coupling.source_row - program.row_begin;
        coupling.target_begin =
            static_cast<std::uint32_t>(program.coupling_targets.size());
        for (std::size_t target = 0; target < num_targets; ++target) {
          const std::size_t destination_slot =
              destination_local_row * num_targets + target;
          if (destination_slot < destination_layout.rhs_remap.size() &&
              destination_layout.rhs_remap[destination_slot] !=
                  linalg::CompactedTapeLayout::INVALID_SLOT) {
            if (program.coupling_targets.size() >=
                std::numeric_limits<std::uint32_t>::max()) {
              throw std::logic_error("block coupling target count exceeds encoding");
            }
            program.coupling_targets.push_back(static_cast<std::uint32_t>(target));
            required.push_back(source_local_row * num_targets + target);
          }
        }
        coupling.target_end =
            static_cast<std::uint32_t>(program.coupling_targets.size());
      }
      std::ranges::sort(required);
      required.erase(std::unique(required.begin(), required.end()), required.end());
    }

    auto& operation_tape = operation_tapes[block];
    auto layout = linalg::prune_and_compact_tape(
        operation_tape, recorded.matrix_slot_count, recorded.rhs_slot_count, required);
    std::vector<std::size_t> original_rhs_slot(layout.rhs_size);
    std::vector<std::uint8_t> block_active_targets(num_targets, 0);
    for (std::size_t slot = 0; slot < layout.rhs_remap.size(); ++slot) {
      if (layout.rhs_remap[slot] == linalg::CompactedTapeLayout::INVALID_SLOT) {
        continue;
      }
      original_rhs_slot[layout.rhs_remap[slot]] = slot;
      const std::size_t local_row = slot / num_targets;
      const std::size_t target = slot % num_targets;
      block_active_targets[target] = 1;
      const std::size_t global_slot =
          (static_cast<std::size_t>(program.row_begin) + local_row) * num_targets +
          target;
      if (global_slot >= live_global_rhs.size())
        throw std::logic_error("live block RHS slot is out of range");
      live_global_rhs[global_slot] = 1;
    }
    maximum_block_targets =
        std::max(maximum_block_targets,
                 static_cast<std::size_t>(std::ranges::count(block_active_targets, 1)));

    std::erase_if(program.skeleton_M, [&](auto& entry) {
      if (entry.flat_idx >= layout.matrix_remap.size() ||
          layout.matrix_remap[entry.flat_idx] ==
              linalg::CompactedTapeLayout::INVALID_SLOT) {
        return true;
      }
      entry.flat_idx = layout.matrix_remap[entry.flat_idx];
      return false;
    });
    std::erase_if(program.top_lp_skeleton_M, [&](auto& entry) {
      if (entry.flat_idx >= layout.matrix_remap.size() ||
          layout.matrix_remap[entry.flat_idx] ==
              linalg::CompactedTapeLayout::INVALID_SLOT) {
        return true;
      }
      entry.flat_idx = layout.matrix_remap[entry.flat_idx];
      return false;
    });
    std::erase_if(program.skeleton_couplings, [&](const auto& entry) {
      return entry.flat_idx >= program.couplings.size() ||
             program.couplings[entry.flat_idx].target_begin ==
                 program.couplings[entry.flat_idx].target_end;
    });
    std::erase_if(program.top_lp_skeleton_couplings, [&](const auto& entry) {
      return entry.flat_idx >= program.couplings.size() ||
             program.couplings[entry.flat_idx].target_begin ==
                 program.couplings[entry.flat_idx].target_end;
    });

    const auto global_rhs_slot = [&](std::size_t local_slot) {
      if (local_slot >= original_rhs_slot.size())
        throw std::logic_error("compacted block RHS slot is invalid");
      const std::size_t original_slot = original_rhs_slot[local_slot];
      const std::size_t local_row = original_slot / num_targets;
      const std::size_t target = original_slot % num_targets;
      const std::size_t global_slot =
          (static_cast<std::size_t>(program.row_begin) + local_row) * num_targets +
          target;
      if (global_slot > linalg::Instruction::OP_MASK)
        throw std::logic_error("block RHS slot exceeds tape encoding");
      return global_slot;
    };
    for (auto& instruction : operation_tape) {
      using enum linalg::OpCode;
      const auto opcode = instruction.opcode();
      if (opcode == MulB) {
        instruction =
            linalg::Instruction::pack(opcode, global_rhs_slot(instruction.offset()), 0);
      } else if (opcode == FmaB) {
        instruction =
            linalg::Instruction::pack(opcode, global_rhs_slot(instruction.offset()),
                                      global_rhs_slot(instruction.source()));
      }
    }
    program.matrix_workspace_size = layout.matrix_size;
    program.tape = std::move(operation_tape);
    for (const auto& instruction : program.tape) {
      ++finalized_opcode_counts[static_cast<std::size_t>(instruction.opcode())];
    }
    block_layouts[block] = std::move(layout);
    finalized_matrix_workspace_size =
        std::max(finalized_matrix_workspace_size,
                 static_cast<std::size_t>(program.matrix_workspace_size));
    finalized_tape_instructions += program.tape.size();
  }

  std::vector<std::uint32_t> rhs_remap(live_global_rhs.size(),
                                       linalg::CompactedTapeLayout::INVALID_SLOT);
  std::uint32_t live_rhs_count = 0;
  for (std::size_t slot = 0; slot < live_global_rhs.size(); ++slot) {
    if (live_global_rhs[slot] == 0) continue;
    if (live_rhs_count == std::numeric_limits<std::uint32_t>::max())
      throw std::logic_error("live RHS workspace exceeds 32-bit slots");
    rhs_remap[slot] = live_rhs_count++;
  }
  const auto compact_rhs_slot = [&](std::size_t slot) {
    if (slot >= rhs_remap.size() ||
        rhs_remap[slot] == linalg::CompactedTapeLayout::INVALID_SLOT) {
      throw std::logic_error("published replay references a dead RHS slot");
    }
    return rhs_remap[slot];
  };
  for (auto& program : programs) {
    for (auto& instruction : program.tape) {
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
    program.compiled_tape = linalg::compile_tape(program.tape);
    finalized_rhs_ranges += program.compiled_tape.rhs_ranges.size();
    finalized_compiled_groups += program.compiled_tape.groups.size();
    finalized_compiled_tape_bytes +=
        program.compiled_tape.groups.size() * sizeof(linalg::TapeGroup) +
        program.compiled_tape.operations.size() * sizeof(linalg::TapeOperation) +
        program.compiled_tape.rhs_ranges.size() * sizeof(linalg::TapeRhsRange);
    for (const auto& range : program.compiled_tape.rhs_ranges)
      finalized_rhs_ranged_operations += range.count;

    std::size_t operation_position = 0;
    const auto count_contiguous = [&](const auto& operations, std::size_t begin,
                                      std::size_t count, auto&& adjacent) {
      std::size_t covered = 0;
      std::size_t run_begin = begin;
      const std::size_t end = begin + count;
      while (run_begin < end) {
        std::size_t run_end = run_begin + 1;
        while (run_end < end && adjacent(operations[run_end - 1], operations[run_end]))
          ++run_end;
        if (run_end - run_begin > 1) covered += run_end - run_begin;
        run_begin = run_end;
      }
      return covered;
    };
    finalized_compiled_rhs_operations += program.compiled_tape.logical_rhs_operations;
    for (const auto& group : program.compiled_tape.groups) {
      finalized_compiled_matrix_operations += group.matrix_operations;
      if (group.kind == linalg::TapeGroupKind::Scale) {
        finalized_matrix_contiguous_operations += count_contiguous(
            program.compiled_tape.operations, operation_position,
            group.matrix_operations, [](const auto& lhs, const auto& rhs) {
              return rhs.destination == lhs.destination + 1;
            });
      } else {
        finalized_matrix_contiguous_operations += count_contiguous(
            program.compiled_tape.operations, operation_position,
            group.matrix_operations, [](const auto& lhs, const auto& rhs) {
              return rhs.destination == lhs.destination + 1 &&
                     rhs.source == lhs.source + 1;
            });
      }
      operation_position += group.matrix_operations + group.rhs_operations;
    }
    program.coupling_operations.clear();
    for (auto& coupling : program.couplings) {
      const std::uint32_t begin =
          static_cast<std::uint32_t>(program.coupling_operations.size());
      for (std::size_t position = coupling.target_begin; position < coupling.target_end;
           ++position) {
        const std::size_t target = program.coupling_targets[position];
        const std::size_t destination =
            static_cast<std::size_t>(coupling.destination_row) * num_targets + target;
        const std::size_t source =
            static_cast<std::size_t>(coupling.source_row) * num_targets + target;
        program.coupling_operations.push_back(
            {compact_rhs_slot(destination), compact_rhs_slot(source)});
      }
      coupling.target_begin = begin;
      coupling.target_end =
          static_cast<std::uint32_t>(program.coupling_operations.size());
      finalized_coupling_fmas += coupling.target_end - coupling.target_begin;
    }
    std::vector<std::uint32_t>().swap(program.coupling_targets);
  }

  std::erase_if(pending->rhs_skeleton, [&](AnsatzEntry& entry) {
    if (entry.flat_idx >= live_global_rhs.size() ||
        live_global_rhs[entry.flat_idx] == 0) {
      return true;
    }
    entry.flat_idx = compact_rhs_slot(entry.flat_idx);
    return false;
  });
  std::erase_if(pending->top_lp_rhs_skeleton, [&](TopLpRhsEntry& entry) {
    if (entry.flat_idx >= live_global_rhs.size() ||
        live_global_rhs[entry.flat_idx] == 0) {
      return true;
    }
    entry.flat_idx = compact_rhs_slot(entry.flat_idx);
    return false;
  });

  std::vector<ReplayOutput> finalized_outputs;
  finalized_outputs.reserve(reconstructed_output_positions.size());
  for (const std::uint32_t output : reconstructed_output_positions) {
    const std::size_t target = output / num_basis;
    const std::size_t basis = output % num_basis;
    const std::size_t original_slot =
        static_cast<std::size_t>(solution_row_by_column[basis]) * num_targets + target;
    if (target > std::numeric_limits<std::uint32_t>::max() ||
        basis > std::numeric_limits<std::uint32_t>::max()) {
      throw std::logic_error("replay output coordinate exceeds compact encoding");
    }
    finalized_outputs.push_back({compact_rhs_slot(original_slot),
                                 static_cast<std::uint32_t>(target),
                                 static_cast<std::uint32_t>(basis), 0, 0});
  }

  const std::size_t block_count = programs.size();
  const std::size_t maximum_block_dimension =
      std::ranges::max(programs | std::views::transform([](const auto& program) {
                         return static_cast<std::size_t>(program.dimension);
                       }));
  replay_blocks = std::move(programs);
  skeleton_B = std::move(pending->rhs_skeleton);
  top_lp_rhs_skeleton = std::move(pending->top_lp_rhs_skeleton);
  replay_outputs = std::move(finalized_outputs);
  matrix_workspace_size = finalized_matrix_workspace_size;
  rhs_workspace_size = live_rhs_count;
  compile_loaders(firefly::FFInt::p);
  replay_coefficients = {};
  replay_coefficients.top_lp_expressions.reserve(top_lp_loader_B.size());
  for (const auto& instruction : top_lp_loader_B)
    replay_coefficients.top_lp_expressions.push_back(instruction.expression);
  replay_coefficients.pooled_expressions.reserve(
      kernel_statistics_.pooled_coefficient_loads);
  for (const auto& instruction : pooled_loader_B)
    replay_coefficients.pooled_expressions.push_back(instruction.expression);
  for (const auto& block : replay_blocks) {
    for (const auto& instruction : block.pooled_loader_M)
      replay_coefficients.pooled_expressions.push_back(instruction.expression);
    for (const auto& instruction : block.pooled_loader_couplings)
      replay_coefficients.pooled_expressions.push_back(instruction.expression);
    for (const auto& instruction : block.top_lp_loader_M)
      replay_coefficients.top_lp_expressions.push_back(instruction.expression);
    for (const auto& instruction : block.top_lp_loader_couplings)
      replay_coefficients.top_lp_expressions.push_back(instruction.expression);
  }
  for (const auto& output : replay_outputs) {
    replay_coefficients.targets.push_back(output.target);
    replay_coefficients.basis.push_back(output.basis);
  }
  complete_coefficient_selection(replay_coefficients);
  kernel_statistics_.block_count = block_count;
  kernel_statistics_.maximum_block_dimension = maximum_block_dimension;
  kernel_statistics_.maximum_block_targets = maximum_block_targets;
  kernel_statistics_.tape_instructions = finalized_tape_instructions;
  kernel_statistics_.tape_opcode_counts = finalized_opcode_counts;
  kernel_statistics_.matrix_slots = matrix_workspace_size;
  kernel_statistics_.rhs_slots = rhs_workspace_size;
  kernel_statistics_.coupling_fmas = finalized_coupling_fmas;
  kernel_statistics_.rhs_ranges = finalized_rhs_ranges;
  kernel_statistics_.rhs_ranged_operations = finalized_rhs_ranged_operations;
  kernel_statistics_.compiled_groups = finalized_compiled_groups;
  kernel_statistics_.compiled_matrix_operations = finalized_compiled_matrix_operations;
  kernel_statistics_.compiled_rhs_operations = finalized_compiled_rhs_operations;
  kernel_statistics_.compiled_rhs_ranges = finalized_rhs_ranges;
  kernel_statistics_.compiled_tape_bytes = finalized_compiled_tape_bytes;
  kernel_statistics_.matrix_contiguous_operations =
      finalized_matrix_contiguous_operations;
  kernel_statistics_.matrix_total_operations = finalized_compiled_matrix_operations;
  kernel_statistics_.top_lp_replay_expressions =
      replay_coefficients.top_lp_expressions.size();
  kernel_statistics_.top_lp_replay_product_nodes =
      replay_coefficients.top_lp_product_nodes.size();
  kernel_statistics_.top_lp_replay_polynomial_factors =
      replay_coefficients.extended_polynomial_factors.size();
  if (progress) {
    progress(std::format("Block replay: layout={}, dimension={}, blocks={}, "
                         "max_block={}, matrix_slots={}, rhs_slots={}, "
                         "coupling_fmas={}, instructions={}",
                         kernel_statistics_.block_layout, square_dim, block_count,
                         maximum_block_dimension, matrix_workspace_size,
                         rhs_workspace_size, finalized_coupling_fmas,
                         finalized_tape_instructions),
             ReductionProgressEvent::info);
  }
}
