#include "reduction/BlackBoxFeynman.hpp"
#include "reduction/FiniteFieldArithmetic.hpp"
#include "reduction/ParameterEvaluation.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

[[nodiscard]] std::vector<firefly::FFInt>
BlackBoxFeynman::execute_replay(const EvaluatedCoeffs<firefly::FFInt>& coeffs,
                                const std::vector<firefly::FFInt>& values,
                                const ReplayVariant* variant) const
{
  static thread_local std::vector<std::uint64_t> matrix;
  static thread_local std::vector<std::uint64_t> right_hand_side;
  static thread_local std::vector<std::uint64_t> target_right_hand_side;
  static thread_local std::vector<std::uint64_t> elimination_factors;
  static thread_local std::vector<std::uint64_t> bilinear_basis;
  static thread_local std::vector<std::uint64_t> top_lp_values;
  static thread_local std::vector<std::uint64_t> target_lp_values;
  static thread_local std::vector<std::uint64_t> basis_lp_inverse_values;
  static thread_local std::vector<std::uint64_t> coefficient_values;
  static thread_local finite_field::MontgomeryArithmetic arithmetic(firefly::FFInt::p);

  arithmetic.reset(firefly::FFInt::p);
  const bool master = replay_orientation == ReplayOrientation::Master;
  const SelectedMasterReplay* selected_master =
      master && variant != nullptr && variant->selected_master.has_value()
          ? &*variant->selected_master
          : nullptr;
  const SelectedMasterStructure* selected_structure =
      selected_master == nullptr ? nullptr : selected_master->structure.get();
  if (selected_master != nullptr && selected_structure == nullptr)
    throw std::logic_error("selected master replay structure is unavailable");
  const auto& blocks = variant == nullptr || master ? replay_blocks : variant->blocks;
  const bool use_selected_rhs_loaders =
      variant != nullptr && (!master || selected_master != nullptr);
  const auto& rhs_loader = use_selected_rhs_loaders ? variant->loader_B : loader_B;
  const auto& pooled_rhs_loader =
      use_selected_rhs_loaders ? variant->pooled_loader_B : pooled_loader_B;
  const auto& top_lp_rhs_loader =
      use_selected_rhs_loaders ? variant->top_lp_loader_B : top_lp_loader_B;
  const auto& outputs = variant == nullptr ? replay_outputs : variant->outputs;
  const auto& coefficient_selection =
      variant == nullptr ? replay_coefficients : variant->coefficients;
  const std::size_t selected_matrix_workspace = variant == nullptr || master
                                                    ? matrix_workspace_size
                                                    : variant->matrix_workspace_size;
  const std::size_t selected_rhs_workspace =
      variant == nullptr || master ? rhs_workspace_size : variant->rhs_workspace_size;

  if (matrix.size() < selected_matrix_workspace)
    matrix.resize(selected_matrix_workspace);
  if (right_hand_side.size() < selected_rhs_workspace)
    right_hand_side.resize(selected_rhs_workspace);
  if (target_right_hand_side.size() < target_rhs_workspace_size)
    target_right_hand_side.resize(target_rhs_workspace_size);

  const size_t num_params = reduction::detail::coefficient_parameter_count(cfg);
  const size_t num_bilinear_basis = 2 * num_params;
  if (bilinear_basis.size() < num_bilinear_basis)
    bilinear_basis.resize(num_bilinear_basis);
  bilinear_basis[0] = arithmetic.one();
  for (size_t index = 0; index < cfg.kinematic_parameters.size(); ++index) {
    bilinear_basis[index + 1] = arithmetic.encode(
        reduction::detail::evaluate_kinematic_parameter(cfg, values, index).n);
  }
  bilinear_basis[num_params] = arithmetic.encode(coeffs.minus_half_d.n);
  for (size_t index = 1; index < num_params; ++index) {
    bilinear_basis[num_params + index] =
        arithmetic.multiply(bilinear_basis[num_params], bilinear_basis[index]);
  }
  if (top_lp_values.size() < coeffs.top_lp_coefficients.size())
    top_lp_values.resize(coeffs.top_lp_coefficients.size());
  for (const std::uint32_t expression : coefficient_selection.top_lp_expressions) {
    if (expression >= coeffs.top_lp_coefficients.size())
      throw std::runtime_error("selected top-LP expression is out of range");
    top_lp_values[expression] =
        arithmetic.encode(coeffs.top_lp_coefficients[expression].n);
  }

  if (coefficient_values.size() < coefficient_expressions.size())
    coefficient_values.resize(coefficient_expressions.size());
  const auto evaluate_pooled_expression = [&](std::size_t expression) {
    if (expression >= coefficient_expressions.size())
      throw std::runtime_error("selected coefficient expression is out of range");
    const auto& program = coefficient_expressions[expression];
    std::uint64_t value = 0;
    const std::size_t end =
        static_cast<std::size_t>(program.term_begin) + program.term_count;
    for (std::size_t term = program.term_begin; term < end; ++term) {
      value = arithmetic.add(
          value, arithmetic.multiply(coefficient_terms[term].w_ff,
                                     bilinear_basis[coefficient_terms[term].bb_idx]));
    }
    coefficient_values[expression] = value;
  };
  if (variant == nullptr) {
    for (std::size_t expression = 0; expression < coefficient_expressions.size();
         ++expression)
      evaluate_pooled_expression(expression);
  } else {
    for (const std::uint32_t expression : variant->coefficients.pooled_expressions)
      evaluate_pooled_expression(expression);
  }

  auto execute_loader = [&](const auto& loader, auto& workspace) {
    for (size_t begin = 0; begin < loader.size();) {
      const auto slot = loader[begin].flat_idx;
      std::uint64_t value = 0;
      const size_t end = begin + loader[begin].group_size;
      for (size_t index = begin; index < end; ++index) {
        value = arithmetic.add(
            value, arithmetic.multiply(loader[index].w_ff,
                                       bilinear_basis[loader[index].bb_idx]));
      }
      workspace[slot] = value;
      begin = end;
    }
  };
  auto execute_pooled_loader = [&](const auto& loader, auto& workspace) {
    for (const auto& instruction : loader) {
      workspace[instruction.flat_idx] = coefficient_values[instruction.expression];
    }
  };
  auto& loaded_rhs = master ? target_right_hand_side : right_hand_side;
  const std::size_t loaded_rhs_size =
      master ? target_rhs_workspace_size : selected_rhs_workspace;
  const auto clear_runs = [](auto& workspace, std::span<const ContiguousSlotRun> runs) {
    for (const auto run : runs) {
      const std::size_t end = static_cast<std::size_t>(run.begin) + run.count;
      if (end > workspace.size())
        throw std::logic_error("selected replay clear run is out of range");
      std::fill(workspace.begin() + static_cast<std::ptrdiff_t>(run.begin),
                workspace.begin() + static_cast<std::ptrdiff_t>(end), std::uint64_t{0});
    }
  };
  if (selected_master != nullptr) {
    clear_runs(loaded_rhs, selected_master->target_clear_runs);
  } else {
    std::fill(loaded_rhs.begin(),
              loaded_rhs.begin() + static_cast<ptrdiff_t>(loaded_rhs_size),
              std::uint64_t{0});
  }
  execute_pooled_loader(pooled_rhs_loader, loaded_rhs);
  execute_loader(rhs_loader, loaded_rhs);
  for (const auto& instruction : top_lp_rhs_loader) {
    if (instruction.expression >= coeffs.top_lp_coefficients.size())
      throw std::runtime_error("top-LP RHS expression is out of range");
    loaded_rhs[instruction.flat_idx] = arithmetic.add(
        loaded_rhs[instruction.flat_idx],
        arithmetic.multiply(instruction.w_ff, top_lp_values[instruction.expression]));
  }

  if (blocks.empty()) {
    throw std::runtime_error("block replay program is unavailable");
  }
  if (master) {
    const auto& active_columns =
        variant == nullptr ? master_rhs_columns : variant->master_columns;
    if (selected_master != nullptr) {
      if (selected_structure->blocks.size() != blocks.size())
        throw std::logic_error("selected master block shape is inconsistent");
      clear_runs(right_hand_side, selected_structure->rhs_clear_runs);
      clear_runs(right_hand_side, selected_master->rhs_clear_runs);
    } else {
      std::fill(right_hand_side.begin(),
                right_hand_side.begin() +
                    static_cast<ptrdiff_t>(selected_rhs_workspace),
                std::uint64_t{0});
    }
    for (const std::uint32_t column : active_columns) {
      const std::uint32_t slot = master_seed_slots[column];
      if (slot != linalg::CompactedTapeLayout::INVALID_SLOT)
        right_hand_side[slot] = arithmetic.one();
    }
    if (elimination_factors.size() < master_factor_workspace_size)
      elimination_factors.resize(master_factor_workspace_size);

    for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
      const auto& block = blocks[block_index];
      if (block.master == nullptr)
        throw std::runtime_error("master replay program is unavailable");
      const auto& master_program = *block.master;
      const SelectedMasterBlock* selected_block =
          selected_structure == nullptr ? nullptr
                                        : &selected_structure->blocks[block_index];
      const auto& slot_ranges = selected_block == nullptr ? master_program.slot_ranges
                                                          : selected_block->slot_ranges;
      const auto for_each_support_range = [&](std::size_t range_begin,
                                              std::size_t range_end, auto&& apply) {
        if (range_begin > range_end || range_end > slot_ranges.size())
          throw std::logic_error("master replay support range is invalid");
        for (std::size_t range_index = range_begin; range_index < range_end;
             ++range_index) {
          const auto range = slot_ranges[range_index];
          apply(range);
        }
      };
      auto apply_transposed_coupling = [&](std::size_t coupling_index,
                                           std::uint64_t coefficient) {
        if (coupling_index >= master_program.couplings.size())
          throw std::runtime_error("master coupling loader index is invalid");
        const auto& couplings = selected_block == nullptr ? master_program.couplings
                                                          : selected_block->couplings;
        if (coupling_index >= couplings.size())
          throw std::runtime_error("selected master coupling index is invalid");
        const auto coupling = couplings[coupling_index];
        for_each_support_range(coupling.range_begin, coupling.range_end,
                               [&](const MasterSlotRange& range) {
                                 auto* destination_value =
                                     right_hand_side.data() + range.destination;
                                 const auto* source_value =
                                     right_hand_side.data() + range.source;
                                 auto* const end = destination_value + range.count;
                                 while (destination_value != end) {
                                   *destination_value = arithmetic.subtract_multiply(
                                       *destination_value, coefficient, *source_value);
                                   ++destination_value;
                                   ++source_value;
                                 }
                               });
      };
      const auto& pooled_couplings = selected_block == nullptr
                                         ? block.pooled_loader_couplings
                                         : selected_block->pooled_loader_couplings;
      const auto& top_lp_couplings = selected_block == nullptr
                                         ? block.top_lp_loader_couplings
                                         : selected_block->top_lp_loader_couplings;
      const auto& direct_couplings = selected_block == nullptr
                                         ? block.loader_couplings
                                         : selected_block->loader_couplings;
      for (const auto& instruction : pooled_couplings) {
        apply_transposed_coupling(instruction.flat_idx,
                                  coefficient_values[instruction.expression]);
      }
      for (const auto& instruction : top_lp_couplings) {
        apply_transposed_coupling(
            instruction.flat_idx,
            arithmetic.multiply(instruction.w_ff,
                                top_lp_values[instruction.expression]));
      }
      for (std::size_t begin = 0; begin < direct_couplings.size();) {
        const std::size_t coupling_index = direct_couplings[begin].flat_idx;
        std::uint64_t coefficient = 0;
        const std::size_t end = begin + direct_couplings[begin].group_size;
        for (std::size_t index = begin; index < end; ++index) {
          coefficient = arithmetic.add(
              coefficient,
              arithmetic.multiply(direct_couplings[index].w_ff,
                                  bilinear_basis[direct_couplings[index].bb_idx]));
        }
        apply_transposed_coupling(coupling_index, coefficient);
        begin = end;
      }

      std::fill(matrix.begin(),
                matrix.begin() + static_cast<ptrdiff_t>(block.matrix_workspace_size),
                std::uint64_t{0});
      execute_pooled_loader(block.pooled_loader_M, matrix);
      execute_loader(block.loader_M, matrix);
      for (const auto& instruction : block.top_lp_loader_M) {
        matrix[instruction.flat_idx] =
            arithmetic.add(matrix[instruction.flat_idx],
                           arithmetic.multiply(instruction.w_ff,
                                               top_lp_values[instruction.expression]));
      }
      if (!linalg::execute_matrix_tape_capture_compact(
              matrix, block.compiled_tape, elimination_factors, arithmetic,
              master_program.factor_capture_slots, master_program.factor_count)) {
        return {};
      }
      const auto& operations = selected_block == nullptr ? master_program.operations
                                                         : selected_block->operations;
      for (const auto operation : operations) {
        const std::uint64_t factor = elimination_factors[operation.factor_group];
        if (operation.kind == MasterOperationKind::Scale) {
          for_each_support_range(operation.range_begin, operation.range_end,
                                 [&](const MasterSlotRange& range) {
                                   auto* value =
                                       right_hand_side.data() + range.destination;
                                   auto* const end = value + range.count;
                                   while (value != end) {
                                     *value = arithmetic.multiply(*value, factor);
                                     ++value;
                                   }
                                 });
        } else {
          for_each_support_range(operation.range_begin, operation.range_end,
                                 [&](const MasterSlotRange& range) {
                                   auto* destination_value =
                                       right_hand_side.data() + range.destination;
                                   const auto* source_value =
                                       right_hand_side.data() + range.source;
                                   auto* const end = destination_value + range.count;
                                   while (destination_value != end) {
                                     *destination_value = arithmetic.subtract_multiply(
                                         *destination_value, factor, *source_value);
                                     ++destination_value;
                                     ++source_value;
                                   }
                                 });
        }
      }
    }
  } else {
    for (std::size_t reverse = blocks.size(); reverse > 0; --reverse) {
      const auto& block = blocks[reverse - 1];
      std::fill(matrix.begin(),
                matrix.begin() + static_cast<ptrdiff_t>(block.matrix_workspace_size),
                std::uint64_t{0});
      execute_pooled_loader(block.pooled_loader_M, matrix);
      execute_loader(block.loader_M, matrix);
      for (const auto& instruction : block.top_lp_loader_M) {
        matrix[instruction.flat_idx] =
            arithmetic.add(matrix[instruction.flat_idx],
                           arithmetic.multiply(instruction.w_ff,
                                               top_lp_values[instruction.expression]));
      }
      if (!linalg::execute_tape_raw(matrix, right_hand_side, block.compiled_tape,
                                    arithmetic)) {
        return {};
      }
      auto apply_coupling = [&](std::size_t coupling_index, std::uint64_t coefficient) {
        if (coupling_index >= block.couplings.size())
          throw std::runtime_error("block coupling loader index is invalid");
        const auto& coupling = block.couplings[coupling_index];
        for (std::size_t operation = coupling.target_begin;
             operation < coupling.target_end; ++operation) {
          const auto slots = block.coupling_operations[operation];
          right_hand_side[slots.destination] =
              arithmetic.subtract_multiply(right_hand_side[slots.destination],
                                           coefficient, right_hand_side[slots.source]);
        }
      };
      for (const auto& instruction : block.pooled_loader_couplings) {
        apply_coupling(instruction.flat_idx,
                       coefficient_values[instruction.expression]);
      }
      for (const auto& instruction : block.top_lp_loader_couplings) {
        apply_coupling(instruction.flat_idx,
                       arithmetic.multiply(instruction.w_ff,
                                           top_lp_values[instruction.expression]));
      }
      for (std::size_t begin = 0; begin < block.loader_couplings.size();) {
        const std::size_t coupling_index = block.loader_couplings[begin].flat_idx;
        std::uint64_t coefficient = 0;
        const std::size_t end = begin + block.loader_couplings[begin].group_size;
        for (std::size_t index = begin; index < end; ++index) {
          coefficient = arithmetic.add(
              coefficient, arithmetic.multiply(
                               block.loader_couplings[index].w_ff,
                               bilinear_basis[block.loader_couplings[index].bb_idx]));
        }
        apply_coupling(coupling_index, coefficient);
        begin = end;
      }
    }
  }

  std::vector<firefly::FFInt> results;
  if (!output_support_initialized)
    throw std::logic_error("reduction output support is not initialized");
  if (variant == nullptr &&
      replay_outputs.size() != reconstructed_output_positions.size())
    throw std::logic_error("replay result slot shape is inconsistent");
  if (target_lp_values.size() < cfg.targets.size())
    target_lp_values.resize(cfg.targets.size());
  if (basis_lp_inverse_values.size() < cfg.basis.size())
    basis_lp_inverse_values.resize(cfg.basis.size());
  for (const std::uint32_t target : coefficient_selection.targets) {
    if (target >= coeffs.targets_lp.size())
      throw std::logic_error("replay target normalization is out of range");
    target_lp_values[target] = arithmetic.encode(coeffs.targets_lp[target].n);
  }
  for (const std::uint32_t basis : coefficient_selection.basis) {
    if (basis >= coeffs.basis_lp_inv.size())
      throw std::logic_error("replay basis normalization is out of range");
    basis_lp_inverse_values[basis] = arithmetic.encode(coeffs.basis_lp_inv[basis].n);
  }
  results.reserve(outputs.size());
  for (const auto& output : outputs) {
    std::uint64_t value = 0;
    if (master) {
      for (std::size_t contraction = output.contraction_begin;
           contraction < output.contraction_end; ++contraction) {
        const auto operation = master_contractions[contraction];
        value = arithmetic.add(
            value, arithmetic.multiply(target_right_hand_side[operation.target_slot],
                                       right_hand_side[operation.master_slot]));
      }
    } else {
      value = right_hand_side[output.rhs_slot];
    }
    value = arithmetic.multiply(value, target_lp_values[output.target]);
    value = arithmetic.multiply(value, basis_lp_inverse_values[output.basis]);
    value = arithmetic.decode(value);
    results.emplace_back(value);
  }
  return results;
}
