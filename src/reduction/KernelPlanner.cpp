#include "reduction/BlackBoxFeynman.hpp"
#include "reduction/EquationGenerator.hpp"
#include "reduction/KernelPlanning.hpp"
#include "reduction/ParameterEvaluation.hpp"
#include "reduction/ProjectedSeedExpansion.hpp"
#include "reduction/SymbolicSignature.hpp"
#include "reduction/SymmetryCanonicalizer.hpp"
#include "topology/IntegralLayout.hpp"
#include "topology/SectorUtils.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct KernelPlanner {
  KernelPlanner(const Config& config, const TopLpTargetPlan* target_plan)
      : sectors(config, config.propagator_slots),
        equations(config, sectors, target_plan),
        canonicalizer(config, sectors, config.propagator_slots)
  {}

  SectorUtils sectors;
  EquationGenerator equations;
  SymmetryCanonicalizer canonicalizer;
};

using reduction::detail::AnsatzColumnMeta;
using reduction::detail::checked_add_i64;
using reduction::detail::checked_mul_i64;
using reduction::detail::IndexedColumns;
using reduction::detail::IndexedTerm;
using reduction::detail::reduction_polynomial_terms;

using reduction::detail::projected_g_shift;
using reduction::detail::ProjectedSeedGroup;
using reduction::detail::ProjectedSeedGroupLess;

std::uint32_t projected_dot_excess(std::span<const int> powers)
{
  if (powers.empty())
    throw std::logic_error("projected seed has no G-power coordinate");
  return reduction::detail::positive_power_sum_u32(
      powers.subspan(1), "projected seed dot excess exceeds uint32");
}

} // namespace

void BlackBoxFeynman::plan_kernel(
    const EvaluatedCoeffs<firefly::FFInt>& coeffs,
    const std::vector<firefly::FFInt>& values,
    const ReductionProgressCallback& progress,
    std::span<const std::uint32_t> relation_source_sectors)
{
  if (numerator_strategy == NumeratorReductionStrategy::Direct) {
    plan_direct_numerator_kernel(coeffs, values, progress, relation_source_sectors);
    return;
  }
  plan_kernel_impl(coeffs, values, progress, relation_source_sectors);
}

void BlackBoxFeynman::plan_kernel_impl(
    const EvaluatedCoeffs<firefly::FFInt>& coeffs,
    const std::vector<firefly::FFInt>& values,
    const ReductionProgressCallback& progress,
    std::span<const std::uint32_t> relation_source_sectors)
{
  using T = firefly::FFInt;
  const TopLpTargetPlan* target_plan =
      top_lp_target_plan.columns.empty() ? nullptr : &top_lp_target_plan;
  KernelPlanner planner(cfg, target_plan);
  const auto variable_slots = std::span<const std::uint32_t>(cfg.propagator_slots);
  const auto polynomial_terms = reduction_polynomial_terms(cfg);

  auto check_basis_representatives = [&](std::span<const Integral> basis) {
    std::unordered_set<std::vector<int>, VectorHash> representatives;
    for (const auto& integral : basis) {
      std::vector<int> powers(variable_slots.size() + 1, 0);
      for (std::size_t variable = 0; variable < variable_slots.size(); ++variable)
        powers[variable + 1] = integral.indices[variable_slots[variable]] - 1;
      auto canonical = planner.canonicalizer.canonicalize(powers);
      if (!representatives.insert(std::move(canonical)).second) {
        throw std::runtime_error("basis contains symmetry-equivalent integrals");
      }
    }
  };
  check_basis_representatives(cfg.basis);

  std::vector<Integral> relation_anchors;
  relation_anchors.reserve(relation_source_sectors.size());
  for (const std::uint32_t sector : relation_source_sectors)
    relation_anchors.push_back(integral_layout::sector_corner(cfg, sector));

  const size_t num_params = reduction::detail::coefficient_parameter_count(cfg);

  // --- 预计算探测阶段初次消元所需的 G 多项式系数 ---
  // 注意：这只在探测期计算一次，热路径 execute_replay 不再重复此步骤
  std::vector<T> polynomial_values(polynomial_terms.size());
  for (size_t i = 0; i < polynomial_terms.size(); ++i) {
    polynomial_values[i] = reduction::detail::evaluate_polynomial_coefficient(
        cfg, polynomial_terms[i], values);
  }

  auto basis_cols = planner.equations.build_basis_columns();
  const auto target_cols = planner.equations.build_target_columns();
  const auto seed_layers = planner.equations.build_ansatz_seed_layers(relation_anchors);
  std::vector<std::vector<int>> envelope_grid;
  const auto initial_domain_start = std::chrono::high_resolution_clock::now();
  std::size_t initial_layer_points = 0;
  for (const auto& seed_layer : seed_layers) {
    auto shifted = planner.equations.build_initial_ansatz_domain(seed_layer);
    initial_layer_points += shifted.size();
    envelope_grid.insert(envelope_grid.end(), std::make_move_iterator(shifted.begin()),
                         std::make_move_iterator(shifted.end()));
  }
  planner.canonicalizer.canonicalize_grid(envelope_grid);
  auto pending_layer_points = initial_layer_points;
  double pending_domain_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::high_resolution_clock::now() - initial_domain_start)
          .count();
  std::map<ProjectedSeedGroup, unsigned, ProjectedSeedGroupLess> expanded_groups;
  std::size_t expansion_rounds = 0;
  std::size_t expanded_points = 0;

  for (;;) {
    const auto& grid = envelope_grid;
    const auto domain_ready = std::chrono::high_resolution_clock::now();
    if (progress) {
      progress(std::format("Ansatz domain: expansion_round={}, layer_points={}, "
                           "canonical_points={}, elapsed_ms={:.2f}",
                           expansion_rounds, pending_layer_points, grid.size(),
                           pending_domain_ms),
               ReductionProgressEvent::info);
    }
    std::vector<std::uint32_t> grid_sectors(grid.size());
    for (std::size_t index = 0; index < grid.size(); ++index) {
      grid_sectors[index] = planner.sectors.sector_from_powers(grid[index]);
    }

    // 行幂次只在这里驻留一次；列随生成随即转换为紧凑 row id。
    struct InternedRow {
      std::uint32_t id = 0;
      std::uint32_t sector = 0;
      std::uint8_t sector_popcount = 0;
    };
    std::unordered_map<std::vector<int>, InternedRow, VectorHash> local_row_map;
    auto intern_row = [&](const std::vector<int>& powers) -> InternedRow {
      if (local_row_map.size() >= std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("equation row count exceeds 32-bit ids");
      }
      auto canonical = planner.canonicalizer.canonicalize(powers);
      const auto [iterator, inserted] = local_row_map.try_emplace(std::move(canonical));
      if (inserted) {
        iterator->second.id = static_cast<std::uint32_t>(local_row_map.size() - 1);
        iterator->second.sector = planner.sectors.sector_from_powers(iterator->first);
        iterator->second.sector_popcount =
            static_cast<std::uint8_t>(std::popcount(iterator->second.sector));
      }
      return iterator->second;
    };
    constexpr std::uint32_t unrestricted_sector =
        std::numeric_limits<std::uint32_t>::max();
    auto append_column = [&](IndexedColumns& output, const auto& column,
                             std::uint32_t seed_sector) {
      for (const auto& term : column) {
        const InternedRow row = intern_row(term.powers);
        if (seed_sector != unrestricted_sector) {
          const auto seed_popcount = std::popcount(seed_sector);
          const auto row_popcount = std::popcount(row.sector);
          const bool changes_same_layer_sector =
              row_popcount == seed_popcount && row.sector != seed_sector;
          if (row_popcount > seed_popcount || changes_same_layer_sector) {
            throw std::runtime_error(
                "canonical ansatz term escapes its seed-sector layer");
          }
        }
        output.terms.push_back({row.id, term.polynomial_term_index,
                                term.coefficient_expression,
                                term.with_polynomial_coefficient, term.coeff_int,
                                term.coeff_minus_half_d});
      }
      output.offsets.push_back(output.terms.size());
    };

    IndexedColumns indexed_basis_cols;
    indexed_basis_cols.offsets.reserve(basis_cols.size() + 1);
    for (const auto& column : basis_cols)
      append_column(indexed_basis_cols, column, unrestricted_sector);

    IndexedColumns indexed_target_cols;
    indexed_target_cols.offsets.reserve(target_cols.size() + 1);
    for (const auto& column : target_cols)
      append_column(indexed_target_cols, column, unrestricted_sector);

    IndexedColumns indexed_ansatz_cols;
    std::vector<AnsatzColumnMeta> ansatz_metadata;
    const std::size_t raw_ansatz_count = (variable_slots.size() + 2) * grid.size();
    indexed_ansatz_cols.offsets.reserve(raw_ansatz_count + 1);
    ansatz_metadata.reserve(raw_ansatz_count);

    const auto for_each_bilinear_weight = [&](const IndexedTerm& term, auto&& emit) {
      if (term.coefficient_expression != std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error(
            "top-LP coefficient expression cannot be expanded as a bilinear term");
      }
      if (term.with_polynomial_coefficient) {
        const auto& weights = polynomial_terms[term.polynomial_term_index].weights;
        for (std::size_t parameter = 0; parameter < num_params; ++parameter) {
          emit(parameter, checked_mul_i64(term.coeff_int, weights[parameter]));
          emit(num_params + parameter,
               checked_mul_i64(term.coeff_minus_half_d, weights[parameter]));
        }
      } else {
        emit(0, term.coeff_int);
        emit(num_params, term.coeff_minus_half_d);
      }
    };

    auto symbolic_signature = [&](std::size_t column) {
      SymbolicSignature signature;
      for (const auto& term : indexed_ansatz_cols.column(column)) {
        auto append_weight = [&](std::size_t bb_idx, std::int64_t weight) {
          if (weight != 0) {
            signature.push_back({static_cast<std::int64_t>(term.row),
                                 static_cast<std::int64_t>(bb_idx), weight});
          }
        };
        if (term.coefficient_expression != std::numeric_limits<std::uint32_t>::max()) {
          const std::uint64_t coefficient_key =
              static_cast<std::uint64_t>(2 * num_params) + term.coefficient_expression;
          if (coefficient_key >
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("top-LP symbolic coefficient key exceeds int64");
          }
          append_weight(static_cast<std::size_t>(coefficient_key), term.coeff_int);
        } else {
          for_each_bilinear_weight(term, append_weight);
        }
      }
      std::ranges::sort(signature);
      std::vector<SymbolicSignatureTerm> merged;
      merged.reserve(signature.size());
      for (const auto& entry : signature) {
        if (!merged.empty() && merged.back()[0] == entry[0] &&
            merged.back()[1] == entry[1]) {
          merged.back()[2] = checked_add_i64(merged.back()[2], entry[2]);
        } else {
          merged.push_back(entry);
        }
      }
      std::erase_if(merged, [](const auto& entry) { return entry[2] == 0; });
      make_primitive_symbolic_signature(merged);
      return merged;
    };
    auto signature_hash = [](const auto& signature) {
      std::size_t seed = signature.size();
      for (const auto& entry : signature) {
        for (const auto value : entry) {
          seed ^= std::hash<std::int64_t>{}(value) + 0x9e3779b9 + (seed << 6U) +
                  (seed >> 2U);
        }
      }
      return seed;
    };
    std::unordered_map<std::size_t, std::vector<SymbolicSignature>> signature_columns;
    signature_columns.reserve(raw_ansatz_count);
    const auto column_stays_in_seed_sector = [&](const std::vector<Monomial>& terms,
                                                 std::uint32_t seed_sector) {
      const auto seed_popcount = std::popcount(seed_sector);
      return std::ranges::all_of(terms, [&](const Monomial& term) {
        const auto canonical = planner.canonicalizer.canonicalize(term.powers);
        const auto row_sector = planner.sectors.sector_from_powers(canonical);
        const auto row_popcount = std::popcount(row_sector);
        return row_popcount < seed_popcount || row_sector == seed_sector;
      });
    };
    auto append_ansatz_candidate = [&](std::size_t stable_seed_index,
                                       std::uint32_t seed_sector, AnsatzFamily family,
                                       unsigned derivative_index,
                                       const std::vector<Monomial>& terms) {
      if (stable_seed_index > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("ansatz seed index exceeds 32-bit ids");
      }
      if (derivative_index > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("ansatz derivative index exceeds 16-bit ids");
      }
      const std::size_t term_begin = indexed_ansatz_cols.terms.size();
      append_column(indexed_ansatz_cols, terms, seed_sector);
      const std::size_t column = indexed_ansatz_cols.size() - 1;
      auto signature = symbolic_signature(column);
      bool duplicate = signature.empty();
      const auto hash = signature_hash(signature);
      if (!duplicate) {
        auto& candidates = signature_columns[hash];
        duplicate = std::ranges::find(candidates, signature) != candidates.end();
        if (!duplicate) candidates.push_back(std::move(signature));
      }
      if (duplicate) {
        indexed_ansatz_cols.terms.resize(term_begin);
        indexed_ansatz_cols.offsets.pop_back();
        return false;
      }
      ansatz_metadata.push_back({static_cast<std::uint32_t>(stable_seed_index),
                                 seed_sector,
                                 static_cast<std::uint16_t>(derivative_index), family});
      ansatz_metadata.back().seed_dot_excess =
          projected_dot_excess(grid[stable_seed_index]);
      return true;
    };
    planner.equations.for_each_ansatz_column(
        grid, [&](std::size_t grid_index, AnsatzFamily family,
                  unsigned derivative_index, std::vector<Monomial>&& terms) {
          const std::uint32_t seed_sector = grid_sectors[grid_index];
          if (!column_stays_in_seed_sector(terms, seed_sector)) return;
          static_cast<void>(append_ansatz_candidate(grid_index, seed_sector, family,
                                                    derivative_index, terms));
        });
    std::vector<bool> matrix_row_present(local_row_map.size(), false);
    const auto mark_matrix_rows = [&](const IndexedColumns& columns) {
      if (matrix_row_present.size() < local_row_map.size())
        matrix_row_present.resize(local_row_map.size(), false);
      for (std::size_t column = 0; column < columns.size(); ++column) {
        for (const auto& term : columns.column(column))
          matrix_row_present[term.row] = true;
      }
    };
    mark_matrix_rows(indexed_basis_cols);
    mark_matrix_rows(indexed_ansatz_cols);

    std::size_t uncovered_rhs_rows = 0;
    for (std::size_t target = 0; target < indexed_target_cols.size(); ++target) {
      for (const auto& term : indexed_target_cols.column(target)) {
        if (term.row >= matrix_row_present.size() || !matrix_row_present[term.row]) {
          ++uncovered_rhs_rows;
        }
      }
    }
    if (target_plan != nullptr && progress) {
      progress(
          std::format("Top-LP row coverage: uncovered_rhs_rows={}", uncovered_rhs_rows),
          ReductionProgressEvent::info);
    }
    if (target_plan != nullptr && uncovered_rhs_rows != 0) {
      throw std::runtime_error(
          "projected nonnegative seed domain does not cover every target row");
    }
    decltype(signature_columns)().swap(signature_columns);

    const std::size_t num_rows = local_row_map.size();
    using RowRecord = decltype(local_row_map)::value_type;
    std::vector<const RowRecord*> row_records;
    row_records.reserve(num_rows);
    for (const auto& record : local_row_map)
      row_records.push_back(&record);
    std::ranges::sort(row_records, [](const auto* lhs, const auto* rhs) {
      if (lhs->second.sector_popcount != rhs->second.sector_popcount)
        return lhs->second.sector_popcount < rhs->second.sector_popcount;
      if (lhs->second.sector != rhs->second.sector)
        return lhs->second.sector < rhs->second.sector;
      return lhs->first < rhs->first;
    });

    std::vector<std::uint32_t> old_to_sector_order(num_rows);
    std::vector<std::uint32_t> row_sectors(num_rows);
    for (std::size_t row = 0; row < row_records.size(); ++row) {
      old_to_sector_order[row_records[row]->second.id] =
          static_cast<std::uint32_t>(row);
      row_sectors[row] = row_records[row]->second.sector;
    }

    auto remap_rows = [](IndexedColumns& columns, const auto& remap) {
      for (auto& term : columns.terms)
        term.row = remap[term.row];
    };
    remap_rows(indexed_basis_cols, old_to_sector_order);
    remap_rows(indexed_target_cols, old_to_sector_order);
    remap_rows(indexed_ansatz_cols, old_to_sector_order);
    auto sector_less = [](std::uint32_t lhs, std::uint32_t rhs) {
      const int lhs_count = std::popcount(lhs);
      const int rhs_count = std::popcount(rhs);
      return lhs_count != rhs_count ? lhs_count < rhs_count : lhs < rhs;
    };
    std::vector<std::uint32_t> target_sectors(indexed_target_cols.size(), 0);
    for (std::size_t target = 0; target < indexed_target_cols.size(); ++target) {
      for (const auto& term : indexed_target_cols.column(target)) {
        const auto sector = row_sectors[term.row];
        if (sector_less(target_sectors[target], sector))
          target_sectors[target] = sector;
      }
    }
    std::vector<ProjectedSeedGroup> residual_row_keys;
    residual_row_keys.reserve(num_rows);
    for (std::size_t row = 0; row < num_rows; ++row) {
      residual_row_keys.push_back(
          {projected_g_shift(row_records[row]->first), row_sectors[row]});
    }
    std::vector<std::uint32_t> row_groups;
    std::vector<std::uint32_t> ordered_groups;
    if (target_plan != nullptr) {
      std::vector<ProjectedSeedGroup> ansatz_group_keys;
      ansatz_group_keys.reserve(ansatz_metadata.size());
      std::set<ProjectedSeedGroup, ProjectedSeedGroupLess> all_group_keys;
      std::set<ProjectedSeedGroup, ProjectedSeedGroupLess> relation_group_keys;
      for (const auto& key : residual_row_keys)
        all_group_keys.insert(key);
      for (std::size_t column = 0; column < indexed_ansatz_cols.size(); ++column) {
        unsigned deepest_shift = 0;
        const auto terms = indexed_ansatz_cols.column(column);
        if (terms.empty())
          throw std::logic_error("projected ansatz column has no equation row");
        for (const auto& term : terms) {
          deepest_shift =
              std::max(deepest_shift, projected_g_shift(row_records[term.row]->first));
        }
        const ProjectedSeedGroup key{deepest_shift,
                                     ansatz_metadata[column].seed_sector};
        ansatz_group_keys.push_back(key);
        all_group_keys.insert(key);
        relation_group_keys.insert(key);
      }
      if (all_group_keys.size() >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("projected G-layer groups exceed uint32");
      }
      std::map<ProjectedSeedGroup, std::uint32_t, ProjectedSeedGroupLess> group_ids;
      for (const auto& key : all_group_keys) {
        group_ids.emplace(key, static_cast<std::uint32_t>(group_ids.size()));
      }
      row_groups.reserve(num_rows);
      for (const auto& key : residual_row_keys)
        row_groups.push_back(group_ids.at(key));
      for (std::size_t column = 0; column < ansatz_metadata.size(); ++column)
        ansatz_metadata[column].pivot_group = group_ids.at(ansatz_group_keys[column]);
      ordered_groups.reserve(relation_group_keys.size());
      for (const auto& key : relation_group_keys)
        ordered_groups.push_back(group_ids.at(key));
    } else {
      row_groups = row_sectors;
    }
    decltype(local_row_map)().swap(local_row_map);
    decltype(row_records)().swap(row_records);
    const auto candidate_ready = std::chrono::high_resolution_clock::now();

    const size_t num_basis_cols = indexed_basis_cols.size();
    const size_t provisional_ansatz_cols = indexed_ansatz_cols.size();
    const size_t generated_column_count = num_basis_cols + provisional_ansatz_cols;
    const auto effective_dot_ordering = resolve_ansatz_dot_ordering(
        ansatz_dot_ordering, target_plan != nullptr, expansion_rounds);
    switch (effective_dot_ordering) {
    case AnsatzDotOrdering::Auto:
      throw std::logic_error("automatic ansatz dot ordering was not resolved");
    case AnsatzDotOrdering::Markowitz:
      kernel_statistics_.compact_policy =
          target_plan == nullptr ? "sector-markowitz" : "g-layer-sector-markowitz";
      break;
    case AnsatzDotOrdering::LowFirst:
      kernel_statistics_.compact_policy = target_plan == nullptr
                                              ? "sector-low-dot-markowitz"
                                              : "g-layer-sector-low-dot-markowitz";
      break;
    case AnsatzDotOrdering::HighFirst:
      kernel_statistics_.compact_policy = target_plan == nullptr
                                              ? "sector-high-dot-markowitz"
                                              : "g-layer-sector-high-dot-markowitz";
      break;
    }
    auto selection = reduction::detail::plan_compact_kernel(
        indexed_basis_cols, indexed_target_cols, indexed_ansatz_cols, ansatz_metadata,
        row_sectors, polynomial_values, coeffs.top_lp_coefficients, coeffs.minus_half_d,
        row_groups, ordered_groups, effective_dot_ordering, cfg.threads);
    if (!selection.closed) {
      if (progress) {
        progress(std::format("Kernel elimination: expansion_round={}, rows={}, "
                             "generated_columns={}, "
                             "residual_rows={}, closed=false, elapsed_ms={:.2f}",
                             expansion_rounds, num_rows, generated_column_count,
                             selection.residual_rows.size(),
                             selection.timings.total_ms),
                 ReductionProgressEvent::info);
      }

      const auto expansion_start = std::chrono::high_resolution_clock::now();
      std::set<ProjectedSeedGroup, ProjectedSeedGroupLess> residual_groups;
      std::set<ProjectedSeedGroup, ProjectedSeedGroupLess> requested_groups;
      const unsigned deepest_seed_shift = seed_layers.back().g_shift;
      for (const std::size_t row : selection.residual_rows) {
        if (row >= residual_row_keys.size()) {
          throw std::logic_error("compact residual row is out of range");
        }
        const auto residual_key = residual_row_keys[row];
        residual_groups.insert(residual_key);
        requested_groups.insert(
            {std::min(residual_key.g_shift, deepest_seed_shift), residual_key.sector});
      }

      const std::size_t previous_points = envelope_grid.size();
      const std::vector requested(requested_groups.begin(), requested_groups.end());
      auto expansion = reduction::detail::expand_projected_seed_groups(
          envelope_grid, requested, expanded_groups, planner.sectors,
          planner.canonicalizer);
      envelope_grid.insert(envelope_grid.end(),
                           std::make_move_iterator(expansion.points.begin()),
                           std::make_move_iterator(expansion.points.end()));
      planner.canonicalizer.canonicalize_grid(envelope_grid);
      const std::size_t added_points = envelope_grid.size() - previous_points;

      const auto format_groups = [](const auto& groups) {
        std::string text;
        for (const auto& key : groups) {
          if (!text.empty()) text += ',';
          text += std::format("g{}:s{}", key.g_shift, key.sector);
        }
        return text;
      };
      if (progress) {
        progress(std::format(
                     "Projected residual expansion: round={}, residual_groups={}, "
                     "added_groups={}, added_points={}, residual_keys=[{}], "
                     "added_keys=[{}]",
                     expansion_rounds + 1, residual_groups.size(),
                     expansion.groups.size(), added_points,
                     format_groups(residual_groups), format_groups(expansion.groups)),
                 ReductionProgressEvent::info);
      }
      if (selection.residual_rows.empty() || expansion.groups.empty() ||
          added_points == 0) {
        throw AnsatzClosureError();
      }
      ++expansion_rounds;
      expanded_points += added_points;
      pending_layer_points = added_points;
      pending_domain_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::high_resolution_clock::now() - expansion_start)
              .count();
      continue;
    }
    if (progress) {
      const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
      };
      progress(std::format("Compact phases: generated_assembly_ms={:.2f}, "
                           "provisional_rhs={}, provisional_fallback={}, "
                           "provisional_build_ms={:.2f}, "
                           "provisional_elimination_ms={:.2f}, "
                           "relation_elimination_ms={:.2f}, "
                           "score_refresh_ms={:.2f}, "
                           "row_elimination_ms={:.2f}, "
                           "back_substitution_ms={:.2f}, "
                           "support_selection_ms={:.2f}, compact_build_ms={:.2f}, "
                           "compact_elimination_ms={:.2f}",
                           milliseconds(domain_ready, candidate_ready),
                           selection.provisional_rhs_columns,
                           selection.provisional_rhs_fallback,
                           selection.timings.provisional_build_ms,
                           selection.timings.provisional_elimination_ms,
                           selection.timings.provisional_relation_elimination_ms,
                           selection.timings.provisional_score_refresh_ms,
                           selection.timings.provisional_row_elimination_ms,
                           selection.timings.provisional_back_substitution_ms,
                           selection.timings.support_selection_ms,
                           selection.timings.compact_build_ms,
                           selection.timings.compact_elimination_ms),
               ReductionProgressEvent::info);
      progress(
          std::format("Compact provisional ansatz: policy={}, "
                      "provisional_dimension={}, live_ansatz_columns={}, "
                      "compact_dimension={}, cross_group_pivots={}",
                      kernel_statistics_.compact_policy,
                      selection.provisional_dimension, selection.ansatz_order.size(),
                      selection.solution_columns.size(), selection.cross_group_pivots),
          ReductionProgressEvent::info);
      progress(
          std::format("Compact planning work: score_columns={}, incidence_scans={}, "
                      "parallel_refresh_batches={}, parallel_refresh_columns={}, "
                      "stale_choices={}, row_eliminations={}, parallel_row_batches={}, "
                      "parallel_row_eliminations={}",
                      selection.provisional_score_refresh_columns,
                      selection.provisional_incidence_records_scanned,
                      selection.provisional_parallel_refresh_batches,
                      selection.provisional_parallel_refresh_columns,
                      selection.provisional_stale_choice_pops,
                      selection.provisional_row_eliminations,
                      selection.provisional_parallel_row_batches,
                      selection.provisional_parallel_row_eliminations),
          ReductionProgressEvent::info);
    }

    publish_ansatz_statistics(selection, ansatz_metadata, num_rows,
                              generated_column_count, expansion_rounds,
                              expanded_groups.size(), expanded_points);
    std::vector<std::size_t> ansatz_order = std::move(selection.ansatz_order);
    std::vector<std::size_t> solution_cols = std::move(selection.solution_columns);
    std::vector<std::size_t> elim_row_map = std::move(selection.elimination_row_map);
    const std::size_t provisional_dimension = selection.provisional_dimension;
    const std::size_t num_ansatz_cols = ansatz_order.size();
    if (progress) {
      progress(std::format("Compact selection: policy={}, expansion_rounds={}, "
                           "expanded_groups={}, "
                           "expanded_points={}, rows={}, "
                           "generated_columns={}, provisional_dimension={}, "
                           "live_ansatz_columns={}, compact_dimension={}, "
                           "cross_group_pivots={}, compression_ratio={:.3f}, "
                           "elapsed_ms={:.2f}",
                           kernel_statistics_.compact_policy, expansion_rounds,
                           expanded_groups.size(), expanded_points, num_rows,
                           generated_column_count, provisional_dimension,
                           num_ansatz_cols, solution_cols.size(),
                           selection.cross_group_pivots,
                           generated_column_count == 0
                               ? 1.0
                               : static_cast<double>(solution_cols.size()) /
                                     static_cast<double>(generated_column_count),
                           selection.timings.total_ms),
               ReductionProgressEvent::info);
    }
    publish_kernel_plan({std::move(indexed_basis_cols), std::move(indexed_target_cols),
                         std::move(indexed_ansatz_cols), std::move(ansatz_metadata),
                         std::move(row_sectors), std::move(row_groups),
                         std::move(ordered_groups), std::move(target_sectors),
                         std::move(ansatz_order), std::move(solution_cols),
                         std::move(elim_row_map), std::move(polynomial_values)},
                        coeffs, values);
    return;
  }
}
