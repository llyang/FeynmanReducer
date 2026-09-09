#include "reduction/BlackBoxFeynman.hpp"
#include "reduction/KernelErrors.hpp"

#include "reduction/DirectSeedExpansion.hpp"
#include "reduction/JetEquationGenerator.hpp"
#include "reduction/JetSymmetryCanonicalizer.hpp"
#include "reduction/KernelPlanning.hpp"
#include "reduction/ParameterEvaluation.hpp"
#include "reduction/SymbolicSignature.hpp"
#include "reduction/SymmetryCanonicalizer.hpp"
#include "topology/IntegralLayout.hpp"
#include "topology/SectorUtils.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <format>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
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

using reduction::detail::AnsatzColumnMeta;
using reduction::detail::checked_add_i64;
using reduction::detail::checked_mul_i64;
using reduction::detail::CompactSelection;
using reduction::detail::IndexedColumns;
using reduction::detail::IndexedTerm;

struct BuiltJetSystem {
  IndexedColumns basis;
  IndexedColumns targets;
  IndexedColumns ansatz;
  std::vector<AnsatzColumnMeta> metadata;
  std::vector<std::vector<int>> row_powers;
  std::vector<std::uint32_t> row_sectors;
  std::vector<std::uint32_t> row_groups;
  std::vector<std::uint32_t> ordered_groups;
  std::vector<std::uint32_t> target_sectors;
  std::vector<bool> rhs_rank_zero;
  CompactSelection selection;
  std::size_t group_count = 0;
};

struct ExplicitJetRelation {
  std::vector<int> source;
  AnsatzFamily family = AnsatzFamily::Nabla;
  unsigned derivative_slot = 0;
  std::vector<int> required_state;

  auto operator<=>(const ExplicitJetRelation&) const = default;
};

struct DirectBlockGroup {
  unsigned g_shift = 0;
  std::uint32_t total_rank = 0;
  reduction::detail::DirectSeedGroup state;
};

struct DirectBlockGroupLess {
  DirectGroupOrdering grouping = DirectGroupOrdering::ExactJet;
  DirectRankOrdering rank_ordering = DirectRankOrdering::LowFirst;

  bool operator()(const DirectBlockGroup& lhs, const DirectBlockGroup& rhs) const
  {
    if (lhs.g_shift != rhs.g_shift) return lhs.g_shift < rhs.g_shift;
    const auto rank_less = [&](std::uint32_t left, std::uint32_t right) {
      return rank_ordering == DirectRankOrdering::LowFirst ? left < right
                                                           : left > right;
    };
    if (grouping == DirectGroupOrdering::SectorRank) {
      const auto lhs_sector =
          std::tuple(std::popcount(lhs.state.sector), lhs.state.sector);
      const auto rhs_sector =
          std::tuple(std::popcount(rhs.state.sector), rhs.state.sector);
      if (lhs_sector != rhs_sector) return lhs_sector < rhs_sector;
      return lhs.total_rank != rhs.total_rank
                 ? rank_less(lhs.total_rank, rhs.total_rank)
                 : false;
    }
    if (lhs.total_rank != rhs.total_rank)
      return rank_less(lhs.total_rank, rhs.total_rank);
    const auto lhs_sector =
        std::tuple(std::popcount(lhs.state.sector), lhs.state.sector);
    const auto rhs_sector =
        std::tuple(std::popcount(rhs.state.sector), rhs.state.sector);
    if (lhs_sector != rhs_sector) return lhs_sector < rhs_sector;
    if (grouping == DirectGroupOrdering::RankSector) return false;
    return lhs.state.orders < rhs.state.orders;
  }
};

struct RelationPredecessorExpansion {
  std::vector<ExplicitJetRelation> relations;
  std::vector<std::vector<int>> sources;
  std::vector<std::vector<int>> states;
  std::size_t raw_sources = 0;
  std::size_t same_layer_relations = 0;
  std::size_t previous_layer_relations = 0;
  std::size_t required_states = 0;
  unsigned maximum_g_shift = 0;
};

struct DirectResidualAnalysis {
  std::vector<std::vector<int>> powers;
  std::set<reduction::detail::DirectSeedGroup, reduction::detail::DirectSeedGroupLess>
      local_dot_groups;
  std::size_t layer_zero_rows = 0;
  std::size_t layer_one_rows = 0;
  std::size_t rank_zero_layer_one_rows = 0;
};

unsigned direct_g_shift(std::span<const int> powers)
{
  if (powers.empty() || powers.front() > 0)
    throw std::invalid_argument("direct jet state has a positive G power");
  const auto shift = -static_cast<std::int64_t>(powers.front());
  if (shift > std::numeric_limits<unsigned>::max())
    throw std::overflow_error("direct jet G shift exceeds unsigned range");
  return static_cast<unsigned>(shift);
}

std::uint32_t direct_total_rank(const reduction::detail::DirectSeedGroup& group)
{
  const std::uint64_t rank =
      std::reduce(group.orders.begin(), group.orders.end(), std::uint64_t{0});
  if (rank > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("direct jet total rank exceeds uint32");
  return static_cast<std::uint32_t>(rank);
}

std::uint32_t seed_dot_excess(std::span<const int> powers)
{
  const auto exponents = powers.empty() ? powers : powers.subspan(1);
  return reduction::detail::positive_power_sum_u32(
      exponents, "direct seed dot excess exceeds uint32");
}

std::vector<int> integral_powers(const Config& config, const Integral& integral)
{
  if (integral.indices.size() != config.integral_count)
    throw std::invalid_argument("jet integral has the wrong dimension");
  std::vector<int> powers(config.integral_count + 1, -1);
  powers[0] = 0;
  for (std::size_t slot = 0; slot < config.integral_count; ++slot) {
    const int index = integral.indices[slot];
    if (config.top_sector[slot] == 0 && index > 0)
      throw std::invalid_argument("positive ISP indices are not supported");
    if (index == std::numeric_limits<int>::min())
      throw std::overflow_error("negative integral index exceeds jet encoding");
    powers[slot + 1] = index - 1;
  }
  return powers;
}

} // namespace

void BlackBoxFeynman::plan_direct_numerator_kernel(
    const EvaluatedCoeffs<firefly::FFInt>& coeffs,
    const std::vector<firefly::FFInt>& values,
    const ReductionProgressCallback& progress,
    std::span<const std::uint32_t> relation_source_sectors)
{
  using T = firefly::FFInt;
  SectorUtils sectors(cfg, cfg.propagator_slots);
  JetEquationGenerator equations(cfg, sectors);
  SymmetryCanonicalizer denominator_symmetry(cfg, sectors, cfg.propagator_slots);
  JetSymmetryCanonicalizer symmetry(cfg, denominator_symmetry);
  const auto& polynomial_terms = cfg.extended_lp.polynomial_terms;
  const std::size_t num_params = reduction::detail::coefficient_parameter_count(cfg);
  const auto integral_has_rank = [](const Integral& integral) {
    return std::ranges::any_of(integral.indices,
                               [](const int index) { return index < 0; });
  };

  const auto check_basis_representatives = [&](std::span<const Integral> basis) {
    std::set<std::vector<int>> representatives;
    for (const auto& integral : basis) {
      auto canonical = symmetry.canonicalize(integral_powers(cfg, integral));
      if (!representatives.insert(std::move(canonical)).second)
        throw reduction::detail::SymmetryEquivalentBasisError();
    }
  };
  check_basis_representatives(cfg.basis);

  std::vector<T> polynomial_values(polynomial_terms.size());
  for (std::size_t term = 0; term < polynomial_terms.size(); ++term) {
    polynomial_values[term] = reduction::detail::evaluate_polynomial_coefficient(
        cfg, polynomial_terms[term], values);
  }

  const auto build_unit_columns = [&](std::span<const Integral> integrals) {
    std::vector<std::vector<Monomial>> columns;
    columns.reserve(integrals.size());
    for (const auto& integral : integrals) {
      auto powers = symmetry.canonicalize(integral_powers(cfg, integral));
      if (!sectors.is_valid_sector(equations.sector_from_powers(powers))) {
        columns.push_back({});
      } else {
        columns.push_back({{std::move(powers), false, 0, 1, 0}});
      }
    }
    return columns;
  };
  const auto basis_columns = build_unit_columns(cfg.basis);
  const auto target_columns = build_unit_columns(cfg.targets);
  std::vector<Integral> closure_guides;
  auto closure_guide_columns = build_unit_columns(closure_guides);

  std::vector<Integral> relation_anchors;
  relation_anchors.reserve(relation_source_sectors.size());
  for (const std::uint32_t sector : relation_source_sectors) {
    auto corner = integral_layout::sector_corner(cfg, sector);
    relation_anchors.push_back(corner);
    for (std::size_t variable = 0; variable < cfg.propagator_count; ++variable) {
      if (((sector >> variable) & 1U) == 0) continue;
      auto dot = corner;
      const auto slot = cfg.propagator_slots[variable];
      if (dot.indices[slot] == std::numeric_limits<int>::max())
        throw std::overflow_error("relation-source single dot exceeds int range");
      ++dot.indices[slot];
      relation_anchors.push_back(dot);
    }
  }

  const auto build_system = [&](const std::vector<std::vector<int>>& seeds,
                                DirectGroupOrdering group_ordering,
                                DirectRankOrdering rank_ordering,
                                AnsatzDotOrdering dot_ordering) {
    BuiltJetSystem system;
    std::map<DirectBlockGroup, std::uint32_t, DirectBlockGroupLess> group_ids(
        DirectBlockGroupLess{group_ordering, rank_ordering});
    const auto intern_group = [&](std::span<const int> powers) {
      auto state = reduction::detail::direct_seed_group(powers, equations, symmetry);
      DirectBlockGroup key{0, direct_total_rank(state), std::move(state)};
      const auto [found, inserted] = group_ids.try_emplace(std::move(key), 0);
      if (inserted) {
        if (group_ids.size() > std::numeric_limits<std::uint32_t>::max())
          throw std::runtime_error("jet group count exceeds uint32");
        found->second = static_cast<std::uint32_t>(group_ids.size() - 1);
      }
      return found->second;
    };

    struct InternedRow {
      std::uint32_t id = 0;
      std::uint32_t sector = 0;
      std::uint32_t group = 0;
    };
    std::unordered_map<std::vector<int>, InternedRow, VectorHash> row_map;
    const auto intern_row = [&](std::span<const int> powers) {
      auto canonical = symmetry.canonicalize(powers);
      const auto [found, inserted] = row_map.try_emplace(std::move(canonical));
      if (inserted) {
        if (row_map.size() > std::numeric_limits<std::uint32_t>::max())
          throw std::runtime_error("jet row count exceeds uint32");
        found->second.id = static_cast<std::uint32_t>(row_map.size() - 1);
        found->second.sector = equations.sector_from_powers(found->first);
        found->second.group = intern_group(found->first);
      }
      return found->second;
    };
    const auto append_column = [&](IndexedColumns& destination,
                                   const std::vector<Monomial>& column) {
      for (const auto& term : column) {
        const auto row = intern_row(term.powers);
        destination.terms.push_back(IndexedTerm{
            row.id, term.polynomial_term_index, term.coefficient_expression,
            term.with_polynomial_coefficient, term.coeff_int, term.coeff_minus_half_d});
      }
      destination.offsets.push_back(destination.terms.size());
    };
    for (const auto& column : basis_columns)
      append_column(system.basis, column);
    for (const auto& column : target_columns)
      append_column(system.targets, column);
    IndexedColumns planning_targets = system.targets;
    system.rhs_rank_zero.reserve(cfg.targets.size() + closure_guides.size());
    for (const auto& target : cfg.targets)
      system.rhs_rank_zero.push_back(!integral_has_rank(target));
    for (const auto& column : closure_guide_columns)
      append_column(planning_targets, column);
    for (const auto& guide : closure_guides)
      system.rhs_rank_zero.push_back(!integral_has_rank(guide));
    const auto for_each_weight = [&](const IndexedTerm& term, auto&& emit) {
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
    const auto signature_for = [&](std::size_t column) {
      SymbolicSignature signature;
      for (const auto& term : system.ansatz.column(column)) {
        for_each_weight(term, [&](std::size_t coefficient, std::int64_t weight) {
          if (weight != 0)
            signature.push_back({static_cast<std::int64_t>(term.row),
                                 static_cast<std::int64_t>(coefficient), weight});
        });
      }
      std::ranges::sort(signature);
      SymbolicSignature merged;
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
    std::set<SymbolicSignature> signatures;
    const auto append_ansatz = [&](std::size_t seed_index, AnsatzFamily family,
                                   unsigned derivative_slot, std::uint32_t pivot_group,
                                   std::vector<Monomial> column) {
      if (column.empty()) return;
      const auto seed_sector = equations.sector_from_powers(seeds[seed_index]);
      const std::size_t begin = system.ansatz.terms.size();
      append_column(system.ansatz, column);
      auto signature = signature_for(system.ansatz.size() - 1);
      if (signature.empty() || !signatures.insert(std::move(signature)).second) {
        system.ansatz.terms.resize(begin);
        system.ansatz.offsets.pop_back();
        return;
      }
      if (seed_index > std::numeric_limits<std::uint32_t>::max() ||
          derivative_slot > std::numeric_limits<std::uint16_t>::max())
        throw std::runtime_error("jet ansatz metadata exceeds compact encoding");
      system.metadata.push_back({static_cast<std::uint32_t>(seed_index), seed_sector,
                                 static_cast<std::uint16_t>(derivative_slot), family,
                                 pivot_group});
      system.metadata.back().seed_dot_excess = seed_dot_excess(seeds[seed_index]);
    };

    const std::unordered_set<std::vector<int>, VectorHash> seed_set(seeds.begin(),
                                                                    seeds.end());
    for (std::size_t seed_index = 0; seed_index < seeds.size(); ++seed_index) {
      const auto& seed = seeds[seed_index];
      const std::uint32_t seed_group = intern_group(seed);
      for (std::size_t slot = 0; slot < cfg.integral_count; ++slot) {
        bool allow_boundary_raise = false;
        std::uint32_t pivot_group = seed_group;
        if (seed[slot + 1] < 0 && seed[slot + 1] != std::numeric_limits<int>::min()) {
          auto raised = seed;
          --raised[slot + 1];
          raised = symmetry.canonicalize(raised);
          allow_boundary_raise = seed_set.contains(raised);
          if (!allow_boundary_raise) continue;
          pivot_group = intern_group(raised);
        }
        append_ansatz(seed_index, AnsatzFamily::Nabla, static_cast<unsigned>(slot + 1),
                      pivot_group,
                      equations.build_column(seed, AnsatzFamily::Nabla,
                                             static_cast<unsigned>(slot),
                                             allow_boundary_raise));
      }
      append_ansatz(seed_index, AnsatzFamily::Euler, 0, seed_group,
                    equations.build_column(seed, AnsatzFamily::Euler));
      append_ansatz(seed_index, AnsatzFamily::DimensionShift, 0, seed_group,
                    equations.build_column(seed, AnsatzFamily::DimensionShift));
    }
    const std::size_t num_rows = row_map.size();
    using RowRecord = decltype(row_map)::value_type;
    std::vector<const RowRecord*> rows;
    rows.reserve(num_rows);
    for (const auto& row : row_map)
      rows.push_back(&row);
    std::unordered_map<std::uint32_t, std::size_t> group_rank;
    std::size_t rank = 0;
    for (const auto& [key, id] : group_ids)
      group_rank[id] = rank++;
    std::ranges::sort(rows, [&](const auto* lhs, const auto* rhs) {
      if (group_rank.at(lhs->second.group) != group_rank.at(rhs->second.group))
        return group_rank.at(lhs->second.group) < group_rank.at(rhs->second.group);
      return lhs->first < rhs->first;
    });
    std::vector<std::uint32_t> old_to_new(num_rows);
    system.row_powers.reserve(num_rows);
    system.row_sectors.resize(num_rows);
    system.row_groups.resize(num_rows);
    for (std::size_t row = 0; row < rows.size(); ++row) {
      old_to_new[rows[row]->second.id] = static_cast<std::uint32_t>(row);
      system.row_powers.push_back(rows[row]->first);
      system.row_sectors[row] = rows[row]->second.sector;
      system.row_groups[row] = rows[row]->second.group;
    }
    for (auto* columns :
         {&system.basis, &system.targets, &planning_targets, &system.ansatz}) {
      for (auto& term : columns->terms)
        term.row = old_to_new[term.row];
    }

    std::set<std::uint32_t> relation_groups;
    for (const auto& meta : system.metadata)
      relation_groups.insert(reduction::detail::ansatz_pivot_group(meta));
    for (const auto& [key, id] : group_ids) {
      (void)key;
      if (relation_groups.contains(id)) system.ordered_groups.push_back(id);
    }
    system.group_count = group_ids.size();
    system.target_sectors.assign(system.targets.size(), 0);
    for (std::size_t target = 0; target < system.targets.size(); ++target) {
      for (const auto& term : system.targets.column(target)) {
        const auto sector = system.row_sectors[term.row];
        if (std::tuple(std::popcount(system.target_sectors[target]),
                       system.target_sectors[target]) <
            std::tuple(std::popcount(sector), sector))
          system.target_sectors[target] = sector;
      }
    }

    system.selection = reduction::detail::plan_compact_kernel(
        system.basis, planning_targets, system.ansatz, system.metadata,
        system.row_sectors, polynomial_values, {}, coeffs.minus_half_d,
        system.row_groups, system.ordered_groups, dot_ordering, cfg.threads,
        check_master_independence);
    if (check_master_independence && progress)
      progress(std::format(
                   "Master independence check: completion_ms={:.2f}, check_ms={:.2f}",
                   system.selection.timings.master_rank_completion_ms,
                   system.selection.timings.master_rank_check_ms),
               ReductionProgressEvent::info);
    return system;
  };

  const auto relation_predecessors = [&](std::span<const std::vector<int>> residuals,
                                         std::span<const std::vector<int>> states) {
    RelationPredecessorExpansion expansion;
    std::set<std::vector<int>> raw_sources;
    const auto append_raw_source = [&](std::vector<int> source) {
      source = symmetry.canonicalize(source);
      if (source.front() != 0) return;
      const auto sector = equations.sector_from_powers(source);
      if (!sectors.is_valid_sector(sector)) return;
      for (std::size_t slot = 0; slot < cfg.integral_count; ++slot) {
        if (cfg.top_sector[slot] == 0 && source[slot + 1] >= 0) return;
      }
      raw_sources.insert(std::move(source));
    };
    for (const auto& raw_residual : residuals) {
      const auto residual = symmetry.canonicalize(raw_residual);
      const unsigned residual_shift = direct_g_shift(residual);
      if (residual_shift > 1) {
        throw AnsatzClosureError(
            "direct layer-0 sources produced a residual below G^-1");
      }
      if (residual_shift == 0) {
        append_raw_source(residual);
        for (std::size_t slot = 0; slot < cfg.integral_count; ++slot) {
          if (residual[slot + 1] == std::numeric_limits<int>::max())
            throw std::overflow_error("direct predecessor source exceeds int range");
          auto source = residual;
          ++source[slot + 1];
          append_raw_source(std::move(source));
        }
        continue;
      }
      for (const auto& polynomial : polynomial_terms) {
        auto source = residual;
        ++source.front();
        for (std::size_t slot = 0; slot < cfg.integral_count; ++slot) {
          const int exponent = polynomial.powers[slot];
          if (source[slot + 1] < std::numeric_limits<int>::min() + exponent)
            throw std::overflow_error("direct predecessor power exceeds int range");
          source[slot + 1] -= exponent;
        }
        append_raw_source(source);
        for (std::size_t slot = 0; slot < cfg.integral_count; ++slot) {
          if (polynomial.powers[slot] == 0) continue;
          auto nabla_source = source;
          if (nabla_source[slot + 1] == std::numeric_limits<int>::max())
            throw std::overflow_error("direct predecessor power exceeds int range");
          ++nabla_source[slot + 1];
          append_raw_source(std::move(nabla_source));
        }
      }
    }
    expansion.raw_sources = raw_sources.size();

    const std::set<std::vector<int>> residual_set(residuals.begin(), residuals.end());
    const std::set<std::vector<int>> state_set(states.begin(), states.end());
    std::set<std::vector<int>> new_states;
    std::set<std::vector<int>> relation_sources;
    std::set<std::vector<int>> required_states;
    const auto matches_residual = [&](const std::vector<Monomial>& column) {
      return std::ranges::any_of(column, [&](const Monomial& term) {
        if (term.coeff_int == 0 && term.coeff_minus_half_d == 0) return false;
        return residual_set.contains(symmetry.canonicalize(term.powers));
      });
    };
    const auto append_relation = [&](const std::vector<int>& source,
                                     AnsatzFamily family, unsigned derivative_slot) {
      if (source.front() != 0)
        throw std::logic_error("direct predecessor source is not in G^0");
      std::vector<int> required;
      bool allow_boundary_raise = false;
      if (family == AnsatzFamily::Nabla && source[derivative_slot + 1] < 0) {
        required = source;
        if (required[derivative_slot + 1] == std::numeric_limits<int>::min())
          throw std::overflow_error("direct predecessor jet order exceeds int range");
        --required[derivative_slot + 1];
        required = symmetry.canonicalize(required);
        if (!sectors.is_valid_sector(equations.sector_from_powers(required))) return;
        allow_boundary_raise = true;
      }
      auto column =
          equations.build_column(source, family, derivative_slot, allow_boundary_raise);
      if (!matches_residual(column)) return;
      relation_sources.insert(source);
      ExplicitJetRelation relation{source, family, derivative_slot, required};
      expansion.relations.push_back(std::move(relation));
      const unsigned source_shift = direct_g_shift(source);
      expansion.maximum_g_shift = std::max(expansion.maximum_g_shift, source_shift);
      if (std::ranges::any_of(column, [&](const Monomial& term) {
            return residual_set.contains(symmetry.canonicalize(term.powers)) &&
                   direct_g_shift(term.powers) == source_shift;
          })) {
        ++expansion.same_layer_relations;
      } else {
        ++expansion.previous_layer_relations;
      }
      if (!state_set.contains(source)) new_states.insert(source);
      if (!required.empty() && !state_set.contains(required)) {
        new_states.insert(required);
        required_states.insert(std::move(required));
      }
    };
    for (const auto& source : raw_sources) {
      append_relation(source, AnsatzFamily::Euler, 0);
      append_relation(source, AnsatzFamily::DimensionShift, 0);
      for (std::size_t slot = 0; slot < cfg.integral_count; ++slot)
        append_relation(source, AnsatzFamily::Nabla, static_cast<unsigned>(slot));
    }
    std::ranges::sort(expansion.relations);
    expansion.relations.erase(
        std::unique(expansion.relations.begin(), expansion.relations.end()),
        expansion.relations.end());
    expansion.states.assign(new_states.begin(), new_states.end());
    expansion.sources.assign(relation_sources.begin(), relation_sources.end());
    expansion.required_states = required_states.size();
    return expansion;
  };

  const auto analyze_residuals = [&](const BuiltJetSystem& system) {
    if (system.selection.residual_rhs_support.size() !=
        system.selection.residual_rows.size()) {
      throw std::logic_error("direct residual RHS support is incomplete");
    }
    DirectResidualAnalysis analysis;
    analysis.powers.reserve(system.selection.residual_rows.size());
    for (std::size_t residual_index = 0;
         residual_index < system.selection.residual_rows.size(); ++residual_index) {
      const std::size_t row = system.selection.residual_rows[residual_index];
      const auto& powers = system.row_powers.at(row);
      const unsigned shift = direct_g_shift(powers);
      if (shift > 1) {
        throw AnsatzClosureError(
            "direct layer-0 sources produced a residual below G^-1");
      }
      analysis.powers.push_back(powers);
      auto group = reduction::detail::direct_seed_group(powers, equations, symmetry);
      if (shift == 0) {
        ++analysis.layer_zero_rows;
        analysis.local_dot_groups.insert(std::move(group));
        continue;
      }
      ++analysis.layer_one_rows;
      const auto& rhs_support = system.selection.residual_rhs_support[residual_index];
      const bool has_only_rank_zero_rhs =
          !rhs_support.empty() && std::ranges::all_of(rhs_support, [&](const auto rhs) {
            return rhs < system.rhs_rank_zero.size() && system.rhs_rank_zero[rhs];
          });
      if (direct_total_rank(group) == 0 && has_only_rank_zero_rhs) {
        ++analysis.rank_zero_layer_one_rows;
        analysis.local_dot_groups.insert(std::move(group));
      }
    }
    return analysis;
  };

  const auto plan_boundary_closure = [&](const BuiltJetSystem& system) {
    std::set<std::size_t> unresolved_targets;
    for (const auto& support : system.selection.residual_rhs_support) {
      for (const std::size_t rhs : support) {
        if (rhs < cfg.targets.size()) unresolved_targets.insert(rhs);
      }
    }
    std::set<std::vector<int>> known_guides;
    for (const auto& target : cfg.targets)
      known_guides.insert(target.indices);
    for (const auto& guide : closure_guides)
      known_guides.insert(guide.indices);
    std::vector<Integral> guides;
    for (const std::size_t target_index : unresolved_targets) {
      const auto& target = cfg.targets[target_index];
      std::uint64_t total_rank = 0;
      for (const int index : target.indices) {
        if (index >= 0) continue;
        if (index == std::numeric_limits<int>::min())
          throw std::overflow_error("direct target rank exceeds int range");
        total_rank += static_cast<std::uint64_t>(-index);
      }
      if (total_rank != 1) continue;
      for (std::size_t slot = 0; slot < target.indices.size(); ++slot) {
        if (target.indices[slot] >= 0) continue;
        auto indices = target.indices;
        if (indices[slot] == std::numeric_limits<int>::min())
          throw std::overflow_error("direct boundary closure exceeds int range");
        --indices[slot];
        if (known_guides.insert(indices).second) guides.push_back({std::move(indices)});
      }
    }
    return guides;
  };

  const auto domain_start = std::chrono::high_resolution_clock::now();
  auto final_seeds =
      equations.get_rank_dot_ansatz_domain(relation_anchors, direct_pinched_dot_halo);
  for (auto& seed : final_seeds)
    seed = symmetry.canonicalize(seed);
  std::ranges::sort(final_seeds);
  final_seeds.erase(std::unique(final_seeds.begin(), final_seeds.end()),
                    final_seeds.end());
  if (progress) {
    progress(std::format("Jet rank-dot domain: pinched_dot_halo={}, "
                         "relation_source_sectors={}, "
                         "closure_guides={}, "
                         "expansion_round=0, seeds={}, "
                         "elapsed_ms={:.2f}",
                         direct_pinched_dot_halo, relation_source_sectors.size(),
                         closure_guides.size(), final_seeds.size(),
                         std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - domain_start)
                             .count()),
             ReductionProgressEvent::info);
  }
  std::map<reduction::detail::DirectSeedGroup, unsigned,
           reduction::detail::DirectSeedGroupLess>
      expanded_groups;
  std::size_t expansion_rounds = 0;
  std::size_t expanded_points = 0;
  bool boundary_closure_attempted = false;
  constexpr auto baseline_group_ordering = DirectGroupOrdering::ExactJet;
  constexpr auto baseline_rank_ordering = DirectRankOrdering::LowFirst;
  constexpr auto baseline_dot_ordering = AnsatzDotOrdering::Markowitz;
  constexpr std::size_t coarse_group_seed_threshold = 1024;
  const auto resolve_group_ordering = [&](std::size_t seed_count) {
    if (direct_group_ordering != DirectGroupOrdering::Auto)
      return direct_group_ordering;
    return seed_count >= coarse_group_seed_threshold ? DirectGroupOrdering::SectorRank
                                                     : DirectGroupOrdering::ExactJet;
  };
  const auto domain_is_rank_zero = [&](const auto& seeds) {
    return std::ranges::all_of(seeds, [&](const auto& seed) {
      return direct_total_rank(
                 reduction::detail::direct_seed_group(seed, equations, symmetry)) == 0;
    });
  };
  const auto resolve_dot_ordering = [&](const auto& seeds, std::size_t rounds) {
    if (ansatz_dot_ordering != AnsatzDotOrdering::Auto) return ansatz_dot_ordering;
    return domain_is_rank_zero(seeds) && rounds != 0 ? AnsatzDotOrdering::HighFirst
                                                     : AnsatzDotOrdering::LowFirst;
  };
  const auto build_requested_system = [&](const auto& seeds, std::size_t rounds) {
    return build_system(seeds, resolve_group_ordering(seeds.size()),
                        direct_rank_ordering, resolve_dot_ordering(seeds, rounds));
  };
  BuiltJetSystem final_system = build_requested_system(final_seeds, expansion_rounds);
  bool final_system_ready = final_system.selection.closed;
  bool use_requested_closure =
      final_system_ready || plan_boundary_closure(final_system).empty();
  bool final_system_uses_requested_ordering = true;
  bool reuse_unclosed_system = !final_system_ready && use_requested_closure;
  if (progress && !use_requested_closure) {
    progress(std::format("Direct production-order closure probe: seeds={}, rows={}, "
                         "relations={}, residual_rows={}, closed={}, elapsed_ms={:.2f}",
                         final_seeds.size(), final_system.row_powers.size(),
                         final_system.metadata.size(),
                         final_system.selection.residual_rows.size(),
                         final_system_ready, final_system.selection.timings.total_ms),
             ReductionProgressEvent::info);
  }
  for (; !final_system_ready;) {
    if (reuse_unclosed_system) {
      reuse_unclosed_system = false;
    } else {
      final_system = use_requested_closure
                         ? build_requested_system(final_seeds, expansion_rounds)
                         : build_system(final_seeds, baseline_group_ordering,
                                        baseline_rank_ordering, baseline_dot_ordering);
      final_system_uses_requested_ordering = use_requested_closure;
    }
    if (progress) {
      progress(
          std::format("Jet rank-dot closure: expansion_round={}, seeds={}, rows={}, "
                      "relations={}, residual_rows={}, closed={}, ordering={}, "
                      "elapsed_ms={:.2f}",
                      expansion_rounds, final_seeds.size(),
                      final_system.row_powers.size(), final_system.metadata.size(),
                      final_system.selection.residual_rows.size(),
                      final_system.selection.closed,
                      use_requested_closure ? "production" : "baseline",
                      final_system.selection.timings.total_ms),
          ReductionProgressEvent::info);
    }
    if (final_system.selection.closed) break;
    auto residual_analysis = analyze_residuals(final_system);
    const auto expansion_start = std::chrono::high_resolution_clock::now();
    const std::vector requested(residual_analysis.local_dot_groups.begin(),
                                residual_analysis.local_dot_groups.end());
    const auto format_groups = [](const auto& groups) {
      std::string text;
      for (const auto& key : groups) {
        if (!text.empty()) text += ',';
        text += std::format("s{}:j[", key.sector);
        for (std::size_t slot = 0; slot < key.orders.size(); ++slot) {
          if (slot != 0) text += ':';
          text += std::to_string(key.orders[slot]);
        }
        text += ']';
      }
      return text;
    };
    if (expansion_rounds >= reduction::detail::maximum_direct_group_expansions) {
      throw AnsatzClosureError("direct predecessor expansion exceeded four rounds");
    }
    auto dot_expansion = reduction::detail::expand_direct_seed_groups(
        final_seeds, requested, expanded_groups, equations, symmetry);
    std::size_t dot_points = dot_expansion.points.size();
    std::size_t dot_groups = dot_expansion.groups.size();
    for (auto& point : dot_expansion.points) {
      final_seeds.push_back(std::move(point));
    }
    std::ranges::sort(final_seeds);
    final_seeds.erase(std::unique(final_seeds.begin(), final_seeds.end()),
                      final_seeds.end());
    expanded_points += dot_points;

    const BuiltJetSystem* residual_system = &final_system;
    BuiltJetSystem dot_trial;
    if (dot_points != 0) {
      const bool use_requested_dot_trial =
          use_requested_closure || final_seeds.size() >= coarse_group_seed_threshold;
      dot_trial = use_requested_dot_trial
                      ? build_requested_system(final_seeds, expansion_rounds + 1)
                      : build_system(final_seeds, baseline_group_ordering,
                                     baseline_rank_ordering, baseline_dot_ordering);
      residual_system = &dot_trial;
      auto dot_analysis = analyze_residuals(dot_trial);
      const bool has_next_local_dot_residual = !dot_analysis.local_dot_groups.empty();
      if (progress) {
        progress(std::format(
                     "Direct local-dot action: round={}, "
                     "rank_zero_layer1_residuals={}, added_groups={}, "
                     "added_points={}, seeds={}, rows={}, "
                     "relations={}, residual_rows={}, closed={}, "
                     "continue_local_dot={}, elapsed_ms={:.2f}",
                     expansion_rounds + 1, residual_analysis.rank_zero_layer_one_rows,
                     dot_expansion.groups.size(), dot_points, final_seeds.size(),
                     dot_trial.row_powers.size(), dot_trial.metadata.size(),
                     dot_trial.selection.residual_rows.size(),
                     dot_trial.selection.closed, has_next_local_dot_residual,
                     dot_trial.selection.timings.total_ms),
                 ReductionProgressEvent::info);
      }
      if (dot_trial.selection.closed) {
        final_system = std::move(dot_trial);
        final_system_uses_requested_ordering = use_requested_dot_trial;
        ++expansion_rounds;
        break;
      }
      if (has_next_local_dot_residual) {
        final_system = std::move(dot_trial);
        final_system_uses_requested_ordering = use_requested_dot_trial;
        reuse_unclosed_system = use_requested_dot_trial;
        ++expansion_rounds;
        continue;
      }
      if (!use_requested_closure && use_requested_dot_trial) {
        dot_trial = build_system(final_seeds, baseline_group_ordering,
                                 baseline_rank_ordering, baseline_dot_ordering);
        residual_system = &dot_trial;
        dot_analysis = analyze_residuals(dot_trial);
      }
      residual_analysis = std::move(dot_analysis);
    }

    if (!boundary_closure_attempted) {
      boundary_closure_attempted = true;
      auto added_guides = plan_boundary_closure(*residual_system);
      if (!added_guides.empty()) {
        use_requested_closure = false;
        closure_guides.insert(closure_guides.end(), added_guides.begin(),
                              added_guides.end());
        closure_guide_columns = build_unit_columns(closure_guides);
        relation_anchors.insert(relation_anchors.end(), added_guides.begin(),
                                added_guides.end());
        const auto action_start = std::chrono::high_resolution_clock::now();
        final_seeds = equations.get_rank_dot_ansatz_domain(relation_anchors,
                                                           direct_pinched_dot_halo);
        for (auto& seed : final_seeds)
          seed = symmetry.canonicalize(seed);
        std::ranges::sort(final_seeds);
        final_seeds.erase(std::unique(final_seeds.begin(), final_seeds.end()),
                          final_seeds.end());
        expanded_groups.clear();
        expanded_points = 0;
        if (progress) {
          progress(std::format(
                       "Direct boundary-closure action: added_guides={}, "
                       "total_guides={}, seeds={}, elapsed_ms={:.2f}",
                       added_guides.size(), closure_guides.size(), final_seeds.size(),
                       std::chrono::duration<double, std::milli>(
                           std::chrono::high_resolution_clock::now() - action_start)
                           .count()),
                   ReductionProgressEvent::info);
        }
        continue;
      }
    }

    auto predecessor = relation_predecessors(residual_analysis.powers, final_seeds);
    const std::size_t matched_relations = predecessor.relations.size();
    std::size_t added_full_sources = 0;
    std::set<reduction::detail::DirectSeedGroup, reduction::detail::DirectSeedGroupLess>
        stalled_source_groups;
    for (const auto& source : predecessor.sources) {
      if (std::ranges::binary_search(final_seeds, source)) {
        stalled_source_groups.insert(
            reduction::detail::direct_seed_group(source, equations, symmetry));
      } else {
        ++added_full_sources;
      }
    }
    const std::size_t states_before = final_seeds.size();
    final_seeds.insert(final_seeds.end(),
                       std::make_move_iterator(predecessor.states.begin()),
                       std::make_move_iterator(predecessor.states.end()));
    std::ranges::sort(final_seeds);
    final_seeds.erase(std::unique(final_seeds.begin(), final_seeds.end()),
                      final_seeds.end());
    const std::size_t added_states = final_seeds.size() - states_before;
    if (!stalled_source_groups.empty()) {
      const std::vector source_groups(stalled_source_groups.begin(),
                                      stalled_source_groups.end());
      auto source_expansion = reduction::detail::expand_direct_seed_groups(
          final_seeds, source_groups, expanded_groups, equations, symmetry);
      dot_groups += source_expansion.groups.size();
      dot_points += source_expansion.points.size();
      expanded_points += source_expansion.points.size();
      for (auto& point : source_expansion.points)
        final_seeds.push_back(std::move(point));
    }
    std::ranges::sort(final_seeds);
    final_seeds.erase(std::unique(final_seeds.begin(), final_seeds.end()),
                      final_seeds.end());
    if (final_seeds.size() > 1'000'000)
      throw AnsatzClosureError(
          "direct predecessor expansion exceeds one million states");
    if (progress) {
      progress(
          std::format("Direct predecessor action: round={}, layer0_residuals={}, "
                      "layer1_residuals={}, residual_groups={}, "
                      "dot_groups={}, dot_points={}, raw_sources={}, "
                      "matched_relations={}, "
                      "same_layer_relations={}, previous_layer_relations={}, "
                      "added_full_sources={}, added_states={}, required_states={}, "
                      "maximum_g_shift={}, "
                      "residual_keys=[{}], elapsed_ms={:.2f}",
                      expansion_rounds + 1, residual_analysis.layer_zero_rows,
                      residual_analysis.layer_one_rows,
                      residual_analysis.local_dot_groups.size(), dot_groups, dot_points,
                      predecessor.raw_sources, matched_relations,
                      predecessor.same_layer_relations,
                      predecessor.previous_layer_relations, added_full_sources,
                      added_states, predecessor.required_states,
                      predecessor.maximum_g_shift,
                      format_groups(residual_analysis.local_dot_groups),
                      std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - expansion_start)
                          .count()),
          ReductionProgressEvent::info);
    }
    if (dot_points == 0 && added_full_sources == 0 && added_states == 0) {
      throw AnsatzClosureError("direct predecessor expansion made no progress");
    }
    ++expansion_rounds;
  }
  const auto final_group_ordering = resolve_group_ordering(final_seeds.size());
  const auto final_dot_ordering = resolve_dot_ordering(final_seeds, expansion_rounds);
  const bool replan_final_system = !final_system_uses_requested_ordering &&
                                   (final_group_ordering != baseline_group_ordering ||
                                    direct_rank_ordering != baseline_rank_ordering ||
                                    final_dot_ordering != baseline_dot_ordering);
  if (replan_final_system) {
    final_system = build_system(final_seeds, final_group_ordering, direct_rank_ordering,
                                final_dot_ordering);
    if (!final_system.selection.closed)
      throw AnsatzClosureError("direct final production ordering is not closed");
    if (progress) {
      const auto group_name = [&] {
        switch (final_group_ordering) {
        case DirectGroupOrdering::Auto:
          return "auto";
        case DirectGroupOrdering::ExactJet:
          return "exact-jet";
        case DirectGroupOrdering::RankSector:
          return "rank-sector";
        case DirectGroupOrdering::SectorRank:
          return "sector-rank";
        }
        return "unknown";
      }();
      const auto rank_name =
          direct_rank_ordering == DirectRankOrdering::LowFirst ? "low" : "high";
      const auto dot_name = [&] {
        switch (final_dot_ordering) {
        case AnsatzDotOrdering::Markowitz:
          return "markowitz";
        case AnsatzDotOrdering::LowFirst:
          return "low";
        case AnsatzDotOrdering::HighFirst:
          return "high";
        case AnsatzDotOrdering::Auto:
          break;
        }
        return "unknown";
      }();
      progress(std::format("Direct frozen-system replan: group_order={}, "
                           "rank_order={}, dot_order={}, rows={}, relations={}, "
                           "compact_dimension={}, elapsed_ms={:.2f}",
                           group_name, rank_name, dot_name,
                           final_system.row_powers.size(), final_system.metadata.size(),
                           final_system.selection.solution_columns.size(),
                           final_system.selection.timings.total_ms),
               ReductionProgressEvent::info);
    }
  }
  auto& indexed_basis = final_system.basis;
  auto& indexed_targets = final_system.targets;
  auto& indexed_ansatz = final_system.ansatz;
  auto& metadata = final_system.metadata;
  auto& row_sectors = final_system.row_sectors;
  auto& row_groups = final_system.row_groups;
  auto& ordered_groups = final_system.ordered_groups;
  auto& target_sectors = final_system.target_sectors;
  auto& selection = final_system.selection;
  const std::size_t input_basis_count = indexed_basis.size();

  const auto final_group_name = [&] {
    switch (final_group_ordering) {
    case DirectGroupOrdering::Auto:
      return "auto";
    case DirectGroupOrdering::ExactJet:
      return "exact-jet";
    case DirectGroupOrdering::RankSector:
      return "rank-sector";
    case DirectGroupOrdering::SectorRank:
      return "sector-rank";
    }
    return "unknown";
  }();
  const auto final_rank_name =
      direct_rank_ordering == DirectRankOrdering::LowFirst ? "low-rank" : "high-rank";
  const auto final_dot_name = [&] {
    switch (final_dot_ordering) {
    case AnsatzDotOrdering::Markowitz:
      return "markowitz";
    case AnsatzDotOrdering::LowFirst:
      return "low-dot";
    case AnsatzDotOrdering::HighFirst:
      return "high-dot";
    case AnsatzDotOrdering::Auto:
      break;
    }
    return "unknown";
  }();
  kernel_statistics_.compact_policy =
      std::format("jet-{}-{}-{}", final_group_name, final_rank_name, final_dot_name);
  publish_ansatz_statistics(selection, metadata, final_system.row_powers.size(),
                            input_basis_count + metadata.size(), expansion_rounds,
                            expanded_groups.size(), expanded_points);
  kernel_statistics_.jet_state_count = final_seeds.size();
  kernel_statistics_.jet_row_count = final_system.row_powers.size();
  kernel_statistics_.jet_relation_count = metadata.size();
  kernel_statistics_.jet_group_count = final_system.group_count;
  for (const auto& row : final_system.row_powers) {
    if (row.front() < 0)
      kernel_statistics_.maximum_g_shift = std::max(
          kernel_statistics_.maximum_g_shift, static_cast<std::size_t>(-row.front()));
  }
  if (progress) {
    progress(
        std::format("Jet compact selection: expansion_rounds={}, "
                    "expanded_groups={}, expanded_points={}, states={}, groups={}, "
                    "rows={}, relations={}, provisional_dimension={}, "
                    "live_relations={}, compact_dimension={}, elapsed_ms={:.2f}",
                    expansion_rounds, expanded_groups.size(), expanded_points,
                    final_seeds.size(), final_system.group_count,
                    final_system.row_powers.size(), metadata.size(),
                    selection.provisional_dimension, selection.ansatz_order.size(),
                    selection.solution_columns.size(), selection.timings.total_ms),
        ReductionProgressEvent::info);
  }

  publish_kernel_plan(
      {std::move(indexed_basis), std::move(indexed_targets), std::move(indexed_ansatz),
       std::move(metadata), std::move(row_sectors), std::move(row_groups),
       std::move(ordered_groups), std::move(target_sectors),
       std::move(selection.ansatz_order), std::move(selection.solution_columns),
       std::move(selection.elimination_row_map), std::move(polynomial_values)},
      coeffs, values);
}
