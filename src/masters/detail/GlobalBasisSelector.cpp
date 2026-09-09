#include "masters/detail/GlobalBasisSelector.hpp"

#include "core/ProbeValues.hpp"
#include "reduction/EliminationTape.hpp"
#include "reduction/EquationGenerator.hpp"
#include "reduction/FiniteFieldArithmetic.hpp"
#include "reduction/KernelPlanning.hpp"
#include "reduction/ParameterEvaluation.hpp"
#include "reduction/ProjectedSeedExpansion.hpp"
#include "reduction/SymbolicSignature.hpp"
#include "reduction/SymmetryCanonicalizer.hpp"
#include "topology/IntegralLayout.hpp"
#include "topology/SectorUtils.hpp"

#include <firefly/FFInt.hpp>
#include <firefly/ReconstHelper.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using T = firefly::FFInt;

struct SymbolicSystem {
  struct Relation {
    std::vector<Monomial> terms;
    std::uint32_t seed_sector = 0;
    std::uint32_t seed_dot_excess = 0;
    unsigned derivative_index = 0;
    AnsatzFamily family = AnsatzFamily::Nabla;
  };

  std::vector<Relation> relations;
  std::vector<std::vector<Monomial>> candidates;
};

Config selection_config(const MasterFinderConfig& source,
                        std::span<const Integral> ordered_candidates)
{
  Config result;
  static_cast<MasterFinderConfig&>(result) = source;
  result.basis.assign(ordered_candidates.begin(), ordered_candidates.end());
  return result;
}

std::vector<T> parameter_values(std::size_t count, std::uint64_t prime,
                                std::size_t point)
{
  std::vector<T> result;
  result.reserve(count);
  for (std::size_t parameter = 0; parameter < count; ++parameter) {
    result.emplace_back(probe_values::planning_field_value(prime, parameter, point));
  }
  return result;
}

std::vector<T> polynomial_values(const Config& config, std::span<const T> values)
{
  std::vector<T> result(config.polynomial_terms.size());
  for (std::size_t term = 0; term < config.polynomial_terms.size(); ++term) {
    result[term] = reduction::detail::evaluate_polynomial_coefficient(
        config, config.polynomial_terms[term], values);
  }
  return result;
}

template <typename Term>
T evaluate(const Term& term, std::span<const T> polynomial_coefficients,
           const T& minus_half_d)
{
  if (term.coefficient_expression != std::numeric_limits<std::uint32_t>::max()) {
    throw std::logic_error(
        "global master selection does not accept top-LP coefficient expressions");
  }
  T value = T(term.coeff_int) + T(term.coeff_minus_half_d) * minus_half_d;
  if (term.with_polynomial_coefficient) {
    if (term.polynomial_term_index >= polynomial_coefficients.size()) {
      throw std::logic_error("global master-selection polynomial term is invalid");
    }
    value = value * polynomial_coefficients[term.polynomial_term_index];
  }
  return value;
}

struct ProbeSelection {
  std::vector<std::size_t> selected_candidates;
};

struct IndexedSelectionSystem {
  reduction::detail::IndexedColumns columns;
  std::unordered_map<std::vector<int>, std::uint32_t, VectorHash> row_ids;
  std::vector<std::vector<int>> row_powers;
  std::size_t relation_count = 0;
};

IndexedSelectionSystem
index_selection_system(const SymbolicSystem& symbolic,
                       const SymmetryCanonicalizer& canonicalizer)
{
  IndexedSelectionSystem result;
  result.relation_count = symbolic.relations.size();
  const auto intern = [&](std::span<const int> powers) {
    auto canonical = canonicalizer.canonicalize(powers);
    const auto [entry, inserted] = result.row_ids.try_emplace(std::move(canonical));
    if (inserted) {
      if (result.row_ids.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("global master-selection rows exceed 32-bit ids");
      entry->second = static_cast<std::uint32_t>(result.row_ids.size() - 1);
      result.row_powers.push_back(entry->first);
    }
    return entry->second;
  };
  const auto append = [&](const std::vector<Monomial>& column) {
    for (const auto& term : column) {
      result.columns.terms.push_back(
          {intern(term.powers), term.polynomial_term_index, term.coefficient_expression,
           term.with_polynomial_coefficient, term.coeff_int, term.coeff_minus_half_d});
    }
    result.columns.offsets.push_back(result.columns.terms.size());
  };
  for (const auto& relation : symbolic.relations)
    append(relation.terms);
  for (const auto& candidate : symbolic.candidates)
    append(candidate);
  return result;
}

ProbeSelection select_at_probe(const IndexedSelectionSystem& symbolic,
                               const Config& config, std::uint64_t prime,
                               std::size_t point)
{
  using Field = finite_field::MontgomeryFieldElement;
  Field::set_prime(prime);
  const auto values = parameter_values(config.parameters.size(), prime, point);
  const auto coefficients = polynomial_values(config, values);
  const T minus_half_d =
      (T(0) - reduction::detail::evaluate_dimension(config, values)) / T(2);

  const std::size_t column_count = symbolic.columns.size();
  linalg::SparseMatrix<Field> matrix(symbolic.row_powers.size());
  // Only one evaluated column is needed while assembling the row matrix.
  // Release it before elimination rather than retaining a second numeric system.
  {
    std::vector<std::pair<std::uint32_t, Field>> entries;
    for (std::size_t column = 0; column < column_count; ++column) {
      entries.clear();
      const auto terms = symbolic.columns.column(column);
      entries.reserve(terms.size());
      for (const auto& term : terms) {
        const T value = evaluate(term, coefficients, minus_half_d);
        if (value != T(0)) entries.emplace_back(term.row, Field::from_residue(value.n));
      }
      std::ranges::sort(entries, {}, &std::pair<std::uint32_t, Field>::first);
      std::size_t write = 0;
      for (const auto& entry : entries) {
        if (write != 0 && entries[write - 1].first == entry.first) {
          entries[write - 1].second = entries[write - 1].second + entry.second;
          if (entries[write - 1].second == Field(0)) --write;
        } else {
          entries[write++] = entry;
        }
      }
      entries.resize(write);
      for (const auto& [row, value] : entries)
        matrix[row].push_back({column, value});
    }
  }
  std::vector<Field> no_rhs;
  ProbeSelection result;
  auto elimination = linalg::sparse_gaussian_elimination<Field, true>(
      matrix, column_count, no_rhs, 0, column_count);
  for (const std::size_t column : elimination.solution_cols) {
    if (column >= symbolic.relation_count) {
      result.selected_candidates.push_back(column - symbolic.relation_count);
    }
  }
  return result;
}

bool column_stays_in_seed_sector(const std::vector<Monomial>& terms,
                                 std::uint32_t seed_sector, const SectorUtils& sectors,
                                 const SymmetryCanonicalizer& canonicalizer)
{
  const auto seed_popcount = std::popcount(seed_sector);
  return std::ranges::all_of(terms, [&](const Monomial& term) {
    const auto canonical = canonicalizer.canonicalize(term.powers);
    const auto row_sector = sectors.sector_from_powers(canonical);
    const auto row_popcount = std::popcount(row_sector);
    return row_popcount < seed_popcount || row_sector == seed_sector;
  });
}

std::vector<SymbolicSystem::Relation>
deduplicate_relations(std::vector<SymbolicSystem::Relation> relations,
                      const Config& config, const SymmetryCanonicalizer& canonicalizer)
{
  std::unordered_map<std::vector<int>, std::int64_t, VectorHash> row_ids;
  const auto row_id = [&](std::span<const int> powers) {
    auto canonical = canonicalizer.canonicalize(powers);
    const auto [entry, inserted] = row_ids.try_emplace(std::move(canonical));
    if (inserted) {
      if (row_ids.size() >
          static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error(
            "global master-selection signature rows exceed int64 ids");
      }
      entry->second = static_cast<std::int64_t>(row_ids.size() - 1);
    }
    return entry->second;
  };
  const auto signature_hash = [](const SymbolicSignature& signature) {
    std::size_t seed = signature.size();
    for (const auto& entry : signature) {
      for (const auto value : entry) {
        seed ^=
            std::hash<std::int64_t>{}(value) + 0x9e3779b9 + (seed << 6U) + (seed >> 2U);
      }
    }
    return seed;
  };

  std::unordered_map<std::size_t, std::vector<SymbolicSignature>> signatures;
  std::vector<SymbolicSystem::Relation> result;
  result.reserve(relations.size());
  const std::size_t parameter_count =
      reduction::detail::coefficient_parameter_count(config);
  for (auto& relation : relations) {
    SymbolicSignature signature;
    for (const auto& term : relation.terms) {
      if (term.coefficient_expression != std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error(
            "global master-selection relation contains a top-LP coefficient");
      }
      const auto append = [&](std::size_t basis_index, std::int64_t weight) {
        if (weight != 0) {
          signature.push_back(
              {row_id(term.powers), static_cast<std::int64_t>(basis_index), weight});
        }
      };
      if (term.with_polynomial_coefficient) {
        const auto& weights =
            config.polynomial_terms.at(term.polynomial_term_index).weights;
        for (std::size_t parameter = 0; parameter < parameter_count; ++parameter) {
          append(parameter, reduction::detail::checked_mul_i64(term.coeff_int,
                                                               weights[parameter]));
          append(parameter_count + parameter,
                 reduction::detail::checked_mul_i64(term.coeff_minus_half_d,
                                                    weights[parameter]));
        }
      } else {
        append(0, term.coeff_int);
        append(parameter_count, term.coeff_minus_half_d);
      }
    }
    std::ranges::sort(signature);
    SymbolicSignature merged;
    merged.reserve(signature.size());
    for (const auto& entry : signature) {
      if (!merged.empty() && merged.back()[0] == entry[0] &&
          merged.back()[1] == entry[1]) {
        merged.back()[2] =
            reduction::detail::checked_add_i64(merged.back()[2], entry[2]);
      } else {
        merged.push_back(entry);
      }
    }
    std::erase_if(merged, [](const auto& entry) { return entry[2] == 0; });
    make_primitive_symbolic_signature(merged);
    if (merged.empty()) continue;
    const auto hash = signature_hash(merged);
    auto& bucket = signatures[hash];
    if (std::ranges::find(bucket, merged) != bucket.end()) continue;
    bucket.push_back(std::move(merged));
    result.push_back(std::move(relation));
  }
  return result;
}

std::uint32_t seed_dot_excess(std::span<const int> powers)
{
  const auto exponents = powers.empty() ? powers : powers.subspan(1);
  return reduction::detail::positive_power_sum_u32(
      exponents, "master-selection seed dot excess exceeds uint32");
}

std::vector<Integral> selected_basis(std::span<const Integral> candidates,
                                     std::span<const std::size_t> selected_candidates)
{
  std::vector<Integral> result;
  result.reserve(selected_candidates.size());
  for (const std::size_t selected : selected_candidates)
    result.push_back(candidates[selected]);
  return result;
}

std::vector<std::vector<Monomial>>
selected_basis_border(const Config& config, std::span<const Integral> basis)
{
  std::set<std::vector<int>> powers;
  for (const auto& integral : basis) {
    for (const std::uint32_t slot : config.propagator_slots) {
      if (integral.indices.at(slot) <= 0) continue;
      auto dotted = integral.indices;
      if (dotted[slot] == std::numeric_limits<int>::max()) {
        throw std::overflow_error("master-basis border index exceeds int");
      }
      ++dotted[slot];
      std::vector<int> row;
      row.reserve(config.propagator_count + 1);
      row.push_back(0);
      for (const std::uint32_t active_slot : config.propagator_slots)
        row.push_back(dotted[active_slot] - 1);
      powers.insert(std::move(row));
    }
  }

  std::vector<std::vector<Monomial>> result;
  result.reserve(powers.size());
  while (!powers.empty()) {
    auto node = powers.extract(powers.begin());
    result.push_back({{.powers = std::move(node.value()),
                       .with_polynomial_coefficient = false,
                       .polynomial_term_index = 0,
                       .coeff_int = 1,
                       .coeff_minus_half_d = 0}});
  }
  return result;
}

struct ClosureProbe {
  bool closed = false;
  std::set<reduction::detail::ProjectedSeedGroup,
           reduction::detail::ProjectedSeedGroupLess>
      residual_groups;
  std::vector<std::size_t> ansatz_order;
  std::vector<std::size_t> solution_columns;
  std::vector<std::size_t> elimination_row_map;
};

struct ClosureSystem {
  reduction::detail::IndexedColumns basis_columns;
  reduction::detail::IndexedColumns exact_target_columns;
  reduction::detail::IndexedColumns projected_target_columns;
  reduction::detail::IndexedColumns ansatz_columns;
  std::vector<reduction::detail::AnsatzColumnMeta> metadata;
  std::vector<std::uint32_t> row_sectors;
};

ClosureSystem build_closure_system(const SymbolicSystem& symbolic,
                                   IndexedSelectionSystem& indexed,
                                   const SectorUtils& sectors,
                                   const SymmetryCanonicalizer& canonicalizer,
                                   std::span<const std::size_t> selected_candidates,
                                   const std::vector<std::vector<Monomial>>& border)
{
  ClosureSystem result;
  result.row_sectors.reserve(indexed.row_powers.size());
  for (const auto& powers : indexed.row_powers)
    result.row_sectors.push_back(sectors.sector_from_powers(powers));
  const auto intern = [&](std::span<const int> powers) {
    auto canonical = canonicalizer.canonicalize(powers);
    const auto [entry, inserted] = indexed.row_ids.try_emplace(std::move(canonical));
    if (inserted) {
      if (indexed.row_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("master closure rows exceed 32-bit ids");
      }
      entry->second = static_cast<std::uint32_t>(indexed.row_ids.size() - 1);
      indexed.row_powers.push_back(entry->first);
      result.row_sectors.push_back(sectors.sector_from_powers(entry->first));
    }
    return entry->second;
  };
  const auto append_indexed =
      [&](reduction::detail::IndexedColumns& output,
          std::span<const reduction::detail::IndexedTerm> column) {
        output.terms.insert(output.terms.end(), column.begin(), column.end());
        output.offsets.push_back(output.terms.size());
      };
  const auto append_border = [&](const std::vector<Monomial>& column) {
    for (const auto& term : column) {
      result.exact_target_columns.terms.push_back(
          {intern(term.powers), term.polynomial_term_index, term.coefficient_expression,
           term.with_polynomial_coefficient, term.coeff_int, term.coeff_minus_half_d});
    }
    result.exact_target_columns.offsets.push_back(
        result.exact_target_columns.terms.size());
  };

  for (const std::size_t selected : selected_candidates) {
    append_indexed(result.basis_columns,
                   indexed.columns.column(indexed.relation_count + selected));
  }
  for (std::size_t candidate = 0; candidate < symbolic.candidates.size(); ++candidate) {
    append_indexed(result.exact_target_columns,
                   indexed.columns.column(indexed.relation_count + candidate));
  }
  for (const auto& column : border)
    append_border(column);

  const std::size_t target_count = result.exact_target_columns.size();
  for (std::size_t projection = 0; projection < 2; ++projection) {
    for (std::size_t target = 0; target < target_count; ++target) {
      const auto column = result.exact_target_columns.column(target);
      const auto index = static_cast<std::int64_t>(target + 1);
      const auto weight = projection == 0 ? index : index * (index + 1);
      for (const auto& term : column) {
        auto scaled = term;
        scaled.coeff_int = reduction::detail::checked_mul_i64(scaled.coeff_int, weight);
        scaled.coeff_minus_half_d =
            reduction::detail::checked_mul_i64(scaled.coeff_minus_half_d, weight);
        result.projected_target_columns.terms.push_back(scaled);
      }
    }
    result.projected_target_columns.offsets.push_back(
        result.projected_target_columns.terms.size());
  }
  result.metadata.reserve(symbolic.relations.size());
  for (std::size_t relation = 0; relation < symbolic.relations.size(); ++relation) {
    const auto& source = symbolic.relations[relation];
    append_indexed(result.ansatz_columns, indexed.columns.column(relation));
    result.metadata.push_back({static_cast<std::uint32_t>(relation), source.seed_sector,
                               static_cast<std::uint16_t>(source.derivative_index),
                               source.family});
    result.metadata.back().seed_dot_excess = source.seed_dot_excess;
  }
  return result;
}

ClosureSystem restrict_closure_system(const ClosureSystem& source,
                                      std::span<const std::size_t> ansatz_order)
{
  ClosureSystem result;
  result.basis_columns = source.basis_columns;
  result.exact_target_columns = source.exact_target_columns;
  result.projected_target_columns = source.projected_target_columns;
  result.row_sectors = source.row_sectors;
  result.metadata.reserve(ansatz_order.size());
  for (const std::size_t column : ansatz_order) {
    if (column >= source.ansatz_columns.size())
      throw std::logic_error("master closure ansatz support is out of range");
    const auto terms = source.ansatz_columns.column(column);
    result.ansatz_columns.terms.insert(result.ansatz_columns.terms.end(), terms.begin(),
                                       terms.end());
    result.ansatz_columns.offsets.push_back(result.ansatz_columns.terms.size());
    result.metadata.push_back(source.metadata[column]);
  }
  return result;
}

ClosureProbe close_at_probe(const ClosureSystem& system, const Config& config,
                            std::uint64_t prime, std::size_t point,
                            std::size_t expansion_rounds, bool exact_targets)
{
  firefly::FFInt::set_new_prime(prime);
  const auto values = parameter_values(config.parameters.size(), prime, point);
  const auto coefficients = polynomial_values(config, values);
  const T minus_half_d =
      (T(0) - reduction::detail::evaluate_dimension(config, values)) / T(2);

  const auto ordering =
      resolve_ansatz_dot_ordering(AnsatzDotOrdering::Auto, false, expansion_rounds);
  const auto& target_columns =
      exact_targets ? system.exact_target_columns : system.projected_target_columns;
  auto selection = reduction::detail::plan_compact_kernel(
      system.basis_columns, target_columns, system.ansatz_columns, system.metadata,
      system.row_sectors, coefficients, {}, minus_half_d, system.row_sectors, {},
      ordering, config.threads);
  ClosureProbe result{selection.closed,
                      {},
                      std::move(selection.ansatz_order),
                      std::move(selection.solution_columns),
                      std::move(selection.elimination_row_map)};
  for (const std::size_t row : selection.residual_rows) {
    if (row >= system.row_sectors.size())
      throw std::logic_error("master closure residual row is out of range");
    result.residual_groups.insert({0, system.row_sectors[row]});
  }
  return result;
}

ClosureProbe verify_exact_closure(const ClosureSystem& system, const Config& config,
                                  std::uint64_t prime, std::size_t point)
{
  firefly::FFInt::set_new_prime(prime);
  const auto values = parameter_values(config.parameters.size(), prime, point);
  const auto coefficients = polynomial_values(config, values);
  const T minus_half_d =
      (T(0) - reduction::detail::evaluate_dimension(config, values)) / T(2);
  const std::size_t source_columns =
      system.basis_columns.size() + system.ansatz_columns.size();
  const std::size_t total_columns = source_columns + system.exact_target_columns.size();
  linalg::SparseMatrix<T> matrix(system.row_sectors.size());
  const auto append_column = [&](const reduction::detail::IndexedColumns& columns,
                                 std::size_t source, std::size_t destination) {
    std::map<std::uint32_t, T> entries;
    for (const auto& term : columns.column(source))
      entries[term.row] =
          entries[term.row] + evaluate(term, coefficients, minus_half_d);
    for (const auto& [row, value] : entries) {
      if (value != T(0)) matrix[row].push_back({destination, value});
    }
  };
  for (std::size_t column = 0; column < system.basis_columns.size(); ++column)
    append_column(system.basis_columns, column, column);
  for (std::size_t column = 0; column < system.ansatz_columns.size(); ++column) {
    append_column(system.ansatz_columns, column, system.basis_columns.size() + column);
  }
  for (std::size_t column = 0; column < system.exact_target_columns.size(); ++column)
    append_column(system.exact_target_columns, column, source_columns + column);
  for (auto& row : matrix)
    std::ranges::sort(row, {}, &linalg::SparseEntry<T>::column);

  std::vector<T> no_rhs;
  const auto elimination = linalg::sparse_gaussian_elimination(
      matrix, total_columns, no_rhs, 0, total_columns);
  ClosureProbe result;
  result.closed = true;
  for (const std::size_t column : elimination.solution_cols) {
    if (column < source_columns) continue;
    result.closed = false;
    const std::size_t target = column - source_columns;
    for (const auto& term : system.exact_target_columns.column(target))
      result.residual_groups.insert({0, system.row_sectors[term.row]});
  }
  return result;
}

struct ClosureReplayPlan {
  struct MatrixLoad {
    std::uint32_t slot = 0;
    reduction::detail::IndexedTerm term;
  };

  ClosureSystem system;
  linalg::TapeResult recorded;
  std::vector<MatrixLoad> matrix_loads;
  std::vector<std::size_t> pivot_rows;
};

bool execute_closure_tape(std::vector<T>& matrix, std::vector<T>& rhs,
                          std::span<const linalg::Instruction> tape)
{
  T factor(0);
  for (const auto instruction : tape) {
    switch (instruction.opcode()) {
    case linalg::OpCode::Inv:
      if (matrix[instruction.source()] == T(0)) return false;
      factor = T(1) / matrix[instruction.source()];
      break;
    case linalg::OpCode::MulM:
      matrix[instruction.offset()] = matrix[instruction.offset()] * factor;
      break;
    case linalg::OpCode::MulB:
      rhs[instruction.offset()] = rhs[instruction.offset()] * factor;
      break;
    case linalg::OpCode::LoadF:
      factor = matrix[instruction.source()];
      break;
    case linalg::OpCode::FmaM:
      matrix[instruction.offset()] =
          matrix[instruction.offset()] - factor * matrix[instruction.source()];
      break;
    case linalg::OpCode::FmaB:
      rhs[instruction.offset()] =
          rhs[instruction.offset()] - factor * rhs[instruction.source()];
      break;
    }
  }
  return true;
}

ClosureReplayPlan build_closure_replay_plan(ClosureSystem system,
                                            const ClosureProbe& anchor,
                                            const Config& config, std::uint64_t prime,
                                            std::size_t point)
{
  constexpr std::size_t rhs_columns = 2;
  const std::size_t dimension =
      system.basis_columns.size() + system.ansatz_columns.size();
  if (anchor.solution_columns.size() != dimension ||
      !std::ranges::equal(anchor.solution_columns,
                          std::views::iota(std::size_t{0}, dimension)) ||
      anchor.elimination_row_map.size() < dimension) {
    throw std::logic_error("master closure support is not a square full-rank system");
  }

  ClosureReplayPlan result;
  result.system = std::move(system);
  result.pivot_rows.assign(anchor.elimination_row_map.begin(),
                           anchor.elimination_row_map.begin() +
                               static_cast<std::ptrdiff_t>(dimension));
  std::vector<std::size_t> local_row(result.system.row_sectors.size(),
                                     std::numeric_limits<std::size_t>::max());
  for (std::size_t row = 0; row < dimension; ++row)
    local_row[result.pivot_rows[row]] = row;

  std::map<std::pair<std::size_t, std::size_t>,
           std::vector<reduction::detail::IndexedTerm>>
      coordinates;
  const auto collect_column = [&](const reduction::detail::IndexedColumns& columns,
                                  std::size_t source, std::size_t destination) {
    for (const auto& term : columns.column(source)) {
      const std::size_t row = local_row.at(term.row);
      if (row != std::numeric_limits<std::size_t>::max())
        coordinates[{row, destination}].push_back(term);
    }
  };
  for (std::size_t column = 0; column < result.system.basis_columns.size(); ++column)
    collect_column(result.system.basis_columns, column, column);
  for (std::size_t column = 0; column < result.system.ansatz_columns.size(); ++column) {
    collect_column(result.system.ansatz_columns, column,
                   result.system.basis_columns.size() + column);
  }

  firefly::FFInt::set_new_prime(prime);
  const auto values = parameter_values(config.parameters.size(), prime, point);
  const auto coefficients = polynomial_values(config, values);
  const T minus_half_d =
      (T(0) - reduction::detail::evaluate_dimension(config, values)) / T(2);
  linalg::SparseMatrix<T> matrix(dimension);
  std::uint32_t slot = 0;
  for (const auto& [coordinate, terms] : coordinates) {
    T value(0);
    for (const auto& term : terms)
      value = value + evaluate(term, coefficients, minus_half_d);
    if (value == T(0)) continue;
    matrix[coordinate.first].push_back({coordinate.second, slot, value});
    for (const auto& term : terms)
      result.matrix_loads.push_back({slot, term});
    ++slot;
  }

  std::vector<T> rhs(dimension * rhs_columns, T(0));
  std::vector<std::uint8_t> rhs_support(rhs.size(), 0);
  for (std::size_t projection = 0; projection < rhs_columns; ++projection) {
    for (const auto& term : result.system.projected_target_columns.column(projection)) {
      const std::size_t row = local_row.at(term.row);
      if (row == std::numeric_limits<std::size_t>::max()) continue;
      const std::size_t coordinate = row * rhs_columns + projection;
      rhs[coordinate] = rhs[coordinate] + evaluate(term, coefficients, minus_half_d);
      rhs_support[coordinate] = 1;
    }
  }
  result.recorded = linalg::record_sparse_tape(matrix, rhs, rhs_columns, rhs_support);
  return result;
}

ClosureProbe replay_closure(const ClosureReplayPlan& plan, const Config& config,
                            std::uint64_t prime, std::size_t point)
{
  constexpr std::size_t rhs_columns = 2;
  firefly::FFInt::set_new_prime(prime);
  const auto values = parameter_values(config.parameters.size(), prime, point);
  const auto coefficients = polynomial_values(config, values);
  const T minus_half_d =
      (T(0) - reduction::detail::evaluate_dimension(config, values)) / T(2);

  std::vector<T> matrix(plan.recorded.matrix_slot_count, T(0));
  for (const auto& load : plan.matrix_loads) {
    matrix[load.slot] =
        matrix[load.slot] + evaluate(load.term, coefficients, minus_half_d);
  }
  std::vector<T> rhs(plan.recorded.rhs_slot_count, T(0));
  std::vector<std::size_t> local_row(plan.system.row_sectors.size(),
                                     std::numeric_limits<std::size_t>::max());
  for (std::size_t row = 0; row < plan.pivot_rows.size(); ++row)
    local_row[plan.pivot_rows[row]] = row;
  for (std::size_t projection = 0; projection < rhs_columns; ++projection) {
    for (const auto& term : plan.system.projected_target_columns.column(projection)) {
      const std::size_t row = local_row.at(term.row);
      if (row != std::numeric_limits<std::size_t>::max()) {
        rhs[row * rhs_columns + projection] =
            rhs[row * rhs_columns + projection] +
            evaluate(term, coefficients, minus_half_d);
      }
    }
  }
  if (!execute_closure_tape(matrix, rhs, plan.recorded.tape))
    throw std::runtime_error("master closure replay encountered a zero pivot");

  const std::size_t dimension = plan.pivot_rows.size();
  std::vector<T> solution(dimension * rhs_columns, T(0));
  for (std::size_t column = 0; column < dimension; ++column) {
    const std::size_t source = plan.recorded.perm[column] * rhs_columns;
    for (std::size_t projection = 0; projection < rhs_columns; ++projection)
      solution[column * rhs_columns + projection] = rhs[source + projection];
  }

  std::vector<T> residual(plan.system.row_sectors.size() * rhs_columns, T(0));
  for (std::size_t projection = 0; projection < rhs_columns; ++projection) {
    for (const auto& term : plan.system.projected_target_columns.column(projection)) {
      residual[term.row * rhs_columns + projection] =
          residual[term.row * rhs_columns + projection] -
          evaluate(term, coefficients, minus_half_d);
    }
  }
  const auto accumulate_column = [&](const reduction::detail::IndexedColumns& columns,
                                     std::size_t source, std::size_t solution_column) {
    for (const auto& term : columns.column(source)) {
      const T value = evaluate(term, coefficients, minus_half_d);
      for (std::size_t projection = 0; projection < rhs_columns; ++projection) {
        residual[term.row * rhs_columns + projection] =
            residual[term.row * rhs_columns + projection] +
            value * solution[solution_column * rhs_columns + projection];
      }
    }
  };
  for (std::size_t column = 0; column < plan.system.basis_columns.size(); ++column)
    accumulate_column(plan.system.basis_columns, column, column);
  for (std::size_t column = 0; column < plan.system.ansatz_columns.size(); ++column) {
    accumulate_column(plan.system.ansatz_columns, column,
                      plan.system.basis_columns.size() + column);
  }

  ClosureProbe result;
  result.closed = true;
  for (std::size_t row = 0; row < plan.system.row_sectors.size(); ++row) {
    if (residual[row * rhs_columns] == T(0) &&
        residual[row * rhs_columns + 1] == T(0)) {
      continue;
    }
    result.closed = false;
    result.residual_groups.insert({0, plan.system.row_sectors[row]});
  }
  return result;
}

} // namespace

namespace masters::detail {

std::vector<Integral>
select_global_master_basis(const MasterFinderConfig& source,
                           std::span<const Integral> candidates,
                           std::span<const std::uint32_t> relation_source_sectors)
{
  if (candidates.empty()) return {};
  Config config = selection_config(source, candidates);
  const auto primes = reduction::detail::usable_firefly_primes(config, 2);
  const std::size_t candidate_count = candidates.size();
  SectorUtils sectors(config);
  EquationGenerator equations(config, sectors);
  SymmetryCanonicalizer canonicalizer(config, sectors, config.propagator_slots);
  auto candidate_columns = equations.build_basis_columns();
  if (candidate_columns.size() != candidate_count) {
    throw std::logic_error("global master candidate columns are incomplete");
  }

  std::vector<Integral> relation_anchors;
  relation_anchors.reserve(relation_source_sectors.size());
  for (const std::uint32_t sector : relation_source_sectors)
    relation_anchors.push_back(integral_layout::sector_corner(source, sector));

  SymbolicSystem symbolic;
  symbolic.candidates = std::move(candidate_columns);
  if (std::ranges::any_of(symbolic.candidates,
                          [](const auto& column) { return column.empty(); })) {
    throw std::runtime_error("global master candidates contain an invalid integral");
  }

  auto envelope = equations.build_initial_ansatz_domain(relation_anchors);
  std::map<reduction::detail::ProjectedSeedGroup, unsigned,
           reduction::detail::ProjectedSeedGroupLess>
      expansion_counts;
  std::size_t expansion_rounds = 0;
  ProbeSelection accepted;
  for (;;) {
    canonicalizer.canonicalize_grid(envelope);
    std::vector<SymbolicSystem::Relation> relations;
    equations.for_each_ansatz_column(
        envelope, [&](std::size_t grid_index, AnsatzFamily family,
                      unsigned derivative_index, std::vector<Monomial>&& terms) {
          const auto seed_sector = sectors.sector_from_powers(envelope[grid_index]);
          if (column_stays_in_seed_sector(terms, seed_sector, sectors, canonicalizer)) {
            relations.push_back({std::move(terms), seed_sector,
                                 seed_dot_excess(envelope[grid_index]),
                                 derivative_index, family});
          }
        });
    symbolic.relations =
        deduplicate_relations(std::move(relations), config, canonicalizer);
    auto indexed_selection = index_selection_system(symbolic, canonicalizer);

    firefly::FFInt::set_new_prime(primes[0]);
    auto first = select_at_probe(indexed_selection, config, primes[0], 0);
    firefly::FFInt::set_new_prime(primes[1]);
    const auto second = select_at_probe(indexed_selection, config, primes[1], 0);
    firefly::FFInt::set_new_prime(primes[0]);
    if (first.selected_candidates != second.selected_candidates) {
      throw std::runtime_error(
          "finite-field probes disagree during global master selection");
    }

    const auto basis = selected_basis(candidates, first.selected_candidates);
    const auto border = selected_basis_border(config, basis);
    const auto closure_system =
        build_closure_system(symbolic, indexed_selection, sectors, canonicalizer,
                             first.selected_candidates, border);
    auto anchor =
        close_at_probe(closure_system, config, primes[0], 0, expansion_rounds, false);

    std::set<reduction::detail::ProjectedSeedGroup,
             reduction::detail::ProjectedSeedGroupLess>
        residual_groups = anchor.residual_groups;

    std::optional<ClosureSystem> supported_system;
    if (residual_groups.empty()) {
      supported_system.emplace(
          restrict_closure_system(closure_system, anchor.ansatz_order));
    }
    if (residual_groups.empty()) {
      const auto exact = verify_exact_closure(*supported_system, config, primes[0], 0);
      residual_groups.insert(exact.residual_groups.begin(),
                             exact.residual_groups.end());
    }
    if (residual_groups.empty()) {
      const auto replay = build_closure_replay_plan(std::move(*supported_system),
                                                    anchor, config, primes[0], 0);
      for (const auto [prime, point] :
           {std::pair{primes[1], std::size_t{0}}, std::pair{primes[0], std::size_t{1}},
            std::pair{primes[1], std::size_t{1}}}) {
        auto closure = replay_closure(replay, config, prime, point);
        residual_groups.insert(closure.residual_groups.begin(),
                               closure.residual_groups.end());
        if (!closure.closed && closure.residual_groups.empty())
          throw AnsatzClosureError();
      }
    }
    if (residual_groups.empty()) {
      accepted = std::move(first);
      break;
    }

    const std::vector requested(residual_groups.begin(), residual_groups.end());
    auto expansion = reduction::detail::expand_projected_seed_groups(
        envelope, requested, expansion_counts, sectors, canonicalizer);
    if (expansion.groups.empty() || expansion.points.empty())
      throw AnsatzClosureError();
    envelope.insert(envelope.end(), std::make_move_iterator(expansion.points.begin()),
                    std::make_move_iterator(expansion.points.end()));
    ++expansion_rounds;
  }

  std::vector<Integral> result;
  result.reserve(accepted.selected_candidates.size());
  for (const std::size_t selected : accepted.selected_candidates) {
    result.push_back(candidates[selected]);
  }
  return result;
}

} // namespace masters::detail
