#include "symmetry/Symmetry.hpp"

#include "symmetry/detail/Graph.hpp"
#include "symmetry/detail/SymmetryInternal.hpp"
#include "topology/SectorPolynomial.hpp"
#include "topology/SectorUtils.hpp"

#include <algorithm>
#include <bit>
#include <compare>
#include <cstdint>
#include <deque>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <ranges>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

using symmetry::detail::analyze_graph;
using symmetry::detail::normalize_generators;
using symmetry::detail::validate_canonical_labels;
using symmetry::detail::validate_lp_invariance;
using symmetry::detail::validate_permutation;

struct CanonicalGraphCertificate {
  std::vector<std::uint32_t> colors;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;

  auto operator<=>(const CanonicalGraphCertificate&) const = default;
};

CanonicalGraphCertificate
make_certificate(const symmetry::detail::ColoredGraph& graph,
                 std::span<const std::uint32_t> canonical_labels)
{
  validate_canonical_labels(canonical_labels, graph.colors.size());
  CanonicalGraphCertificate result;
  result.colors.resize(graph.colors.size());
  for (std::size_t vertex = 0; vertex < graph.colors.size(); ++vertex) {
    result.colors[canonical_labels[vertex]] = graph.colors[vertex];
  }
  result.edges.reserve(graph.edges.size());
  for (const auto& [left, right] : graph.edges) {
    auto mapped_left = canonical_labels[left];
    auto mapped_right = canonical_labels[right];
    if (mapped_right < mapped_left) {
      std::swap(mapped_left, mapped_right);
    }
    result.edges.emplace_back(mapped_left, mapped_right);
  }
  std::ranges::sort(result.edges);
  return result;
}

struct RestrictedPolynomial {
  std::vector<PolynomialTerm> terms;
  std::vector<std::uint32_t> active_variables;
};

RestrictedPolynomial restrict_polynomial(std::span<const PolynomialTerm> terms,
                                         std::size_t variable_count,
                                         std::uint32_t sector)
{
  RestrictedPolynomial result;
  for (std::size_t variable = 0; variable < variable_count; ++variable) {
    if (((sector >> variable) & 1U) != 0) {
      result.active_variables.push_back(static_cast<std::uint32_t>(variable));
    }
  }
  for (const auto& term : terms) {
    if (term.powers.size() != variable_count) {
      throw std::runtime_error("LP term has an unexpected variable count");
    }
    if (!polynomial_term_survives_sector(term, sector)) {
      continue;
    }
    PolynomialTerm restricted;
    restricted.weights = term.weights;
    restricted.powers.reserve(result.active_variables.size());
    for (const auto variable : result.active_variables) {
      restricted.powers.push_back(term.powers[variable]);
    }
    result.terms.push_back(std::move(restricted));
  }
  return result;
}

VariablePermutation identity_permutation(std::size_t variable_count)
{
  VariablePermutation result(variable_count);
  for (std::size_t variable = 0; variable < variable_count; ++variable) {
    result[variable] = static_cast<std::uint32_t>(variable);
  }
  return result;
}

VariablePermutation compose_permutations(const VariablePermutation& first,
                                         const VariablePermutation& second)
{
  if (first.size() != second.size()) {
    throw std::runtime_error("cannot compose permutations of different sizes");
  }
  VariablePermutation result(first.size());
  for (std::size_t variable = 0; variable < first.size(); ++variable) {
    result[variable] = second[first[variable]];
  }
  return result;
}

std::uint32_t map_sector(std::uint32_t sector, const VariablePermutation& permutation)
{
  std::uint32_t result = 0;
  for (std::size_t variable = 0; variable < permutation.size(); ++variable) {
    if (((sector >> variable) & 1U) != 0) {
      result |= std::uint32_t{1} << permutation[variable];
    }
  }
  return result;
}

VariablePermutation lift_generator(const VariablePermutation& local,
                                   std::span<const std::uint32_t> active_variables,
                                   std::size_t variable_count)
{
  validate_permutation(local, active_variables.size());
  auto result = identity_permutation(variable_count);
  for (std::size_t local_variable = 0; local_variable < active_variables.size();
       ++local_variable) {
    result[active_variables[local_variable]] = active_variables[local[local_variable]];
  }
  return result;
}

void validate_sector_relation(
    std::span<const PolynomialTerm> terms, std::size_t variable_count,
    std::uint32_t source_sector, std::uint32_t target_sector,
    std::span<const std::pair<std::uint32_t, std::uint32_t>> variable_map)
{
  if (variable_map.size() != static_cast<std::size_t>(std::popcount(source_sector))) {
    throw std::runtime_error("sector relation does not map every active variable");
  }
  std::vector<bool> seen_source(variable_count, false);
  std::vector<bool> seen_target(variable_count, false);
  std::uint32_t mapped_sector = 0;
  for (const auto& [source, target] : variable_map) {
    if (source >= variable_count || target >= variable_count ||
        ((source_sector >> source) & 1U) == 0 ||
        ((target_sector >> target) & 1U) == 0 || seen_source[source] ||
        seen_target[target]) {
      throw std::runtime_error("sector relation contains an invalid variable map");
    }
    seen_source[source] = true;
    seen_target[target] = true;
    mapped_sector |= std::uint32_t{1} << target;
  }
  if (mapped_sector != target_sector) {
    throw std::runtime_error("sector relation maps to the wrong target sector");
  }

  std::map<std::vector<std::uint8_t>, std::vector<std::int64_t>> expected;
  for (const auto& term : terms) {
    if (polynomial_term_survives_sector(term, target_sector)) {
      expected.emplace(term.powers, term.weights);
    }
  }
  std::size_t mapped_term_count = 0;
  for (const auto& term : terms) {
    if (!polynomial_term_survives_sector(term, source_sector)) {
      continue;
    }
    std::vector<std::uint8_t> mapped(variable_count, 0);
    for (const auto& [source, target] : variable_map) {
      mapped[target] = term.powers[source];
    }
    const auto found = expected.find(mapped);
    if (found == expected.end() || found->second != term.weights) {
      throw std::runtime_error(
          "sector relation does not preserve the restricted LP polynomial");
    }
    ++mapped_term_count;
  }
  if (mapped_term_count != expected.size()) {
    throw std::runtime_error(
        "sector relation does not bijectively map restricted LP terms");
  }
}

struct GlobalOrbitMember {
  std::uint32_t sector = 0;
  VariablePermutation transporter;
};

struct AnalyzedOrbit {
  std::uint32_t representative = 0;
  std::vector<GlobalOrbitMember> members;
  RestrictedPolynomial restricted;
  symmetry::detail::GraphAnalysis graph_analysis;
  CanonicalGraphCertificate certificate;
  std::vector<VariablePermutation> internal_generators;
};

std::vector<std::pair<std::uint32_t, std::uint32_t>>
canonical_variable_map(const AnalyzedOrbit& source, const AnalyzedOrbit& target)
{
  if (source.restricted.active_variables.size() !=
      target.restricted.active_variables.size()) {
    throw std::runtime_error(
        "canonically equal sector graphs have different variable counts");
  }
  std::unordered_map<std::uint32_t, std::uint32_t> target_by_label;
  for (std::size_t local = 0; local < target.restricted.active_variables.size();
       ++local) {
    target_by_label.emplace(target.graph_analysis.canonical_labels[local],
                            target.restricted.active_variables[local]);
  }
  std::vector<std::pair<std::uint32_t, std::uint32_t>> result;
  result.reserve(source.restricted.active_variables.size());
  for (std::size_t local = 0; local < source.restricted.active_variables.size();
       ++local) {
    const auto found =
        target_by_label.find(source.graph_analysis.canonical_labels[local]);
    if (found == target_by_label.end()) {
      throw std::runtime_error(
          "canonical sector labeling did not preserve variable colors");
    }
    result.emplace_back(source.restricted.active_variables[local], found->second);
  }
  return result;
}

} // namespace

namespace symmetry {

SubsectorSymmetryAnalysis
find_subsector_symmetry_classes(const TopologyConfig& topology, SymmetryBackend backend,
                                std::span<const VariablePermutation> global_generators)
{
  if (backend == SymmetryBackend::None) {
    if (!global_generators.empty()) {
      throw std::invalid_argument("none symmetry backend received generators");
    }
    SubsectorSymmetryAnalysis result;
    result.nonzero_sector_count =
        SectorUtils(topology).enumerate_nonzero_sectors().size();
    return result;
  }
  if (!backend_available(backend)) {
    throw std::runtime_error(
        std::format("requested symmetry backend '{}' is not available in this build "
                    "(available: {})",
                    backend_name(backend), available_backends_string()));
  }
  for (const auto& generator : global_generators) {
    validate_permutation(generator, topology.propagator_count);
    validate_lp_invariance(generator, topology.polynomial_terms);
  }

  const SectorUtils sector_utils(topology);
  const auto nonzero_sectors = sector_utils.enumerate_nonzero_sectors();
  const std::unordered_set<std::uint32_t> nonzero_set(nonzero_sectors.begin(),
                                                      nonzero_sectors.end());
  std::unordered_set<std::uint32_t> assigned;
  assigned.reserve(nonzero_sectors.size());
  const auto palette = detail::make_lp_color_palette(topology.polynomial_terms);
  std::vector<AnalyzedOrbit> analyzed_orbits;

  for (const std::uint32_t representative : nonzero_sectors) {
    if (assigned.contains(representative)) {
      continue;
    }
    std::unordered_map<std::uint32_t, VariablePermutation> transporters;
    transporters.emplace(representative,
                         identity_permutation(topology.propagator_count));
    std::deque<std::uint32_t> queue{representative};
    while (!queue.empty()) {
      const std::uint32_t current = queue.front();
      queue.pop_front();
      const auto current_transporter = transporters.at(current);
      for (const auto& generator : global_generators) {
        const std::uint32_t next = map_sector(current, generator);
        if (!nonzero_set.contains(next)) {
          throw std::runtime_error(
              "global LP symmetry mapped a non-zero sector to a zero sector");
        }
        if (transporters.contains(next)) {
          continue;
        }
        transporters.emplace(next,
                             compose_permutations(current_transporter, generator));
        queue.push_back(next);
      }
    }

    AnalyzedOrbit orbit;
    orbit.representative = representative;
    orbit.members.reserve(transporters.size());
    for (auto& [sector, transporter] : transporters) {
      if (assigned.contains(sector)) {
        throw std::runtime_error("global symmetry sector orbits overlap");
      }
      assigned.insert(sector);
      orbit.members.push_back({sector, std::move(transporter)});
    }
    std::ranges::sort(orbit.members, {}, &GlobalOrbitMember::sector);

    orbit.restricted = restrict_polynomial(topology.polynomial_terms,
                                           topology.propagator_count, representative);
    if (orbit.restricted.terms.empty()) {
      throw std::runtime_error("non-zero sector has an empty LP restriction");
    }
    const auto graph = detail::make_lp_graph(
        orbit.restricted.terms, orbit.restricted.active_variables.size(), palette);
    orbit.graph_analysis =
        analyze_graph(graph, orbit.restricted.active_variables.size(), backend);
    normalize_generators(orbit.graph_analysis.generators,
                         orbit.restricted.active_variables.size(),
                         orbit.restricted.terms);
    orbit.certificate = make_certificate(graph, orbit.graph_analysis.canonical_labels);
    orbit.internal_generators.reserve(orbit.graph_analysis.generators.size());
    for (const auto& generator : orbit.graph_analysis.generators) {
      orbit.internal_generators.push_back(lift_generator(
          generator, orbit.restricted.active_variables, topology.propagator_count));
    }
    std::ranges::sort(orbit.internal_generators);
    analyzed_orbits.push_back(std::move(orbit));
  }

  if (assigned.size() != nonzero_sectors.size()) {
    throw std::runtime_error("sub-sector symmetry search lost a non-zero sector");
  }

  std::map<CanonicalGraphCertificate, std::vector<std::size_t>> buckets;
  for (std::size_t index = 0; index < analyzed_orbits.size(); ++index) {
    buckets[analyzed_orbits[index].certificate].push_back(index);
  }

  SubsectorSymmetryAnalysis result;
  result.nonzero_sector_count = nonzero_sectors.size();
  auto& classes = result.classes;
  classes.reserve(buckets.size());
  for (const auto& [certificate, indices] : buckets) {
    static_cast<void>(certificate);
    if (indices.empty()) {
      continue;
    }
    const auto& base = analyzed_orbits[indices.front()];
    SectorSymmetryClass sector_class;
    sector_class.representative = base.representative;
    sector_class.generators = base.internal_generators;
    for (const std::size_t orbit_index : indices) {
      const auto& target_orbit = analyzed_orbits[orbit_index];
      const auto base_to_orbit = canonical_variable_map(base, target_orbit);
      for (const auto& member : target_orbit.members) {
        if (member.sector == base.representative) {
          continue;
        }
        SectorSymmetryRelation relation;
        relation.target_sector = member.sector;
        relation.variable_map.reserve(base_to_orbit.size());
        for (const auto& [source, orbit_target] : base_to_orbit) {
          relation.variable_map.emplace_back(source, member.transporter[orbit_target]);
        }
        validate_sector_relation(topology.polynomial_terms, topology.propagator_count,
                                 base.representative, member.sector,
                                 relation.variable_map);
        sector_class.relations.push_back(std::move(relation));
      }
    }
    std::ranges::sort(sector_class.relations, {},
                      &SectorSymmetryRelation::target_sector);
    if (!sector_class.generators.empty() || !sector_class.relations.empty()) {
      classes.push_back(std::move(sector_class));
    }
  }
  std::ranges::sort(classes, {}, &SectorSymmetryClass::representative);
  return result;
}

std::vector<std::size_t> find_sector_monomial_orbit_representatives(
    const TopologyConfig& topology, std::uint32_t sector,
    std::span<const std::vector<int>> monomials, SymmetryBackend backend)
{
  if (!backend_available(backend)) {
    throw std::runtime_error(
        std::format("requested symmetry backend '{}' is not available in this build "
                    "(available: {})",
                    backend_name(backend), available_backends_string()));
  }
  if (monomials.empty()) {
    return {};
  }

  const std::uint32_t propagator_mask =
      (std::uint32_t{1} << topology.propagator_count) - 1U;
  if (sector == 0 || (sector & ~propagator_mask) != 0) {
    throw std::invalid_argument("candidate monomial sector is invalid");
  }
  const auto restricted =
      restrict_polynomial(topology.polynomial_terms, topology.propagator_count, sector);
  if (restricted.terms.empty()) {
    throw std::runtime_error("candidate monomial sector has an empty LP restriction");
  }
  for (const auto& monomial : monomials) {
    if (monomial.size() != restricted.active_variables.size() ||
        std::ranges::any_of(monomial, [](int exponent) { return exponent < 0; })) {
      throw std::invalid_argument(
          "candidate monomial powers do not match the active sector variables");
    }
  }
  if (backend == SymmetryBackend::None) {
    std::vector<std::size_t> representatives(monomials.size());
    std::iota(representatives.begin(), representatives.end(), 0);
    return representatives;
  }

  const auto palette = detail::make_lp_color_palette(restricted.terms);
  const auto base_graph = detail::make_lp_graph(
      restricted.terms, restricted.active_variables.size(), palette);
  const std::uint32_t first_candidate_color =
      *std::ranges::max_element(base_graph.colors) + 1U;
  std::set<int> exponent_values;
  for (const auto& monomial : monomials) {
    exponent_values.insert(monomial.begin(), monomial.end());
  }
  if (exponent_values.size() >
      std::numeric_limits<std::uint32_t>::max() - first_candidate_color) {
    throw std::runtime_error("too many candidate exponent colors");
  }
  std::map<int, std::uint32_t> color_by_exponent;
  std::uint32_t next_color = first_candidate_color;
  for (const int exponent : exponent_values) {
    color_by_exponent.emplace(exponent, next_color++);
  }

  std::map<CanonicalGraphCertificate, std::size_t> first_by_certificate;
  std::vector<std::size_t> representatives;
  representatives.reserve(monomials.size());
  for (std::size_t index = 0; index < monomials.size(); ++index) {
    auto graph = base_graph;
    for (std::size_t variable = 0; variable < restricted.active_variables.size();
         ++variable) {
      graph.colors[variable] = color_by_exponent.at(monomials[index][variable]);
    }
    const auto analysis =
        analyze_graph(graph, restricted.active_variables.size(), backend);
    auto certificate = make_certificate(graph, analysis.canonical_labels);
    if (first_by_certificate.emplace(std::move(certificate), index).second) {
      representatives.push_back(index);
    }
  }
  return representatives;
}

} // namespace symmetry
