#pragma once

#include <compare>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct PolynomialTerm {
  std::vector<std::uint8_t> powers;
  // weights[0] is the constant coefficient. weights[i + 1] multiplies
  // kinematic_parameters[i]; d is represented separately by equation terms.
  std::vector<std::int64_t> weights;
};

struct Integral {
  std::vector<int> indices;

  bool operator==(const Integral&) const = default;
};

struct ExactRationalConstant {
  std::string numerator = "0";
  std::string denominator = "1";

  auto operator<=>(const ExactRationalConstant&) const = default;
};

enum class SymmetryBackend {
  None,
  Nauty,
  Bliss,
};

enum class BasisSelectionPolicy {
  Default,
  DSeparating,
};

using VariablePermutation = std::vector<std::uint32_t>;

struct SectorSymmetryRelation {
  std::uint32_t target_sector = 0;
  // Maps active variables of the class representative to active variables
  // of the target sector. Entries are sorted by their source variable.
  std::vector<std::pair<std::uint32_t, std::uint32_t>> variable_map;
};

struct SectorSymmetryClass {
  std::uint32_t representative = 0;
  // Full-size permutations that fix pinched variables of the representative.
  std::vector<VariablePermutation> generators;
  // Non-identity relations from the representative to equivalent sectors.
  std::vector<SectorSymmetryRelation> relations;
};

struct SymmetryAnalysis {
  SymmetryBackend backend = SymmetryBackend::None;
  std::vector<VariablePermutation> generators;
  std::size_t nonzero_sector_count = 0;
  // Only classes with an internal generator or a cross-sector relation.
  std::vector<SectorSymmetryClass> sector_classes;
};

struct ExtendedLpData {
  std::vector<PolynomialTerm> polynomial_terms;
  // Only exact extended-polynomial permutations are stored here. Sub-sector
  // relations remain attached to the denominator topology.
  std::vector<VariablePermutation> symmetry_generators;
};

struct TopologyConfig {
  unsigned loop_count = 1;
  // Number of active propagators participating in LP computations.
  unsigned propagator_count = 0;
  // Number of user-visible integral slots, including ISP placeholders.
  unsigned integral_count = 0;
  std::string integral_header = "F";
  std::vector<std::uint8_t> top_sector;
  // Maps compact active-propagator indices to user-visible integral slots.
  std::vector<std::uint32_t> propagator_slots;
  // Free FireFly variables. The compiler keeps d first when it is free for
  // output compatibility, but evaluators must use the explicit index below.
  std::vector<std::string> parameters;
  std::vector<std::string> kinematic_parameters;
  std::vector<std::uint32_t> kinematic_parameter_indices;
  std::optional<std::uint32_t> dimension_parameter_index;
  std::optional<ExactRationalConstant> dimension_value;
  std::map<std::string, ExactRationalConstant> numerics;
  std::vector<PolynomialTerm> polynomial_terms;
  std::optional<SymmetryAnalysis> symmetry;
  // Compiled from every propagator/ISP expression. It is consumed by the
  // extended symmetry report and top-LP target projection; denominator
  // equations keep using the compact fields above.
  ExtendedLpData extended_lp;
};

struct MasterFinderConfig : TopologyConfig {
  unsigned threads = 8;
  std::string singular_path = "Singular";
};

struct Config : MasterFinderConfig {
  std::vector<Integral> basis;
  std::vector<Integral> targets;
  BasisSelectionPolicy basis_selection = BasisSelectionPolicy::Default;
  bool factor_scan = false;
  bool shift_scan = false;
};
