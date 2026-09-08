#include "masters/MasterFinder.hpp"

#include "core/ParallelForExecutor.hpp"
#include "core/ProbeValues.hpp"
#include "masters/detail/GlobalBasisSelector.hpp"
#include "masters/detail/IsolatedSymmetryBasis.hpp"
#include "masters/detail/MasterIntegralOrdering.hpp"
#include "masters/detail/MonomialPreference.hpp"
#include "masters/detail/NonisolatedSingular.hpp"
#include "masters/detail/SingularProcess.hpp"
#include "symmetry/Symmetry.hpp"
#include "topology/IntegralLayout.hpp"
#include "topology/SectorPolynomial.hpp"
#include "topology/SectorUtils.hpp"

#include <sys/wait.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <format>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr std::array<std::uint32_t, 2> kProbeCharacteristics{32003, 31991};
std::uint32_t probe_value(std::size_t parameter, std::size_t attempt)
{
  return probe_values::value(parameter, attempt);
}

std::string marker(std::size_t probe, const std::string& name)
{
  return std::format("FR_MASTER_P{}_{}", probe, name);
}

std::string sector_notation(std::uint32_t sector, const TopologyConfig& topology)
{
  std::ostringstream output;
  output << '[';
  std::size_t active_variable = 0;
  for (std::size_t slot = 0; slot < topology.integral_count; ++slot) {
    if (slot != 0) {
      output << ',';
    }
    if (topology.top_sector[slot] == 0) {
      output << '0';
    } else {
      output << ((sector >> active_variable) & 1U);
      ++active_variable;
    }
  }
  output << ']';
  return output.str();
}

std::uint32_t global_variable(const TopologyConfig& topology, std::uint32_t variable)
{
  if (variable >= topology.propagator_slots.size()) {
    throw std::runtime_error("active propagator variable is out of range");
  }
  return topology.propagator_slots[variable];
}

std::uint32_t finite_field_coefficient(const PolynomialTerm& term, std::size_t probe,
                                       std::uint32_t characteristic)
{
  if (term.weights.empty()) {
    throw std::runtime_error("LP term has no coefficient weights");
  }
  auto modular_weight = [characteristic](std::int64_t weight) {
    const std::int64_t modulus = characteristic;
    std::int64_t result = weight % modulus;
    if (result < 0) {
      result += modulus;
    }
    return static_cast<std::uint64_t>(result);
  };
  std::uint64_t value = modular_weight(term.weights[0]);
  for (std::size_t parameter = 1; parameter < term.weights.size(); ++parameter) {
    value +=
        modular_weight(term.weights[parameter]) * probe_value(parameter - 1, probe);
    value %= characteristic;
  }
  return static_cast<std::uint32_t>(value);
}

std::string sector_polynomial(const TopologyConfig& topology, std::uint32_t sector,
                              std::span<const std::uint32_t> active, std::size_t probe,
                              std::uint32_t characteristic)
{
  std::vector<std::string> terms;
  for (const auto& term : topology.polynomial_terms) {
    if (term.powers.size() != topology.propagator_count ||
        term.weights.size() != topology.kinematic_parameters.size() + 1) {
      throw std::runtime_error("compiled LP term has an invalid shape");
    }
    if (!polynomial_term_survives_sector(term, sector)) {
      continue;
    }
    const std::uint32_t coefficient =
        finite_field_coefficient(term, probe, characteristic);
    if (coefficient == 0) {
      continue;
    }
    std::vector<std::string> factors{std::to_string(coefficient)};
    for (const auto variable : active) {
      const auto exponent = term.powers[variable];
      const auto global = global_variable(topology, variable);
      if (exponent == 1) {
        factors.push_back(std::format("x{}", global + 1));
      } else if (exponent > 1) {
        factors.push_back(std::format("x{}^{}", global + 1, exponent));
      }
    }
    terms.push_back(std::accumulate(std::next(factors.begin()), factors.end(),
                                    factors.front(),
                                    [](std::string lhs, const std::string& rhs) {
                                      return std::move(lhs) + "*" + rhs;
                                    }));
  }
  if (terms.empty()) {
    throw std::runtime_error(
        std::format("finite-field probe {} annihilated LP polynomial in sector {}",
                    probe + 1, sector_notation(sector, topology)));
  }
  return std::accumulate(std::next(terms.begin()), terms.end(), terms.front(),
                         [](std::string lhs, const std::string& rhs) {
                           return std::move(lhs) + "+" + rhs;
                         });
}

std::string ring_variables(const TopologyConfig& topology,
                           std::span<const std::uint32_t> active, bool tmp)
{
  std::ostringstream output;
  output << '(';
  bool first = true;
  if (tmp) {
    output << "tmp";
    first = false;
  }
  for (const auto variable : active) {
    if (!first) {
      output << ',';
    }
    first = false;
    output << 'x' << global_variable(topology, variable) + 1;
  }
  output << ')';
  return output.str();
}

const SectorSymmetryClass* sector_symmetry(const TopologyConfig& topology,
                                           std::uint32_t sector)
{
  if (!topology.symmetry || topology.symmetry->backend == SymmetryBackend::None)
    return nullptr;
  for (const auto& symmetry_class : topology.symmetry->sector_classes) {
    if (symmetry_class.representative == sector && !symmetry_class.generators.empty())
      return &symmetry_class;
  }
  return nullptr;
}

std::string symmetry_basis_script(const TopologyConfig& topology, std::uint32_t sector,
                                  std::span<const std::uint32_t> active,
                                  std::size_t probe)
{
  const auto* symmetry_class = sector_symmetry(topology, sector);
  if (symmetry_class == nullptr) return {};
  std::ostringstream output;
  output << "if (dX" << probe << " > 0)\n{\nif (fr_error" << probe << " == 0)\n{\n"
         << "ideal fr_symmetry_relations; int fr_symmetry_i;\n";
  for (std::size_t index = 0; index < symmetry_class->generators.size(); ++index) {
    const auto& permutation = symmetry_class->generators[index];
    output << "map fr_action" << index << "=fr_master_basis_" << probe;
    for (const auto variable : active) {
      if (variable >= permutation.size() ||
          std::ranges::find(active, permutation[variable]) == active.end())
        throw std::logic_error("sector symmetry does not preserve active variables");
      output << ",x" << global_variable(topology, permutation[variable]) + 1;
    }
    output << ";\nideal fr_image" << index << "=fr_action" << index << "(KB);\n"
           << "for (fr_symmetry_i=1; fr_symmetry_i<=size(KB); fr_symmetry_i++)\n"
           << "{ fr_symmetry_relations[size(fr_symmetry_relations)+1]="
           << "KB[fr_symmetry_i]-fr_image" << index << "[fr_symmetry_i]; }\n";
  }
  output << "list fr_symmetry_result=fr_select_symmetry_basis(fr_selected_basis"
         << probe << ",fr_symmetry_relations,KB,GX);\n"
         << "ideal fr_symmetry_basis=fr_symmetry_result[1];\n"
         << "print(\"" << marker(probe, "SYMMETRY_STATUS_BEGIN") << "\");\n"
         << "print(fr_symmetry_result[2]);\n"
         << "print(\"" << marker(probe, "SYMMETRY_STATUS_END") << "\");\n"
         << "print(\"" << marker(probe, "SYMMETRY_DIM_BEGIN") << "\");\n"
         << "print(size(fr_symmetry_basis));\n"
         << "print(\"" << marker(probe, "SYMMETRY_DIM_END") << "\");\n"
         << "print(\"" << marker(probe, "SYMMETRY_KBASE_BEGIN") << "\");\n"
         << "for (fr_symmetry_i=1; fr_symmetry_i<=size(fr_symmetry_basis); "
            "fr_symmetry_i++)\n"
         << "{ print(string(leadexp(fr_symmetry_basis[fr_symmetry_i]))); }\n"
         << "print(\"" << marker(probe, "SYMMETRY_KBASE_END") << "\");\n}\n}\n";
  return output.str();
}

std::string probe_script(const TopologyConfig& topology, std::uint32_t sector,
                         std::span<const std::uint32_t> active, std::size_t probe)
{
  const auto characteristic = kProbeCharacteristics.at(probe);
  const std::string full_ring = std::format("fr_master_full_{}", probe);
  const std::string basis_ring = std::format("fr_master_basis_{}", probe);
  std::ostringstream output;
  output << "ring " << full_ring << " = " << characteristic << ','
         << ring_variables(topology, active, true) << ",dp;\n"
         << "poly LP = "
         << sector_polynomial(topology, sector, active, probe, characteristic) << ";\n"
         << "ideal J = ";
  for (std::size_t index = 0; index < active.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << "diff(LP,x" << global_variable(topology, active[index]) + 1 << ')';
  }
  output << ",1-tmp*LP;\n"
         << "ideal GB = slimgb(J);\n"
         << "print(\"" << marker(probe, "DIM_FULL_BEGIN") << "\");\n"
         << "int d" << probe << " = vdim(GB); print(string(d" << probe << "));\n"
         << "print(\"" << marker(probe, "DIM_FULL_END") << "\");\n"
         << "ideal E = eliminate(GB,tmp);\n"
         << "ring " << basis_ring << " = " << characteristic << ','
         << ring_variables(topology, active, false) << ",dp;\n"
         << "ideal IX = imap(" << full_ring << ",E);\n"
         << "ideal GX = slimgb(IX);\n"
         << "print(\"" << marker(probe, "DIM_BASIS_BEGIN") << "\");\n"
         << "int dX" << probe << " = vdim(GX); print(string(dX" << probe << "));\n"
         << "print(\"" << marker(probe, "DIM_BASIS_END") << "\");\n"
         << "if (dX" << probe << " == -1)\n{\n"
         << "  LIB \"primdec.lib\";\n"
         << masters::detail::kNonisolatedSingularProcedures << "  list fr_count_result"
         << probe << " = fr_count_nonisolated(GX,0);\n"
         << "  ideal fr_nonisolated_basis" << probe << ";\n"
         << "  if (fr_count_result" << probe << "[2] == 0)\n  {\n"
         << "    list fr_selection_result" << probe
         << " = fr_select_nonisolated_basis(fr_count_result" << probe
         << "[3],fr_count_result" << probe << "[1]);\n"
         << "    fr_nonisolated_basis" << probe << " = fr_selection_result" << probe
         << "[1];\n"
         << "    if (fr_selection_result" << probe << "[2] != 0)\n"
         << "    { fr_count_result" << probe << "[2] = fr_selection_result" << probe
         << "[2]; }\n"
         << "  }\n"
         << "  print(\"" << marker(probe, "NONISOLATED_COUNT_BEGIN")
         << "\"); print(string(fr_count_result" << probe << "[1]));\n"
         << "  print(\"" << marker(probe, "NONISOLATED_COUNT_END") << "\");\n"
         << "  print(\"" << marker(probe, "NONISOLATED_STATUS_BEGIN")
         << "\"); print(string(fr_count_result" << probe << "[2]));\n"
         << "  print(\"" << marker(probe, "NONISOLATED_STATUS_END") << "\");\n"
         << "  print(\"" << marker(probe, "KBASE_BEGIN") << "\");\n"
         << "  if (fr_count_result" << probe << "[2] == 0)\n  {\n"
         << "    int fr_nonisolated_index" << probe << ";\n"
         << "    for (fr_nonisolated_index" << probe << "=1; "
         << "fr_nonisolated_index" << probe << "<=size(fr_nonisolated_basis" << probe
         << "); fr_nonisolated_index" << probe << "++)\n"
         << "    { print(string(leadexp(fr_nonisolated_basis" << probe
         << "[fr_nonisolated_index" << probe << "]))); }\n"
         << "  }\n"
         << "  print(\"" << marker(probe, "KBASE_END") << "\");\n"
         << "}\n"
         << "if (dX" << probe << " > 0)\n{\n"
         << "  ideal KB = kbase(GX);\n"
         << "  proc fr_max_exp_" << probe << "(poly fr_m)\n"
         << "  {\n"
         << "    intvec fr_e = leadexp(fr_m); int fr_max = 0; int fr_v;\n"
         << "    for (fr_v=1; fr_v<=size(fr_e); fr_v++)\n"
         << "    { if (fr_e[fr_v] > fr_max) { fr_max = fr_e[fr_v]; } }\n"
         << "    return(fr_max);\n"
         << "  }\n"
         << "  proc fr_preferred_" << probe << "(poly fr_a, poly fr_b)\n"
         << "  {\n"
         << "    int fr_ma = fr_max_exp_" << probe << "(fr_a);\n"
         << "    int fr_mb = fr_max_exp_" << probe << "(fr_b);\n"
         << "    if (fr_ma < fr_mb) { return(1); }\n"
         << "    if (fr_ma > fr_mb) { return(0); }\n"
         << "    intvec fr_ea = leadexp(fr_a); intvec fr_eb = leadexp(fr_b);\n"
         << "    int fr_v;\n"
         << "    for (fr_v=1; fr_v<=size(fr_ea); fr_v++)\n"
         << "    {\n"
         << "      if (fr_ea[fr_v] > fr_eb[fr_v]) { return(1); }\n"
         << "      if (fr_ea[fr_v] < fr_eb[fr_v]) { return(0); }\n"
         << "    }\n"
         << "    return(0);\n"
         << "  }\n"
         << "  matrix fr_matrix" << probe << "[dX" << probe << "][dX" << probe << "];\n"
         << "  ideal fr_selected_basis" << probe << ",fr_degree_candidates" << probe
         << ";\n"
         << "  int fr_selected" << probe << "=0; int fr_degree" << probe
         << "=0; int fr_error" << probe << "=0;\n"
         << "  if (size(KB) != dX" << probe << ") { fr_error" << probe << "=1; }\n"
         << "  int fr_i" << probe << ",fr_j" << probe << ",fr_k" << probe << ",fr_best"
         << probe << ",fr_found" << probe << ";\n"
         << "  poly fr_nf" << probe << ",fr_candidate" << probe << ";\n"
         << "  intvec fr_exp" << probe << ",fr_used" << probe << ";\n"
         << "  number fr_coef" << probe << ";\n"
         << "  while (fr_selected" << probe << " < dX" << probe << " && fr_degree"
         << probe << " < dX" << probe << " && fr_error" << probe << " == 0)\n"
         << "  {\n"
         << "    if (fr_degree" << probe << " == 0)\n"
         << "    { fr_degree_candidates" << probe << "=1; }\n"
         << "    else\n"
         << "    { fr_degree_candidates" << probe << "=maxideal(fr_degree" << probe
         << "); }\n"
         << "    fr_used" << probe << "=0;\n"
         << "    fr_used" << probe << "[size(fr_degree_candidates" << probe << ")]=0;\n"
         << "    for (fr_k" << probe << "=1; fr_k" << probe
         << "<=size(fr_degree_candidates" << probe << "); fr_k" << probe << "++)\n"
         << "    {\n"
         << "      fr_best" << probe << "=0;\n"
         << "      for (fr_i" << probe << "=1; fr_i" << probe
         << "<=size(fr_degree_candidates" << probe << "); fr_i" << probe << "++)\n"
         << "      {\n"
         << "        if (fr_used" << probe << "[fr_i" << probe << "] == 0)\n"
         << "        {\n"
         << "          if (fr_best" << probe << " == 0)\n"
         << "          { fr_best" << probe << "=fr_i" << probe << "; }\n"
         << "          else\n"
         << "          {\n"
         << "            if (fr_preferred_" << probe << "(fr_degree_candidates" << probe
         << "[fr_i" << probe << "],fr_degree_candidates" << probe << "[fr_best" << probe
         << "]))\n"
         << "            { fr_best" << probe << "=fr_i" << probe << "; }\n"
         << "          }\n"
         << "        }\n"
         << "      }\n"
         << "      fr_used" << probe << "[fr_best" << probe << "]=1;\n"
         << "      fr_candidate" << probe << "=fr_degree_candidates" << probe
         << "[fr_best" << probe << "];\n"
         << "      fr_nf" << probe << "=reduce(fr_candidate" << probe << ",GX);\n"
         << "      for (fr_j" << probe << "=1; fr_j" << probe << "<=dX" << probe
         << "; fr_j" << probe << "++)\n"
         << "      { fr_matrix" << probe << "[fr_j" << probe << ",fr_selected" << probe
         << "+1]=0; }\n"
         << "      while (fr_nf" << probe << " != 0 && fr_error" << probe << " == 0)\n"
         << "      {\n"
         << "        fr_exp" << probe << "=leadexp(fr_nf" << probe << ");\n"
         << "        fr_coef" << probe << "=leadcoef(fr_nf" << probe << ");\n"
         << "        fr_found" << probe << "=0;\n"
         << "        for (fr_j" << probe << "=1; fr_j" << probe << "<=size(KB); fr_j"
         << probe << "++)\n"
         << "        {\n"
         << "          if (fr_exp" << probe << " == leadexp(KB[fr_j" << probe << "]))\n"
         << "          { fr_matrix" << probe << "[fr_j" << probe << ",fr_selected"
         << probe << "+1]=fr_coef" << probe << "; fr_found" << probe << "=1; break; }\n"
         << "        }\n"
         << "        if (fr_found" << probe << " == 0)\n"
         << "        { fr_error" << probe << "=1; }\n"
         << "        fr_nf" << probe << "=fr_nf" << probe << "-lead(fr_nf" << probe
         << ");\n"
         << "      }\n"
         << "      if (fr_error" << probe << " == 0)\n"
         << "      {\n"
         << "        if (rank(fr_matrix" << probe << ") > fr_selected" << probe << ")\n"
         << "        { fr_selected" << probe << "++; fr_selected_basis" << probe
         << "[fr_selected" << probe << "]=fr_candidate" << probe << "; }\n"
         << "      }\n"
         << "      if (fr_selected" << probe << " == dX" << probe << " || fr_error"
         << probe << " != 0) { break; }\n"
         << "    }\n"
         << "    fr_degree" << probe << "++;\n"
         << "  }\n"
         << "  print(\"" << marker(probe, "KBASE_BEGIN") << "\");\n"
         << "  if (size(KB) != dX" << probe << " || fr_selected" << probe << " != dX"
         << probe << " || fr_error" << probe << " != 0)\n"
         << "  { print(\"preferred monomial selection failed\"); }\n"
         << "  else\n"
         << "  {\n"
         << "    for (fr_i" << probe << "=1; fr_i" << probe
         << "<=size(fr_selected_basis" << probe << "); fr_i" << probe << "++)\n"
         << "    { print(string(leadexp(fr_selected_basis" << probe << "[fr_i" << probe
         << "]))); }\n"
         << "  }\n"
         << "  print(\"" << marker(probe, "KBASE_END") << "\");\n"
         << "}\n"
         << symmetry_basis_script(topology, sector, active, probe);
  return output.str();
}

std::vector<std::string> lines_between(const std::string& output,
                                       const std::string& begin, const std::string& end)
{
  std::istringstream input(output);
  std::string line;
  bool inside = false;
  std::vector<std::string> result;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line == begin) {
      inside = true;
      continue;
    }
    if (inside && line == end) {
      return result;
    }
    if (inside) {
      result.push_back(line);
    }
  }
  throw std::runtime_error("Singular output is missing marker " + begin);
}

std::string trim(std::string value)
{
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

int parse_dimension(const std::string& output, std::size_t probe,
                    const std::string& label)
{
  const auto lines = lines_between(output, marker(probe, label + "_BEGIN"),
                                   marker(probe, label + "_END"));
  for (auto current = lines.rbegin(); current != lines.rend(); ++current) {
    const std::string value = trim(*current);
    if (value.empty()) {
      continue;
    }
    std::size_t parsed = 0;
    try {
      const int result = std::stoi(value, &parsed);
      if (parsed == value.size()) {
        if (result < -1) {
          throw std::runtime_error("Singular returned a vdim below -1");
        }
        return result;
      }
    } catch (const std::invalid_argument&) {
    } catch (const std::out_of_range&) {
    }
  }
  throw std::runtime_error("Singular output contains no integer " + label);
}

std::vector<int> parse_exponents(const std::string& line, std::size_t variable_count)
{
  std::istringstream input(line);
  std::string item;
  std::vector<int> result;
  while (std::getline(input, item, ',')) {
    const std::string value = trim(item);
    std::size_t parsed = 0;
    int exponent = 0;
    try {
      exponent = std::stoi(value, &parsed);
    } catch (const std::exception&) {
      throw std::runtime_error("invalid Singular exponent vector: " + line);
    }
    if (parsed != value.size() || exponent < 0) {
      throw std::runtime_error("invalid Singular exponent vector: " + line);
    }
    result.push_back(exponent);
  }
  if (result.size() != variable_count) {
    throw std::runtime_error("invalid Singular exponent vector: " + line);
  }
  return result;
}

struct ProbeResult {
  int dimension = 0;
  bool nonisolated = false;
  std::vector<std::vector<int>> monomials;
  std::optional<std::vector<std::vector<int>>> symmetry_monomials;
};

std::vector<std::vector<int>> parse_monomials(const std::string& output,
                                              std::size_t probe,
                                              std::size_t variable_count,
                                              const std::string& label = "KBASE")
{
  std::vector<std::vector<int>> monomials;
  for (const auto& line : lines_between(output, marker(probe, label + "_BEGIN"),
                                        marker(probe, label + "_END"))) {
    const std::string value = trim(line);
    if (!value.empty() && !value.starts_with("//")) {
      if (value == "preferred monomial selection failed") {
        throw std::runtime_error(value);
      }
      monomials.push_back(parse_exponents(value, variable_count));
    }
  }
  std::ranges::sort(monomials, [](const auto& lhs, const auto& rhs) {
    return masters::detail::monomial_preferred(lhs, rhs);
  });
  return monomials;
}

ProbeResult parse_probe_result(const std::string& output, std::size_t probe,
                               std::size_t variable_count, bool has_symmetry)
{
  const int full_dimension = parse_dimension(output, probe, "DIM_FULL");
  const int basis_dimension = parse_dimension(output, probe, "DIM_BASIS");
  if (full_dimension != basis_dimension) {
    throw std::runtime_error(std::format(
        "Singular dimensions before and after eliminating tmp differ: {} != {}",
        full_dimension, basis_dimension));
  }
  if (full_dimension == -1) {
    const int status = parse_dimension(output, probe, "NONISOLATED_STATUS");
    if (status != 0) {
      static constexpr std::array<std::string_view, 9> errors{
          "",
          "recursion did not terminate",
          "component has multiple nonlinear generators",
          "nonlinear generator vanished after linear reduction",
          "recursive critical ideal did not lower the dimension",
          "positive-dimensional primary component is non-radical",
          "terminal quotient dimensions do not match the recursive count",
          "terminal normal form is outside its kbase coordinates",
          "original monomials do not span the terminal quotient direct sum"};
      const std::string_view detail =
          status >= 0 && static_cast<std::size_t>(status) < errors.size()
              ? errors[static_cast<std::size_t>(status)]
              : "unknown recursive counter status";
      throw std::runtime_error(
          std::format("unsupported non-isolated critical component: {}", detail));
    }
    const int count = parse_dimension(output, probe, "NONISOLATED_COUNT");
    if (count < 0) {
      throw std::runtime_error("recursive critical-point count is negative");
    }
    auto monomials = parse_monomials(output, probe, variable_count);
    if (monomials.size() != static_cast<std::size_t>(count)) {
      throw std::runtime_error(std::format(
          "selected non-isolated monomial basis size {} does not match recursive "
          "count {}",
          monomials.size(), count));
    }
    return {count, true, std::move(monomials), std::nullopt};
  }
  if (full_dimension == 0) {
    return {};
  }
  auto monomials = parse_monomials(output, probe, variable_count);
  if (monomials.size() != static_cast<std::size_t>(full_dimension)) {
    throw std::runtime_error(
        std::format("selected monomial basis size {} does not match vdim {}",
                    monomials.size(), full_dimension));
  }
  std::optional<std::vector<std::vector<int>>> symmetry_monomials;
  if (has_symmetry) {
    if (parse_dimension(output, probe, "SYMMETRY_STATUS") != 0)
      throw std::runtime_error("symmetry quotient basis selection failed");
    const int dimension = parse_dimension(output, probe, "SYMMETRY_DIM");
    symmetry_monomials =
        parse_monomials(output, probe, variable_count, "SYMMETRY_KBASE");
    if (dimension < 0 || dimension > full_dimension ||
        symmetry_monomials->size() != static_cast<std::size_t>(dimension))
      throw std::runtime_error("symmetry quotient basis dimension is inconsistent");
    for (const auto& monomial : *symmetry_monomials) {
      if (std::ranges::find(monomials, monomial) == monomials.end())
        throw std::runtime_error("symmetry quotient selected an unknown candidate");
    }
    if (std::adjacent_find(symmetry_monomials->begin(), symmetry_monomials->end()) !=
        symmetry_monomials->end())
      throw std::runtime_error("symmetry quotient selected duplicate candidates");
  }
  return {full_dimension, false, std::move(monomials), std::move(symmetry_monomials)};
}

struct SectorBasis {
  std::uint32_t sector = 0;
  bool nonisolated = false;
  std::vector<std::vector<int>> monomials;
  std::optional<std::vector<std::vector<int>>> symmetry_monomials;
};

SectorBasis calculate_sector_basis(const MasterFinderConfig& config,
                                   std::uint32_t sector)
{
  const auto active =
      integral_layout::active_variables(sector, config.propagator_count);
  std::string script(masters::detail::kIsolatedSymmetryBasisProcedure);
  for (std::size_t probe = 0; probe < kProbeCharacteristics.size(); ++probe) {
    script += probe_script(config, sector, active, probe);
    script += '\n';
  }
  script += "exit;\n";
  const auto completed =
      masters::detail::run_singular_process(config.singular_path, script);
  if (!WIFEXITED(completed.status) || WEXITSTATUS(completed.status) != 0) {
    throw std::runtime_error(std::format(
        "Singular failed in sector {} (status {}): {}", sector_notation(sector, config),
        completed.status, trim(completed.stderr_text)));
  }

  std::array<ProbeResult, 2> probes;
  for (std::size_t probe = 0; probe < probes.size(); ++probe) {
    try {
      probes[probe] = parse_probe_result(completed.stdout_text, probe, active.size(),
                                         sector_symmetry(config, sector) != nullptr);
    } catch (const std::exception& error) {
      throw std::runtime_error(std::format(
          "failed to parse Singular probe {} in sector {}: {}; stderr: {}", probe + 1,
          sector_notation(sector, config), error.what(), trim(completed.stderr_text)));
    }
  }
  if (probes[0].dimension != probes[1].dimension ||
      probes[0].nonisolated != probes[1].nonisolated ||
      probes[0].monomials != probes[1].monomials ||
      probes[0].symmetry_monomials != probes[1].symmetry_monomials) {
    throw std::runtime_error(std::format("finite-field probes disagree in sector {}",
                                         sector_notation(sector, config)));
  }
  return {sector, probes[0].nonisolated, std::move(probes[0].monomials),
          std::move(probes[0].symmetry_monomials)};
}

std::vector<std::uint32_t> representative_sectors(const MasterFinderConfig& config)
{
  const SectorUtils sector_utils(config);
  auto sectors = sector_utils.enumerate_nonzero_sectors();
  if (sectors.size() != config.symmetry->nonzero_sector_count) {
    throw std::runtime_error(
        "stored symmetry analysis disagrees with non-zero sector enumeration");
  }
  std::unordered_set<std::uint32_t> equivalent_targets;
  for (const auto& sector_class : config.symmetry->sector_classes) {
    for (const auto& relation : sector_class.relations) {
      equivalent_targets.insert(relation.target_sector);
    }
  }
  std::erase_if(sectors, [&](std::uint32_t sector) {
    return equivalent_targets.contains(sector);
  });
  std::ranges::sort(sectors, [](std::uint32_t lhs, std::uint32_t rhs) {
    const int lhs_size = std::popcount(lhs);
    const int rhs_size = std::popcount(rhs);
    return lhs_size != rhs_size ? lhs_size < rhs_size : lhs < rhs;
  });
  return sectors;
}

std::vector<SectorBasis>
calculate_all_sector_bases(const MasterFinderConfig& config,
                           const std::vector<std::uint32_t>& sectors)
{
  std::vector<std::optional<SectorBasis>> slots(sectors.size());
  const std::size_t worker_count =
      std::min<std::size_t>(config.threads, sectors.size());
  core::ParallelForExecutor executor(worker_count);
  executor.run(sectors.size(), [&](std::size_t index, std::size_t) {
    slots[index] = calculate_sector_basis(config, sectors[index]);
  });
  std::vector<SectorBasis> result;
  result.reserve(slots.size());
  for (auto& slot : slots) {
    if (!slot) {
      throw std::runtime_error("master finder worker lost a sector result");
    }
    result.push_back(std::move(*slot));
  }
  return result;
}

} // namespace

namespace masters {

MasterCandidateSet find_master_candidates(const MasterFinderConfig& config)
{
  if (!config.symmetry) {
    throw std::runtime_error("master finder requires a completed symmetry analysis");
  }
  if (config.threads == 0) {
    throw std::invalid_argument("master finder thread count must be positive");
  }
  if (config.singular_path.empty()) {
    throw std::invalid_argument("Singular executable path must be non-empty");
  }

  const auto sectors = representative_sectors(config);
  auto sector_bases = calculate_all_sector_bases(config, sectors);

  MasterCandidateSet result;
  const bool needs_global_selection =
      std::ranges::any_of(sector_bases, &SectorBasis::nonisolated);
  result.mode = needs_global_selection ? MasterCandidateMode::GlobalSelection
                                       : MasterCandidateMode::FinalBasis;
  for (auto& sector_basis : sector_bases) {
    const bool has_no_local_candidate = sector_basis.monomials.empty();
    if (sector_basis.nonisolated ||
        (needs_global_selection && has_no_local_candidate)) {
      result.relation_source_sectors.push_back(sector_basis.sector);
    }
    if (needs_global_selection) {
      // A local kbase is only a hint in the presence of positive-dimensional
      // source sectors.  Make every normal corner and one-dot representative
      // available to the global quotient-rank test; the elimination, not the
      // Singular count, decides which of them survive as masters.
      const std::size_t variable_count =
          static_cast<std::size_t>(std::popcount(sector_basis.sector));
      sector_basis.monomials.emplace_back(variable_count, 0);
      for (std::size_t variable = 0; variable < variable_count; ++variable) {
        std::vector<int> dot(variable_count, 0);
        dot[variable] = 1;
        sector_basis.monomials.push_back(std::move(dot));
      }
      std::ranges::sort(sector_basis.monomials, [](const auto& lhs, const auto& rhs) {
        return masters::detail::monomial_preferred(lhs, rhs);
      });
      sector_basis.monomials.erase(
          std::unique(sector_basis.monomials.begin(), sector_basis.monomials.end()),
          sector_basis.monomials.end());
    }
    if (!needs_global_selection && sector_basis.symmetry_monomials)
      sector_basis.monomials = std::move(*sector_basis.symmetry_monomials);
    const auto active =
        integral_layout::active_variables(sector_basis.sector, config.propagator_count);
    for (const auto& monomial : sector_basis.monomials) {
      Integral active_integral;
      active_integral.indices.assign(config.propagator_count, 0);
      for (std::size_t variable = 0; variable < active.size(); ++variable) {
        active_integral.indices[active[variable]] = 1 + monomial[variable];
      }
      auto integral = integral_layout::expand_active(config, active_integral);
      result.integrals.push_back(std::move(integral));
    }
  }
  masters::detail::canonical_sort_master_integrals(config, result.integrals);
  return result;
}

std::vector<Integral> find_master_integrals(const MasterFinderConfig& config)
{
  auto candidates = find_master_candidates(config);
  if (!candidates.requires_global_selection()) {
    return std::move(candidates.integrals);
  }
  return select_global_master_basis(config, candidates);
}

std::vector<Integral> select_global_master_basis(const MasterFinderConfig& config,
                                                 const MasterCandidateSet& candidates)
{
  if (!candidates.requires_global_selection()) {
    throw std::invalid_argument(
        "global master selection requires a global-selection candidate set");
  }
  return masters::detail::select_global_master_basis(
      config, candidates.integrals, candidates.relation_source_sectors);
}

} // namespace masters
