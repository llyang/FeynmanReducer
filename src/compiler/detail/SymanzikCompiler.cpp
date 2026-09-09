#include "compiler/YamlCompiler.hpp"

#include "compiler/detail/ExactPolynomial.hpp"
#include "compiler/detail/ExpressionParser.hpp"
#include "compiler/detail/FlintUtils.hpp"
#include "compiler/detail/IntegralParser.hpp"
#include "symmetry/Symmetry.hpp"

#include <yaml-cpp/yaml.h>

#include <flint/fmpq.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <format>
#include <limits>
#include <map>
#include <numeric>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace compiler::detail;

std::vector<std::string> string_sequence(const YAML::Node& node,
                                         const std::string& label,
                                         bool allow_empty = true)
{
  if (!node || !node.IsSequence() || (!allow_empty && node.size() == 0)) {
    throw std::runtime_error(label + " must be a non-empty sequence");
  }
  std::vector<std::string> result;
  result.reserve(node.size());
  for (std::size_t index = 0; index < node.size(); ++index) {
    if (!node[index].IsScalar()) {
      throw std::runtime_error(std::format("{}[{}] must be a string", label, index));
    }
    result.push_back(node[index].as<std::string>());
  }
  return result;
}

std::vector<std::uint8_t> binary_sequence(const YAML::Node& node,
                                          const std::string& label)
{
  if (!node || !node.IsSequence() || node.size() == 0) {
    throw std::runtime_error(label + " must be a non-empty sequence");
  }
  std::vector<std::uint8_t> result;
  result.reserve(node.size());
  for (std::size_t index = 0; index < node.size(); ++index) {
    if (!node[index].IsScalar()) {
      throw std::runtime_error(std::format("{}[{}] must be 0 or 1", label, index));
    }
    const int value = node[index].as<int>();
    if (value != 0 && value != 1) {
      throw std::runtime_error(std::format("{}[{}] must be 0 or 1", label, index));
    }
    result.push_back(static_cast<std::uint8_t>(value));
  }
  return result;
}

bool valid_integral_header(const std::string& value)
{
  if (value.empty() || !(std::isalpha(static_cast<unsigned char>(value.front())) ||
                         value.front() == '$')) {
    return false;
  }
  return std::ranges::all_of(value | std::views::drop(1), [](char character) {
    return std::isalnum(static_cast<unsigned char>(character)) || character == '$';
  });
}

std::map<std::string, std::string> string_map(const YAML::Node& parent,
                                              const std::string& key)
{
  const YAML::Node node = parent[key];
  if (!node) {
    return {};
  }
  if (!node.IsMap()) {
    throw std::runtime_error(key + " must be a mapping");
  }
  std::map<std::string, std::string> result;
  for (const auto& entry : node) {
    result.emplace(entry.first.as<std::string>(), entry.second.as<std::string>());
  }
  return result;
}

std::map<std::string, std::string> numeric_string_map(const YAML::Node& root)
{
  const YAML::Node node = root["numerics"];
  if (!node) return {};
  if (!node.IsMap()) throw std::runtime_error("numerics must be a mapping");
  std::map<std::string, std::string> result;
  for (const auto& entry : node) {
    if (!entry.first.IsScalar() || !entry.second.IsScalar()) {
      throw std::runtime_error("numerics keys and values must be scalar strings");
    }
    const std::string key = entry.first.as<std::string>();
    if (!result.emplace(key, entry.second.as<std::string>()).second) {
      throw std::runtime_error("duplicate numerics key: " + key);
    }
  }
  return result;
}

std::string integer_text(const fmpz* value)
{
  char* raw = fmpz_get_str(nullptr, 10, value);
  if (raw == nullptr) throw std::runtime_error("FLINT failed to format an integer");
  std::string result(raw);
  flint_free(raw);
  return result;
}

ExactRationalConstant export_rational(const Rational& value)
{
  return {integer_text(fmpq_numref(value.raw())),
          integer_text(fmpq_denref(value.raw()))};
}

std::size_t require_single_symbol(const Polynomial& value, const std::string& label)
{
  if (value.terms.size() != 1 || !value.terms.begin()->second.is_one()) {
    throw std::runtime_error(label + " must be one symbol");
  }
  const Exponents& powers = value.terms.begin()->first;
  if (std::accumulate(powers.begin(), powers.end(), 0) != 1) {
    throw std::runtime_error(label + " must be one symbol");
  }
  return static_cast<std::size_t>(
      std::distance(powers.begin(), std::ranges::find(powers, 1)));
}

std::pair<std::size_t, std::size_t>
external_pair(const Exponents& powers, const std::set<std::size_t>& external_indices)
{
  std::vector<std::size_t> factors;
  for (const std::size_t index : external_indices) {
    for (int count = 0; count < powers[index]; ++count) {
      factors.push_back(index);
    }
  }
  if (factors.size() != 2) {
    throw std::runtime_error(
        "kinematic invariant contains a non-quadratic momentum term");
  }
  std::ranges::sort(factors);
  return {factors[0], factors[1]};
}

Polynomial replace_external_products_with_dummies(
    const Polynomial& value, const std::set<std::size_t>& external_indices,
    const std::map<std::pair<std::size_t, std::size_t>, std::size_t>& dummy_by_pair)
{
  Polynomial result = zero_poly(value.symbol_count);
  for (const auto& [powers, coefficient] : value.terms) {
    const int external_degree = std::accumulate(
        external_indices.begin(), external_indices.end(), 0,
        [&](int sum, std::size_t index) { return sum + powers[index]; });
    if (external_degree == 0) {
      add_term(result, powers, coefficient);
      continue;
    }
    const auto pair = external_pair(powers, external_indices);
    Exponents mapped = powers;
    --mapped[pair.first];
    --mapped[pair.second];
    ++mapped[dummy_by_pair.at(pair)];
    add_term(result, std::move(mapped), coefficient);
  }
  return result;
}

Polynomial replace_external_products(
    const Polynomial& value, const std::set<std::size_t>& external_indices,
    const std::map<std::pair<std::size_t, std::size_t>, Polynomial>& solutions)
{
  Polynomial result = zero_poly(value.symbol_count);
  for (const auto& [powers, coefficient] : value.terms) {
    const int external_degree = std::accumulate(
        external_indices.begin(), external_indices.end(), 0,
        [&](int sum, std::size_t index) { return sum + powers[index]; });
    if (external_degree == 0) {
      add_term(result, powers, coefficient);
      continue;
    }
    const auto pair = external_pair(powers, external_indices);
    Exponents base_powers = powers;
    --base_powers[pair.first];
    --base_powers[pair.second];
    Polynomial base = zero_poly(value.symbol_count);
    add_term(base, std::move(base_powers), coefficient);
    result = result + base * solutions.at(pair);
  }
  return result;
}

struct LinearEquation {
  std::vector<Rational> coefficients;
  Polynomial rhs;
};

LinearEquation build_linear_equation(const Polynomial& equation,
                                     const std::vector<std::size_t>& variables)
{
  LinearEquation result{std::vector<Rational>(variables.size(), Rational(0)),
                        zero_poly(equation.symbol_count)};
  for (const auto& [powers, coefficient] : equation.terms) {
    std::size_t selected = variables.size();
    for (std::size_t index = 0; index < variables.size(); ++index) {
      const int exponent = powers[variables[index]];
      if (exponent > 1 || (exponent == 1 && selected != variables.size())) {
        throw std::runtime_error(
            "kinematic equations must be linear in scalar products");
      }
      if (exponent == 1) {
        selected = index;
      }
    }
    Exponents rest = powers;
    if (selected != variables.size()) {
      rest[variables[selected]] = 0;
      if (std::accumulate(rest.begin(), rest.end(), 0) != 0) {
        throw std::runtime_error("scalar-product coefficient must be constant");
      }
      result.coefficients[selected] = result.coefficients[selected] + coefficient;
    } else {
      add_term(result.rhs, powers, -coefficient);
    }
  }
  return result;
}

std::vector<Polynomial> solve_linear_system(std::vector<std::vector<Rational>> matrix,
                                            std::vector<Polynomial> rhs)
{
  if (matrix.empty() || matrix.size() != rhs.size()) {
    throw std::runtime_error("no kinematic equations available");
  }
  const std::size_t columns = matrix[0].size();
  std::vector<std::size_t> pivot_row(columns, matrix.size());
  std::size_t row = 0;
  for (std::size_t column = 0; column < columns && row < matrix.size(); ++column) {
    std::size_t pivot = row;
    while (pivot < matrix.size() && matrix[pivot][column].is_zero()) {
      ++pivot;
    }
    if (pivot == matrix.size()) {
      continue;
    }
    std::swap(matrix[pivot], matrix[row]);
    std::swap(rhs[pivot], rhs[row]);
    const Rational inverse = Rational(1) / matrix[row][column];
    for (std::size_t current = column; current < columns; ++current) {
      matrix[row][current] = matrix[row][current] * inverse;
    }
    rhs[row] = scale(rhs[row], inverse);
    for (std::size_t other = 0; other < matrix.size(); ++other) {
      if (other == row || matrix[other][column].is_zero()) {
        continue;
      }
      const Rational factor = matrix[other][column];
      for (std::size_t current = column; current < columns; ++current) {
        matrix[other][current] = matrix[other][current] - factor * matrix[row][current];
      }
      rhs[other] = rhs[other] - scale(rhs[row], factor);
    }
    pivot_row[column] = row++;
  }
  for (std::size_t current = 0; current < matrix.size(); ++current) {
    const bool all_zero = std::ranges::all_of(
        matrix[current], [](const Rational& value) { return value.is_zero(); });
    if (all_zero && !rhs[current].terms.empty()) {
      throw std::runtime_error("kinematic equations are inconsistent");
    }
  }
  if (std::ranges::any_of(pivot_row,
                          [&](std::size_t value) { return value == matrix.size(); })) {
    throw std::runtime_error("kinematic equations are underdetermined");
  }
  std::vector<Polynomial> result;
  result.reserve(columns);
  for (const std::size_t pivot : pivot_row) {
    result.push_back(rhs[pivot]);
  }
  return result;
}

std::size_t rational_rank(std::vector<std::vector<Rational>> matrix)
{
  if (matrix.empty()) return 0;
  const std::size_t columns = matrix.front().size();
  std::size_t row = 0;
  for (std::size_t column = 0; column < columns && row < matrix.size(); ++column) {
    std::size_t pivot = row;
    while (pivot < matrix.size() && matrix[pivot][column].is_zero())
      ++pivot;
    if (pivot == matrix.size()) continue;
    std::swap(matrix[row], matrix[pivot]);
    const Rational inverse = Rational(1) / matrix[row][column];
    for (std::size_t current = column; current < columns; ++current)
      matrix[row][current] = matrix[row][current] * inverse;
    for (std::size_t other = row + 1; other < matrix.size(); ++other) {
      if (matrix[other][column].is_zero()) continue;
      const Rational factor = matrix[other][column];
      for (std::size_t current = column; current < columns; ++current) {
        matrix[other][current] = matrix[other][current] - factor * matrix[row][current];
      }
    }
    ++row;
  }
  return row;
}

void validate_integral_form_basis(const std::vector<Polynomial>& forms,
                                  const std::vector<std::size_t>& loop_indices,
                                  const std::set<std::size_t>& independent_indices)
{
  using Pair = std::pair<std::size_t, std::size_t>;
  std::map<Pair, std::size_t> column_by_pair;
  for (std::size_t left = 0; left < loop_indices.size(); ++left) {
    for (std::size_t right = left; right < loop_indices.size(); ++right) {
      column_by_pair.emplace(Pair{loop_indices[left], loop_indices[right]},
                             column_by_pair.size());
    }
  }
  for (const auto loop : loop_indices) {
    for (const auto external : independent_indices)
      column_by_pair.emplace(Pair{loop, external}, column_by_pair.size());
  }

  std::set<std::size_t> momentum_indices(independent_indices);
  momentum_indices.insert(loop_indices.begin(), loop_indices.end());
  std::set<std::size_t> loop_index_set(loop_indices.begin(), loop_indices.end());
  std::vector<std::vector<Rational>> matrix(
      forms.size(), std::vector<Rational>(column_by_pair.size(), Rational(0)));
  for (std::size_t form = 0; form < forms.size(); ++form) {
    for (const auto& [powers, coefficient] : forms[form].terms) {
      int loop_degree = 0;
      int momentum_degree = 0;
      std::vector<std::size_t> factors;
      for (const auto index : momentum_indices) {
        momentum_degree += powers[index];
        if (loop_index_set.contains(index)) loop_degree += powers[index];
        for (int count = 0; count < powers[index]; ++count)
          factors.push_back(index);
      }
      if (loop_degree == 0) continue;
      if (momentum_degree != 2 || factors.size() != 2) {
        throw std::runtime_error(
            "propagator+ISP forms must be quadratic in loop momenta");
      }
      for (std::size_t index = 0; index < powers.size(); ++index) {
        if (powers[index] != 0 && !momentum_indices.contains(index)) {
          throw std::runtime_error("loop scalar-product coefficients must be constant");
        }
      }
      Pair pair;
      if (loop_index_set.contains(factors[0]) && loop_index_set.contains(factors[1])) {
        pair = std::minmax(factors[0], factors[1]);
      } else if (loop_index_set.contains(factors[0])) {
        pair = {factors[0], factors[1]};
      } else if (loop_index_set.contains(factors[1])) {
        pair = {factors[1], factors[0]};
      } else {
        throw std::logic_error("loop degree disappeared from an integral form");
      }
      const auto column = column_by_pair.find(pair);
      if (column == column_by_pair.end()) {
        throw std::runtime_error(
            "propagator+ISP form contains an unsupported scalar product");
      }
      matrix[form][column->second] = matrix[form][column->second] + coefficient;
    }
  }
  if (rational_rank(std::move(matrix)) != column_by_pair.size()) {
    throw std::runtime_error("propagator+ISP forms are linearly dependent in the loop "
                             "scalar-product basis");
  }
}

} // namespace

namespace compiler::detail {

std::map<std::string, ExactRationalConstant>
compile_yaml_numerics_node(const YAML::Node& root)
{
  const auto numeric_text = numeric_string_map(root);
  SymbolTable symbols;
  for (const auto& [key, value] : numeric_text) {
    collect_symbols(key, symbols);
    collect_symbols(value, symbols);
  }
  std::map<std::string, ExactRationalConstant> result;
  for (const auto& [key, value] : numeric_text) {
    const Polynomial key_polynomial = parse_poly(key, symbols);
    const std::size_t variable = require_single_symbol(key_polynomial, "numerics key");
    const Polynomial value_polynomial = parse_poly(value, symbols);
    if (!is_constant(value_polynomial)) {
      throw std::runtime_error("numerics value must be an exact rational number");
    }
    const std::string& name = symbols.names[variable];
    if (!result.emplace(name, export_rational(constant_value(value_polynomial)))
             .second) {
      throw std::runtime_error("duplicate numerics variable: " + name);
    }
  }
  return result;
}

TopologyConfig compile_yaml_topology_node(const YAML::Node& root)
{
  const YAML::Node kinematics = root["kinematics"];
  if (!kinematics || !kinematics.IsMap()) {
    throw std::runtime_error("missing kinematics mapping");
  }
  const auto loop_names =
      string_sequence(kinematics["loop_momenta"], "loop_momenta", false);
  const auto external_names =
      string_sequence(kinematics["external_momenta"], "external_momenta");
  if (std::ranges::find(loop_names, "d") != loop_names.end() ||
      std::ranges::find(external_names, "d") != external_names.end()) {
    throw std::runtime_error("symbol 'd' is reserved for the spacetime dimension");
  }
  const auto conservation_text = string_map(kinematics, "conservation");
  const auto invariant_text = string_map(kinematics, "invariants");
  const auto numeric_text = numeric_string_map(root);
  const auto propagator_text =
      string_sequence(root["propagators"], "propagators", false);
  const auto top_sector = binary_sequence(root["top_sector"], "top_sector");
  if (top_sector.size() != propagator_text.size()) {
    throw std::runtime_error("top_sector length must match the propagator+ISP list");
  }
  std::vector<std::uint32_t> propagator_slots;
  for (std::size_t slot = 0; slot < top_sector.size(); ++slot) {
    if (top_sector[slot] != 0) {
      propagator_slots.push_back(static_cast<std::uint32_t>(slot));
    }
  }
  if (propagator_slots.empty() || propagator_slots.size() >= 32) {
    throw std::runtime_error("active propagator count must be in [1, 31]");
  }
  const std::string integral_header =
      root["integral_header"] ? root["integral_header"].as<std::string>() : "F";
  if (!valid_integral_header(integral_header)) {
    throw std::runtime_error("integral_header must be a Mathematica symbol");
  }

  SymbolTable symbols;
  for (const auto& name : loop_names) {
    symbols.add(name);
  }
  for (const auto& name : external_names) {
    symbols.add(name);
  }
  for (const auto& [lhs, rhs] : conservation_text) {
    collect_symbols(lhs, symbols);
    collect_symbols(rhs, symbols);
  }
  for (const auto& [lhs, rhs] : invariant_text) {
    collect_symbols(lhs, symbols);
    collect_symbols(rhs, symbols);
  }
  for (const auto& [lhs, rhs] : numeric_text) {
    collect_symbols(lhs, symbols);
    collect_symbols(rhs, symbols);
  }
  for (const auto& propagator : propagator_text) {
    collect_symbols(propagator, symbols);
  }
  for (std::size_t slot = 0; slot < propagator_text.size(); ++slot) {
    symbols.add(std::format("__fr_x{}", slot + 1));
  }
  symbols.add("d");

  std::map<std::size_t, Polynomial> conservation;
  std::set<std::string> dependent_external;
  for (const auto& [lhs, rhs] : conservation_text) {
    const std::size_t variable =
        require_single_symbol(parse_poly(lhs, symbols), "conservation lhs");
    if (std::ranges::find(external_names, symbols.names[variable]) ==
        external_names.end()) {
      throw std::runtime_error("conservation lhs must be listed in external_momenta");
    }
    conservation.emplace(variable, parse_poly(rhs, symbols));
    dependent_external.insert(symbols.names[variable]);
  }

  std::vector<std::string> independent_names;
  for (const auto& name : external_names) {
    if (!dependent_external.contains(name)) {
      independent_names.push_back(name);
    }
  }
  const std::size_t loop_count = loop_names.size();
  const std::size_t independent_external_count = independent_names.size();
  const std::size_t expected_integral_count =
      loop_count * (loop_count + 1) / 2 + loop_count * independent_external_count;
  if (propagator_text.size() != expected_integral_count) {
    throw std::runtime_error(std::format(
        "propagator+ISP count must be L*(L+1)/2+L*E = {}", expected_integral_count));
  }
  std::set<std::size_t> independent_indices;
  for (const auto& name : independent_names) {
    independent_indices.insert(symbols.index(name));
  }

  std::vector<std::pair<std::size_t, std::size_t>> scalar_pairs;
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> dummy_by_pair;
  for (std::size_t left = 0; left < independent_names.size(); ++left) {
    for (std::size_t right = left; right < independent_names.size(); ++right) {
      const std::size_t lhs = symbols.index(independent_names[left]);
      const std::size_t rhs = symbols.index(independent_names[right]);
      const auto pair = std::minmax(lhs, rhs);
      const std::string dummy = std::format("__fr_sp_{}_{}", pair.first, pair.second);
      symbols.add(dummy);
      scalar_pairs.push_back(pair);
      dummy_by_pair.emplace(pair, symbols.index(dummy));
    }
  }

  // The dummy symbols were appended after earlier polynomials could have been
  // parsed, so parse all expressions only after the symbol table is complete.
  std::map<std::size_t, Polynomial> numeric_rules;
  std::map<std::string, ExactRationalConstant> numeric_values;
  for (const auto& [key, value] : numeric_text) {
    const Polynomial key_polynomial = parse_poly(key, symbols);
    const std::size_t variable = require_single_symbol(key_polynomial, "numerics key");
    const Polynomial value_polynomial = parse_poly(value, symbols);
    if (!is_constant(value_polynomial)) {
      throw std::runtime_error("numerics value must be an exact rational number");
    }
    const Rational exact = constant_value(value_polynomial);
    if (!numeric_rules.emplace(variable, value_polynomial).second ||
        !numeric_values.emplace(symbols.names[variable], export_rational(exact))
             .second) {
      throw std::runtime_error("duplicate numerics variable: " +
                               symbols.names[variable]);
    }
  }

  conservation.clear();
  for (const auto& [lhs, rhs] : conservation_text) {
    conservation.emplace(
        require_single_symbol(parse_poly(lhs, symbols), "conservation lhs"),
        substitute(parse_poly(rhs, symbols), numeric_rules));
  }

  std::vector<std::size_t> dummy_indices;
  dummy_indices.reserve(scalar_pairs.size());
  for (const auto& pair : scalar_pairs) {
    dummy_indices.push_back(dummy_by_pair.at(pair));
  }
  std::vector<std::vector<Rational>> scalar_matrix;
  std::vector<Polynomial> scalar_rhs;
  for (const auto& [lhs, rhs] : invariant_text) {
    const Polynomial lhs_expanded = substitute(parse_poly(lhs, symbols), conservation);
    const Polynomial mapped = replace_external_products_with_dummies(
        lhs_expanded, independent_indices, dummy_by_pair);
    const LinearEquation equation = build_linear_equation(
        mapped - substitute(parse_poly(rhs, symbols), numeric_rules), dummy_indices);
    scalar_matrix.push_back(equation.coefficients);
    scalar_rhs.push_back(equation.rhs);
  }
  const auto scalar_solutions_raw =
      solve_linear_system(std::move(scalar_matrix), std::move(scalar_rhs));
  std::map<std::pair<std::size_t, std::size_t>, Polynomial> scalar_solutions;
  for (std::size_t index = 0; index < scalar_pairs.size(); ++index) {
    scalar_solutions.emplace(scalar_pairs[index], scalar_solutions_raw[index]);
  }

  std::vector<Polynomial> integral_forms;
  integral_forms.reserve(propagator_text.size());
  for (const auto& text : propagator_text) {
    integral_forms.push_back(
        substitute(substitute(parse_poly(text, symbols), conservation), numeric_rules));
  }
  const std::vector<Polynomial>& propagators = integral_forms;
  std::vector<std::size_t> loop_indices;
  loop_indices.reserve(loop_names.size());
  for (const auto& name : loop_names)
    loop_indices.push_back(symbols.index(name));
  validate_integral_form_basis(integral_forms, loop_indices, independent_indices);

  std::set<std::string> excluded(loop_names.begin(), loop_names.end());
  excluded.insert(external_names.begin(), external_names.end());
  std::set<std::string> numeric_parameter_candidates;
  auto collect_raw_parameters = [&](const Polynomial& value) {
    for (const auto& [powers, coefficient] : value.terms) {
      static_cast<void>(coefficient);
      for (std::size_t index = 0; index < powers.size(); ++index) {
        const std::string& name = symbols.names[index];
        if (powers[index] != 0 && !excluded.contains(name) &&
            !name.starts_with("__fr_")) {
          numeric_parameter_candidates.insert(name);
        }
      }
    }
  };
  for (const auto& [lhs, rhs] : invariant_text) {
    static_cast<void>(lhs);
    collect_raw_parameters(parse_poly(rhs, symbols));
  }
  for (const auto& [lhs, rhs] : conservation_text) {
    static_cast<void>(lhs);
    collect_raw_parameters(parse_poly(rhs, symbols));
  }
  for (const auto& text : propagator_text)
    collect_raw_parameters(parse_poly(text, symbols));
  if (numeric_parameter_candidates.contains("d")) {
    throw std::runtime_error("symbol 'd' is reserved for the spacetime dimension");
  }
  for (const auto& [name, value] : numeric_values) {
    static_cast<void>(value);
    if (name != "d" && !numeric_parameter_candidates.contains(name)) {
      throw std::runtime_error("numerics key is not a kinematic parameter: " + name);
    }
  }

  std::set<std::string> parameter_set;
  auto collect_parameters = [&](const Polynomial& value) {
    for (const auto& [powers, coefficient] : value.terms) {
      static_cast<void>(coefficient);
      for (std::size_t index = 0; index < powers.size(); ++index) {
        const std::string& name = symbols.names[index];
        if (powers[index] != 0 && !excluded.contains(name) &&
            !name.starts_with("__fr_")) {
          parameter_set.insert(name);
        }
      }
    }
  };
  for (const auto& [lhs, rhs] : invariant_text) {
    static_cast<void>(lhs);
    collect_parameters(substitute(parse_poly(rhs, symbols), numeric_rules));
  }
  for (const auto& propagator : propagators) {
    collect_parameters(propagator);
  }
  const std::vector<std::string> kinematic_parameters(parameter_set.begin(),
                                                      parameter_set.end());

  Polynomial denominator = zero_poly(symbols.names.size());
  std::vector<std::size_t> x_indices;
  for (std::size_t index = 0; index < propagators.size(); ++index) {
    const std::size_t x_index = symbols.index(std::format("__fr_x{}", index + 1));
    x_indices.push_back(x_index);
    denominator =
        denominator + symbol_poly(symbols.names.size(), x_index) * propagators[index];
  }
  std::map<std::size_t, Polynomial> zero_loops;
  for (const auto index : loop_indices) {
    zero_loops.emplace(index, zero_poly(symbols.names.size()));
  }
  std::vector<std::vector<Polynomial>> matrix(
      loop_indices.size(),
      std::vector<Polynomial>(loop_indices.size(), zero_poly(symbols.names.size())));
  std::vector<Polynomial> vector_q;
  for (std::size_t row = 0; row < loop_indices.size(); ++row) {
    for (std::size_t column = 0; column < loop_indices.size(); ++column) {
      matrix[row][column] = scale(
          derivative(derivative(denominator, loop_indices[row]), loop_indices[column]),
          Rational(1) / Rational(2));
    }
    vector_q.push_back(
        scale(substitute(derivative(denominator, loop_indices[row]), zero_loops),
              Rational(-1) / Rational(2)));
  }
  const Polynomial scalar_j = substitute(denominator, zero_loops);
  const Polynomial polynomial_u = replace_external_products(
      determinant(matrix), independent_indices, scalar_solutions);
  const Polynomial polynomial_f = replace_external_products(
      -polynomial_u * scalar_j + quadratic_form(vector_q, adjugate(matrix)),
      independent_indices, scalar_solutions);
  const Polynomial polynomial_g = polynomial_u + polynomial_f;

  std::vector<std::size_t> parameter_indices;
  parameter_indices.reserve(kinematic_parameters.size());
  for (const auto& name : kinematic_parameters) {
    parameter_indices.push_back(symbols.index(name));
  }
  fmpz_t coefficient_denominator_lcm;
  fmpz_t coefficient_multiplier;
  fmpz_t integer_coefficient;
  fmpz_init_set_ui(coefficient_denominator_lcm, 1);
  fmpz_init(coefficient_multiplier);
  fmpz_init(integer_coefficient);
  for (const auto& [powers, coefficient] : polynomial_g.terms) {
    static_cast<void>(powers);
    fmpz_lcm(coefficient_denominator_lcm, coefficient_denominator_lcm,
             fmpq_denref(coefficient.raw()));
  }

  std::map<std::vector<std::uint8_t>, std::vector<std::int64_t>> compiled;
  for (const auto& [powers, coefficient] : polynomial_g.terms) {
    std::vector<std::uint8_t> x_powers;
    for (const std::size_t x_index : x_indices) {
      if (powers[x_index] < 0 || powers[x_index] > 255) {
        throw std::runtime_error("Symanzik exponent exceeds uint8");
      }
      x_powers.push_back(static_cast<std::uint8_t>(powers[x_index]));
    }
    std::size_t parameter_slot = 0;
    int parameter_degree = 0;
    for (std::size_t index = 0; index < parameter_indices.size(); ++index) {
      const int exponent = powers[parameter_indices[index]];
      parameter_degree += exponent;
      if (exponent != 0) {
        if (exponent != 1) {
          throw std::runtime_error(
              "Symanzik coefficients must be affine in parameters");
        }
        parameter_slot = index + 1;
      }
    }
    if (parameter_degree > 1) {
      throw std::runtime_error("Symanzik coefficients must be affine in parameters");
    }
    std::set<std::size_t> allowed(parameter_indices.begin(), parameter_indices.end());
    allowed.insert(x_indices.begin(), x_indices.end());
    for (std::size_t index = 0; index < powers.size(); ++index) {
      if (powers[index] != 0 && !allowed.contains(index)) {
        throw std::runtime_error("unresolved momentum symbol in Symanzik polynomial: " +
                                 symbols.names[index]);
      }
    }
    auto& weights =
        compiled
            .try_emplace(x_powers,
                         std::vector<std::int64_t>(kinematic_parameters.size() + 1, 0))
            .first->second;
    fmpz_divexact(coefficient_multiplier, coefficient_denominator_lcm,
                  fmpq_denref(coefficient.raw()));
    fmpz_mul(integer_coefficient, fmpq_numref(coefficient.raw()),
             coefficient_multiplier);
    const std::int64_t value = compiler::detail::checked_int64(
        integer_coefficient, "Symanzik coefficient after numerics");
    if ((value > 0 &&
         weights[parameter_slot] > std::numeric_limits<std::int64_t>::max() - value) ||
        (value < 0 &&
         weights[parameter_slot] < std::numeric_limits<std::int64_t>::min() - value)) {
      throw std::runtime_error("Symanzik coefficient overflow");
    }
    weights[parameter_slot] += value;
  }
  fmpz_clear(integer_coefficient);
  fmpz_clear(coefficient_multiplier);
  fmpz_clear(coefficient_denominator_lcm);

  TopologyConfig config;
  config.scale_homogeneous = true;
  config.scale_homogeneity_reason.clear();
  std::set<std::string> scale_candidates;
  const auto check_homogeneity = [&](const Polynomial& polynomial,
                                      unsigned parameter_degree,
                                      unsigned x_degree, bool compact,
                                      const std::string& label) {
    for (const auto& [powers, coefficient] : polynomial.terms) {
      if (coefficient.is_zero()) continue;
      if (compact) {
        bool survives = true;
        for (std::size_t slot = 0; slot < x_indices.size(); ++slot)
          if (top_sector[slot] == 0 && powers[x_indices[slot]] != 0) survives = false;
        if (!survives) continue;
      }
      std::int64_t degree = 0;
      std::int64_t alpha_degree = 0;
      for (const auto index : parameter_indices) degree += powers[index];
      for (const auto index : x_indices) alpha_degree += powers[index];
      if (degree != parameter_degree || alpha_degree != x_degree) {
        config.scale_homogeneous = false;
        if (config.scale_homogeneity_reason.empty())
          config.scale_homogeneity_reason = std::format(
              "{} is not homogeneous after numerics (kinematic degree {}, expected {}; "
              "Feynman-parameter degree {}, expected {})",
              label, degree, parameter_degree, alpha_degree, x_degree);
      }
      if (parameter_degree == 1)
        for (std::size_t index = 0; index < parameter_indices.size(); ++index)
          if (powers[parameter_indices[index]] != 0)
            scale_candidates.insert(kinematic_parameters[index]);
    }
  };
  for (const bool compact : {true, false}) {
    const std::string prefix = compact ? "denominator " : "extended ";
    check_homogeneity(polynomial_u, 0, static_cast<unsigned>(loop_count), compact,
                       prefix + "U");
    check_homogeneity(polynomial_f, 1, static_cast<unsigned>(loop_count + 1), compact,
                       prefix + "F");
  }
  config.reconstruction_scale_candidates.assign(scale_candidates.begin(), scale_candidates.end());
  config.loop_count = static_cast<unsigned>(loop_names.size());
  config.propagator_count = static_cast<unsigned>(propagator_slots.size());
  config.integral_count = static_cast<unsigned>(propagator_text.size());
  config.integral_header = integral_header;
  config.top_sector = top_sector;
  config.propagator_slots = propagator_slots;
  config.kinematic_parameters = kinematic_parameters;
  config.numerics = std::move(numeric_values);
  if (const auto dimension = config.numerics.find("d");
      dimension != config.numerics.end()) {
    config.dimension_value = dimension->second;
  } else {
    config.dimension_parameter_index = 0;
    config.parameters.push_back("d");
  }
  for (const auto& parameter : config.kinematic_parameters) {
    config.kinematic_parameter_indices.push_back(
        static_cast<std::uint32_t>(config.parameters.size()));
    config.parameters.push_back(parameter);
  }
  config.extended_lp.polynomial_terms.reserve(compiled.size());
  for (const auto& [powers, weights] : compiled) {
    config.extended_lp.polynomial_terms.push_back({powers, weights});
  }
  std::ranges::sort(config.extended_lp.polynomial_terms,
                    [](const PolynomialTerm& lhs, const PolynomialTerm& rhs) {
                      return lhs.powers > rhs.powers;
                    });

  for (const auto& term : config.extended_lp.polynomial_terms) {
    bool survives = true;
    for (std::size_t slot = 0; slot < config.integral_count; ++slot) {
      if (config.top_sector[slot] == 0 && term.powers[slot] != 0) {
        survives = false;
        break;
      }
    }
    if (!survives) continue;
    std::vector<std::uint8_t> powers;
    powers.reserve(config.propagator_count);
    for (const auto slot : config.propagator_slots)
      powers.push_back(term.powers[slot]);
    config.polynomial_terms.push_back({std::move(powers), term.weights});
  }
  if (config.polynomial_terms.empty()) {
    throw std::runtime_error("denominator LP polynomial is empty");
  }
  {
    const YAML::Node backend_node = root["symmetry_backend"];
    if (backend_node && !backend_node.IsScalar())
      throw std::runtime_error("symmetry_backend must be 'none', 'nauty', or 'bliss'");
    const std::string backend = backend_node ? backend_node.as<std::string>() : "none";
    SymmetryAnalysis analysis;
    if (backend == "none") {
      analysis.backend = SymmetryBackend::None;
    } else if (backend == "nauty") {
      analysis.backend = SymmetryBackend::Nauty;
    } else if (backend == "bliss") {
      analysis.backend = SymmetryBackend::Bliss;
    } else {
      throw std::runtime_error("symmetry_backend must be 'none', 'nauty', or 'bliss'");
    }
    if (!symmetry::backend_available(analysis.backend)) {
      throw std::runtime_error(
          std::format("requested symmetry backend '{}' is not available in this build "
                      "(available: {})",
                      symmetry::backend_name(analysis.backend),
                      symmetry::available_backends_string()));
    }
    analysis.generators = symmetry::find_lp_symmetry_generators(
        config.polynomial_terms, config.propagator_count, analysis.backend);
    auto subsectors = symmetry::find_subsector_symmetry_classes(
        config, analysis.backend, analysis.generators);
    analysis.nonzero_sector_count = subsectors.nonzero_sector_count;
    analysis.sector_classes = std::move(subsectors.classes);
    auto extended_generators = symmetry::find_lp_symmetry_generators(
        config.extended_lp.polynomial_terms, config.integral_count, analysis.backend,
        config.top_sector);
    std::erase_if(extended_generators, [&](const VariablePermutation& permutation) {
      for (std::size_t slot = 0; slot < config.integral_count; ++slot) {
        if (config.top_sector[slot] != config.top_sector[permutation[slot]])
          return true;
      }
      return false;
    });
    config.extended_lp.symmetry_generators = std::move(extended_generators);
    config.symmetry = std::move(analysis);
  }
  return config;
}

} // namespace compiler::detail
