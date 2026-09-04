#include "validation/ReductionValidator.hpp"

#include "core/FlintRational.hpp"

#include <flint/fmpz_mpoly.h>
#include <flint/fmpz_mpoly_q.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace validation {
namespace {

using IntegralKey = std::vector<int>;

struct IntegralReference {
  std::string header;
  IntegralKey indices;
  std::size_t end = 0;
};

struct RawRule {
  IntegralKey target;
  std::string rhs;
};

struct RawReduction {
  std::string header;
  std::vector<RawRule> rules;
};

struct RawBasis {
  std::string header;
  std::set<IntegralKey> integrals;
};

struct ParsedRule {
  IntegralKey target;
  std::map<IntegralKey, FlintRational> coefficients;
};

struct ParsedReduction {
  std::string header;
  std::map<IntegralKey, ParsedRule> rules;
};

std::string read_file(const std::filesystem::path& path)
{
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open validation input: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string trim(std::string_view text)
{
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
    text.remove_prefix(1);
  }
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
    text.remove_suffix(1);
  }
  return std::string(text);
}

std::string strip_comments(std::string_view text)
{
  std::string result;
  result.reserve(text.size());
  for (std::size_t index = 0; index < text.size();) {
    if (text[index] == '#') {
      while (index < text.size() && text[index] != '\n') {
        ++index;
      }
      result.push_back(' ');
      continue;
    }
    if (index + 1 < text.size() && text[index] == '(' && text[index + 1] == '*') {
      std::size_t depth = 1;
      index += 2;
      while (index < text.size() && depth != 0) {
        if (index + 1 < text.size() && text[index] == '(' && text[index + 1] == '*') {
          ++depth;
          index += 2;
        } else if (index + 1 < text.size() && text[index] == '*' &&
                   text[index + 1] == ')') {
          --depth;
          index += 2;
        } else {
          ++index;
        }
      }
      if (depth != 0) {
        throw std::runtime_error("unclosed Mathematica comment");
      }
      result.push_back(' ');
      continue;
    }
    result.push_back(text[index++]);
  }
  return result;
}

bool identifier_start(char character)
{
  return std::isalpha(static_cast<unsigned char>(character)) != 0 || character == '$';
}

bool identifier_continue(char character)
{
  return std::isalnum(static_cast<unsigned char>(character)) != 0 || character == '$' ||
         character == '_';
}

void skip_space(std::string_view text, std::size_t& position)
{
  while (position < text.size() &&
         std::isspace(static_cast<unsigned char>(text[position]))) {
    ++position;
  }
}

std::optional<IntegralReference> integral_at(std::string_view text, std::size_t start)
{
  if (start >= text.size() || !identifier_start(text[start])) {
    return std::nullopt;
  }
  std::size_t position = start + 1;
  while (position < text.size() && identifier_continue(text[position])) {
    ++position;
  }
  const std::string header(text.substr(start, position - start));
  skip_space(text, position);
  if (position >= text.size() || text[position] != '[') {
    return std::nullopt;
  }
  ++position;
  IntegralKey indices;
  while (true) {
    skip_space(text, position);
    if (position >= text.size()) {
      throw std::runtime_error("unclosed integral index list");
    }
    if (text[position] == ']') {
      if (indices.empty()) {
        throw std::runtime_error("integral index list must not be empty");
      }
      return IntegralReference{header, std::move(indices), position + 1};
    }
    const std::size_t number_start = position;
    if (text[position] == '+' || text[position] == '-') {
      ++position;
    }
    const std::size_t digits = position;
    while (position < text.size() &&
           std::isdigit(static_cast<unsigned char>(text[position]))) {
      ++position;
    }
    if (digits == position) {
      throw std::runtime_error("integral index must be an integer");
    }
    int value = 0;
    const char* parse_begin = text.data() + number_start;
    if (*parse_begin == '+') {
      ++parse_begin;
    }
    const auto parsed = std::from_chars(parse_begin, text.data() + position, value);
    if (parsed.ec != std::errc()) {
      throw std::runtime_error("integral index is outside the int range");
    }
    indices.push_back(value);
    skip_space(text, position);
    if (position >= text.size()) {
      throw std::runtime_error("unclosed integral index list");
    }
    if (text[position] == ',') {
      ++position;
      continue;
    }
    if (text[position] != ']') {
      throw std::runtime_error("expected ',' or ']' in integral index list");
    }
  }
}

IntegralReference parse_integral(std::string_view text)
{
  std::size_t start = 0;
  skip_space(text, start);
  const auto integral = integral_at(text, start);
  if (!integral) {
    throw std::runtime_error("expected an integral");
  }
  std::size_t end = integral->end;
  skip_space(text, end);
  if (end != text.size()) {
    throw std::runtime_error("unexpected text after integral");
  }
  return *integral;
}

void merge_header(std::string& header, const std::string& candidate)
{
  if (header.empty()) {
    header = candidate;
  } else if (header != candidate) {
    throw std::runtime_error(
        "one validation file contains multiple integral headers: " + header + " and " +
        candidate);
  }
}

std::vector<std::string> split_top_level(std::string_view text, char delimiter)
{
  std::vector<std::string> result;
  int parentheses = 0;
  int brackets = 0;
  int braces = 0;
  std::size_t start = 0;
  for (std::size_t index = 0; index < text.size(); ++index) {
    switch (text[index]) {
    case '(':
      ++parentheses;
      break;
    case ')':
      if (--parentheses < 0) {
        throw std::runtime_error("unmatched ')' in reduction file");
      }
      break;
    case '[':
      ++brackets;
      break;
    case ']':
      if (--brackets < 0) {
        throw std::runtime_error("unmatched ']' in reduction file");
      }
      break;
    case '{':
      ++braces;
      break;
    case '}':
      if (--braces < 0) {
        throw std::runtime_error("unmatched '}' in reduction file");
      }
      break;
    default:
      break;
    }
    if (text[index] == delimiter && parentheses == 0 && brackets == 0 && braces == 0) {
      result.push_back(trim(text.substr(start, index - start)));
      start = index + 1;
    }
  }
  if (parentheses != 0 || brackets != 0 || braces != 0) {
    throw std::runtime_error("unclosed delimiter in reduction file");
  }
  result.push_back(trim(text.substr(start)));
  return result;
}

std::size_t top_level_arrow(std::string_view text)
{
  int parentheses = 0;
  int brackets = 0;
  int braces = 0;
  for (std::size_t index = 0; index + 1 < text.size(); ++index) {
    switch (text[index]) {
    case '(':
      ++parentheses;
      break;
    case ')':
      --parentheses;
      break;
    case '[':
      ++brackets;
      break;
    case ']':
      --brackets;
      break;
    case '{':
      ++braces;
      break;
    case '}':
      --braces;
      break;
    default:
      break;
    }
    if (text[index] == '-' && text[index + 1] == '>' && parentheses == 0 &&
        brackets == 0 && braces == 0) {
      return index;
    }
  }
  throw std::runtime_error("reduction entry is missing '->'");
}

RawReduction parse_raw_reduction(const std::filesystem::path& path)
{
  std::string text = trim(strip_comments(read_file(path)));
  if (!text.empty() && text.back() == ';') {
    text.pop_back();
    text = trim(text);
  }
  if (text.size() < 2 || text.front() != '{' || text.back() != '}') {
    throw std::runtime_error(
        "reduction file must contain a Mathematica replacement list: " + path.string());
  }
  text = text.substr(1, text.size() - 2);
  RawReduction result;
  std::set<IntegralKey> targets;
  for (const auto& entry : split_top_level(text, ',')) {
    if (entry.empty()) {
      continue;
    }
    const std::size_t arrow = top_level_arrow(entry);
    const IntegralReference target =
        parse_integral(std::string_view(entry).substr(0, arrow));
    merge_header(result.header, target.header);
    if (!targets.insert(target.indices).second) {
      throw std::runtime_error("duplicate target integral in " + path.string());
    }
    const std::string rhs = trim(std::string_view(entry).substr(arrow + 2));
    if (rhs.empty()) {
      throw std::runtime_error("empty reduction right-hand side");
    }
    for (std::size_t position = 0; position < rhs.size();) {
      const auto integral = integral_at(rhs, position);
      if (integral) {
        merge_header(result.header, integral->header);
        position = integral->end;
      } else {
        ++position;
      }
    }
    result.rules.push_back({target.indices, rhs});
  }
  if (result.rules.empty()) {
    throw std::runtime_error("reduction file contains no rules: " + path.string());
  }
  return result;
}

RawBasis parse_raw_basis(const std::filesystem::path& path)
{
  const std::string text = strip_comments(read_file(path));
  RawBasis result;
  for (std::size_t position = 0; position < text.size();) {
    const auto integral = integral_at(text, position);
    if (integral) {
      merge_header(result.header, integral->header);
      if (!result.integrals.insert(integral->indices).second) {
        throw std::runtime_error("duplicate master integral in " + path.string());
      }
      position = integral->end;
      continue;
    }
    const char character = text[position++];
    if (!std::isspace(static_cast<unsigned char>(character)) && character != ',' &&
        character != '{' && character != '}' && character != ';') {
      throw std::runtime_error("unexpected text in master file: " + path.string());
    }
  }
  if (result.integrals.empty()) {
    throw std::runtime_error("master file contains no integrals: " + path.string());
  }
  return result;
}

std::string integral_name(const std::string& header, const IntegralKey& indices)
{
  std::ostringstream output;
  output << header << '[';
  for (std::size_t index = 0; index < indices.size(); ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << indices[index];
  }
  output << ']';
  return output.str();
}

struct RawLinearTerm {
  std::optional<IntegralKey> master;
  std::string coefficient;
};

std::vector<std::string> split_additive_terms(std::string_view text)
{
  std::vector<std::string> result;
  int parentheses = 0;
  int brackets = 0;
  std::size_t start = 0;
  for (std::size_t index = 0; index < text.size(); ++index) {
    if (text[index] == '(') {
      ++parentheses;
    } else if (text[index] == ')') {
      --parentheses;
    } else if (text[index] == '[') {
      ++brackets;
    } else if (text[index] == ']') {
      --brackets;
    }
    if ((text[index] != '+' && text[index] != '-') || parentheses != 0 ||
        brackets != 0 || index == start) {
      continue;
    }
    std::size_t previous = index;
    while (previous > start &&
           std::isspace(static_cast<unsigned char>(text[previous - 1]))) {
      --previous;
    }
    if (previous == start ||
        std::string_view("+-*/^(").find(text[previous - 1]) != std::string_view::npos) {
      continue;
    }
    result.push_back(trim(text.substr(start, index - start)));
    start = index;
  }
  result.push_back(trim(text.substr(start)));
  return result;
}

std::vector<RawLinearTerm> linear_terms(std::string_view rhs,
                                        const std::string& target_name)
{
  std::vector<RawLinearTerm> result;
  for (const auto& additive : split_additive_terms(rhs)) {
    if (additive.empty()) {
      throw std::runtime_error("empty term in reduction for " + target_name);
    }
    std::string scalar;
    scalar.reserve(additive.size());
    std::optional<IntegralKey> master;
    int parentheses = 0;
    for (std::size_t position = 0; position < additive.size();) {
      const auto integral = integral_at(additive, position);
      if (integral) {
        if (parentheses != 0 || master.has_value()) {
          throw std::runtime_error(
              "reduction right-hand side is not linear in master integrals for " +
              target_name);
        }
        std::size_t previous = position;
        while (previous > 0 &&
               std::isspace(static_cast<unsigned char>(additive[previous - 1]))) {
          --previous;
        }
        if (previous > 0 && additive[previous - 1] == '/') {
          throw std::runtime_error(
              "an integral appears in a reduction denominator for " + target_name);
        }
        std::size_t next = integral->end;
        while (next < additive.size() &&
               std::isspace(static_cast<unsigned char>(additive[next]))) {
          ++next;
        }
        if (next < additive.size() && additive[next] == '^') {
          throw std::runtime_error(
              "reduction right-hand side is not linear in master integrals for " +
              target_name);
        }
        master = integral->indices;
        scalar += "(1)";
        position = integral->end;
        continue;
      }
      if (additive[position] == '(') {
        ++parentheses;
      } else if (additive[position] == ')') {
        --parentheses;
      }
      scalar.push_back(additive[position++]);
    }
    scalar = trim(scalar);
    if (!scalar.empty() && scalar.front() == '+') {
      scalar = trim(std::string_view(scalar).substr(1));
    }
    if (scalar.empty()) {
      throw std::runtime_error("empty coefficient in reduction for " + target_name);
    }
    result.push_back({std::move(master), std::move(scalar)});
  }
  return result;
}

void collect_parameter_names(std::string_view expression,
                             std::set<std::string>& parameters)
{
  for (std::size_t position = 0; position < expression.size();) {
    const char character = expression[position];
    if (std::isalpha(static_cast<unsigned char>(character)) == 0 && character != '_') {
      ++position;
      continue;
    }
    const std::size_t start = position++;
    while (position < expression.size() &&
           (std::isalnum(static_cast<unsigned char>(expression[position])) != 0 ||
            expression[position] == '_')) {
      ++position;
    }
    parameters.emplace(expression.substr(start, position - start));
  }
}

std::string
substitute_numerics(std::string_view expression,
                    const std::map<std::string, ExactRationalConstant>& numerics)
{
  std::string result;
  result.reserve(expression.size());
  for (std::size_t position = 0; position < expression.size();) {
    if (!identifier_start(expression[position])) {
      result.push_back(expression[position++]);
      continue;
    }
    const std::size_t start = position++;
    while (position < expression.size() && identifier_continue(expression[position]))
      ++position;
    const std::string name(expression.substr(start, position - start));
    const auto found = numerics.find(name);
    if (found == numerics.end()) {
      result += name;
      continue;
    }
    result += "(" + found->second.numerator;
    if (found->second.denominator != "1") result += "/" + found->second.denominator;
    result += ")";
  }
  return result;
}

ParsedReduction
parse_coefficients(const RawReduction& raw,
                   const std::shared_ptr<const FlintRationalContext>& context)
{
  std::vector<const char*> names;
  names.reserve(context->variable_names().size());
  for (const auto& name : context->variable_names()) {
    names.push_back(name.c_str());
  }
  auto* mutable_context = const_cast<fmpz_mpoly_ctx_struct*>(context->raw());
  ParsedReduction result;
  result.header = raw.header;
  for (const auto& raw_rule : raw.rules) {
    ParsedRule rule;
    rule.target = raw_rule.target;
    const std::string target_name = integral_name(raw.header, raw_rule.target);
    for (const auto& term : linear_terms(raw_rule.rhs, target_name)) {
      FlintRational parsed(context);
      if (fmpz_mpoly_q_set_str_pretty(parsed.raw(), term.coefficient.c_str(),
                                      names.data(), mutable_context) != 0) {
        throw std::runtime_error("FLINT cannot parse a coefficient for " + target_name +
                                 ": " + term.coefficient);
      }
      parsed.canonicalise();
      if (!term.master.has_value()) {
        if (!parsed.is_zero()) {
          throw std::runtime_error(
              "reduction right-hand side contains a non-integral term for " +
              target_name);
        }
        continue;
      }
      auto found = rule.coefficients.find(*term.master);
      if (found == rule.coefficients.end()) {
        if (!parsed.is_zero()) {
          rule.coefficients.emplace(*term.master, std::move(parsed));
        }
      } else {
        fmpz_mpoly_q_add(found->second.raw(), found->second.raw(), parsed.raw(),
                         context->raw());
        found->second.canonicalise();
      }
    }
    std::erase_if(rule.coefficients,
                  [](const auto& entry) { return entry.second.is_zero(); });
    result.rules.emplace(rule.target, std::move(rule));
  }
  return result;
}

void add_difference(ValidationReport& report, std::size_t limit, std::string diagnostic)
{
  ++report.difference_count;
  if (report.diagnostics.size() < limit) {
    report.diagnostics.push_back(std::move(diagnostic));
  }
}

const FlintRational* coefficient(const ParsedRule& rule, const IntegralKey& integral)
{
  const auto found = rule.coefficients.find(integral);
  return found == rule.coefficients.end() ? nullptr : &found->second;
}

bool rational_equal(const FlintRational* lhs, const FlintRational* rhs,
                    const FlintRationalContext& context)
{
  if (lhs == nullptr) {
    return rhs == nullptr || rhs->is_zero();
  }
  if (rhs == nullptr) {
    return lhs->is_zero();
  }
  return fmpz_mpoly_q_equal(lhs->raw(), rhs->raw(), context.raw()) != 0;
}

bool is_identity_rule(const ParsedRule& rule, const IntegralKey& target,
                      const FlintRationalContext& context)
{
  if (rule.coefficients.size() != 1) {
    return false;
  }
  const auto coefficient = rule.coefficients.find(target);
  return coefficient != rule.coefficients.end() &&
         fmpz_mpoly_q_is_one(coefficient->second.raw(), context.raw()) != 0;
}

std::string rational_text(const FlintRational* value)
{
  return value == nullptr ? "0" : value->to_string();
}

} // namespace

ValidationReport validate_reduction(const ValidationFiles& files,
                                    std::size_t diagnostic_limit)
{
  const RawReduction produced = parse_raw_reduction(files.result);
  RawReduction reference = parse_raw_reduction(files.kira_result);
  for (auto& rule : reference.rules)
    rule.rhs = substitute_numerics(rule.rhs, files.numerics);
  const RawBasis produced_basis = parse_raw_basis(files.basis);
  const RawBasis reference_basis = parse_raw_basis(files.kira_basis);

  ValidationReport report;
  report.target_count = produced.rules.size();
  report.master_count = produced_basis.integrals.size();
  const auto check_header = [&](const std::string& label, const std::string& header) {
    if (header != produced.header) {
      add_difference(report, diagnostic_limit,
                     label + " uses integral header '" + header + "', expected '" +
                         produced.header + "'");
    }
  };
  check_header("program basis", produced_basis.header);
  check_header("Kira result", reference.header);
  check_header("Kira basis", reference_basis.header);

  for (const auto& integral : produced_basis.integrals) {
    if (!reference_basis.integrals.contains(integral)) {
      add_difference(report, diagnostic_limit,
                     "master only in program output: " +
                         integral_name(produced.header, integral));
    }
  }
  for (const auto& integral : reference_basis.integrals) {
    if (!produced_basis.integrals.contains(integral)) {
      add_difference(report, diagnostic_limit,
                     "master only in Kira output: " +
                         integral_name(reference.header, integral));
    }
  }

  std::set<std::string> parameters;
  auto collect_reduction_parameters = [&](const RawReduction& reduction) {
    for (const auto& rule : reduction.rules) {
      const std::string target_name = integral_name(reduction.header, rule.target);
      for (const auto& term : linear_terms(rule.rhs, target_name)) {
        collect_parameter_names(term.coefficient, parameters);
      }
    }
  };
  collect_reduction_parameters(produced);
  collect_reduction_parameters(reference);

  std::vector<std::string> variable_names(parameters.begin(), parameters.end());
  const auto context =
      std::make_shared<FlintRationalContext>(std::move(variable_names));
  const ParsedReduction parsed_produced = parse_coefficients(produced, context);
  const ParsedReduction parsed_reference = parse_coefficients(reference, context);

  for (const auto& [target, rule] : parsed_produced.rules) {
    if (!parsed_reference.rules.contains(target)) {
      if (produced_basis.integrals.contains(target) &&
          is_identity_rule(rule, target, *context)) {
        ++report.skipped_identity_count;
        continue;
      }
      add_difference(report, diagnostic_limit,
                     "target only in program output: " +
                         integral_name(produced.header, target));
    }
  }
  for (const auto& [target, rule] : parsed_reference.rules) {
    static_cast<void>(rule);
    if (!parsed_produced.rules.contains(target)) {
      add_difference(report, diagnostic_limit,
                     "target only in Kira output: " +
                         integral_name(reference.header, target));
    }
  }

  for (const auto& [target, produced_rule] : parsed_produced.rules) {
    const auto reference_rule = parsed_reference.rules.find(target);
    if (reference_rule == parsed_reference.rules.end()) {
      continue;
    }
    for (const auto& master : produced_basis.integrals) {
      ++report.coefficient_count;
      const FlintRational* produced_value = coefficient(produced_rule, master);
      const FlintRational* reference_value =
          coefficient(reference_rule->second, master);
      if (rational_equal(produced_value, reference_value, *context)) {
        continue;
      }
      FlintRational difference(context);
      if (produced_value != nullptr && reference_value != nullptr) {
        fmpz_mpoly_q_sub(difference.raw(), produced_value->raw(),
                         reference_value->raw(), context->raw());
      } else if (produced_value != nullptr) {
        fmpz_mpoly_q_set(difference.raw(), produced_value->raw(), context->raw());
      } else {
        fmpz_mpoly_q_neg(difference.raw(), reference_value->raw(), context->raw());
      }
      difference.canonicalise();
      add_difference(report, diagnostic_limit,
                     "coefficient mismatch for target " +
                         integral_name(produced.header, target) + ", master " +
                         integral_name(produced.header, master) +
                         "\n    program: " + rational_text(produced_value) +
                         "\n    Kira:    " + rational_text(reference_value) +
                         "\n    delta:   " + difference.to_string());
    }
  }

  const auto check_rhs_basis = [&](const ParsedReduction& reduction,
                                   const RawBasis& basis, const std::string& label) {
    for (const auto& [target, rule] : reduction.rules) {
      for (const auto& [master, value] : rule.coefficients) {
        static_cast<void>(value);
        if (!basis.integrals.contains(master)) {
          add_difference(report, diagnostic_limit,
                         label + " reduction for " +
                             integral_name(reduction.header, target) +
                             " references undeclared master " +
                             integral_name(reduction.header, master));
        }
      }
    }
  };
  check_rhs_basis(parsed_produced, produced_basis, "program");
  check_rhs_basis(parsed_reference, reference_basis, "Kira");
  return report;
}

} // namespace validation
