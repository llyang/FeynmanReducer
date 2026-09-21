#include "compiler/detail/CombinationParser.hpp"

#include "compiler/detail/IntegralParser.hpp"
#include "core/FlintRational.hpp"
#include "topology/IntegralLayout.hpp"

#include <yaml-cpp/yaml.h>

#include <flint/fmpz_mpoly.h>
#include <flint/fmpz_mpoly_q.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace compiler::detail {
namespace {

bool identifier_start(char value)
{
  return std::isalpha(static_cast<unsigned char>(value)) != 0 || value == '$';
}

bool symbol_continue(char value)
{
  return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '$';
}

bool coefficient_identifier_continue(char value)
{
  return std::isalnum(static_cast<unsigned char>(value)) != 0 || value == '$' ||
         value == '_';
}

std::string canonical_label(std::string_view input)
{
  std::size_t position = 0;
  const auto skip_space = [&] {
    while (position < input.size() &&
           std::isspace(static_cast<unsigned char>(input[position])) != 0)
      ++position;
  };
  skip_space();
  if (position == input.size() || !identifier_start(input[position]))
    throw std::runtime_error("combination name must start with a Mathematica symbol");
  std::string result;
  result.push_back(input[position++]);
  while (position < input.size() && symbol_continue(input[position]))
    result.push_back(input[position++]);
  skip_space();
  if (position == input.size()) return result;
  if (input[position++] != '[')
    throw std::runtime_error(
        "combination name must be a symbol or a symbol with integer arguments");
  result.push_back('[');
  bool first = true;
  while (true) {
    skip_space();
    if (position == input.size())
      throw std::runtime_error("unclosed combination name argument list");
    if (input[position] == ']') {
      if (first)
        throw std::runtime_error("combination name argument list must not be empty");
      result.push_back(']');
      ++position;
      skip_space();
      if (position != input.size())
        throw std::runtime_error("unexpected text after combination name");
      return result;
    }
    if (!first) {
      if (input[position++] != ',')
        throw std::runtime_error("combination name arguments must be integers");
      result.push_back(',');
      skip_space();
    }
    const std::size_t begin = position;
    if (input[position] == '-') ++position;
    const std::size_t digits = position;
    while (position < input.size() &&
           std::isdigit(static_cast<unsigned char>(input[position])) != 0)
      ++position;
    if (digits == position)
      throw std::runtime_error("combination name arguments must be integers");
    result.append(input.substr(begin, position - begin));
    skip_space();
    first = false;
  }
}

void require_mapping_fields(const YAML::Node& node,
                            std::initializer_list<std::string_view> allowed,
                            std::string_view label)
{
  if (!node.IsMap())
    throw std::runtime_error(std::string(label) + " must be a mapping");
  for (const auto& entry : node) {
    if (!entry.first.IsScalar())
      throw std::runtime_error(std::string(label) + " field names must be strings");
    const std::string field = entry.first.as<std::string>();
    if (std::ranges::find(allowed, field) == allowed.end())
      throw std::runtime_error(std::format("unknown {} field: {}", label, field));
  }
}

std::string substitute_numerics(std::string_view expression, const Config& config)
{
  std::set<std::string> free(config.parameters.begin(), config.parameters.end());
  std::string result;
  for (std::size_t position = 0; position < expression.size();) {
    if (!identifier_start(expression[position])) {
      result.push_back(expression[position++]);
      continue;
    }
    const std::size_t begin = position++;
    while (position < expression.size() &&
           coefficient_identifier_continue(expression[position]))
      ++position;
    const std::string name(expression.substr(begin, position - begin));
    if (const auto fixed = config.numerics.find(name); fixed != config.numerics.end()) {
      result += "(" + fixed->second.numerator;
      if (fixed->second.denominator != "1") result += "/" + fixed->second.denominator;
      result += ')';
    } else if (free.contains(name)) {
      result += name;
    } else {
      throw std::runtime_error("unknown parameter in combination coefficient: " + name);
    }
  }
  return result;
}

FlintRational
parse_coefficient(const std::shared_ptr<const FlintRationalContext>& context,
                  std::string_view expression, const Config& config)
{
  const std::string substituted = substitute_numerics(expression, config);
  std::vector<const char*> names;
  names.reserve(context->variable_names().size());
  for (const auto& name : context->variable_names())
    names.push_back(name.c_str());
  FlintRational result(context);
  if (fmpz_mpoly_q_set_str_pretty(result.raw(), substituted.c_str(), names.data(),
                                  const_cast<fmpz_mpoly_ctx_struct*>(context->raw())) !=
      0)
    throw std::runtime_error("cannot parse combination coefficient: " +
                             std::string(expression));
  result.canonicalise();
  return result;
}

std::optional<std::int64_t>
homogeneous_degree(const fmpz_mpoly_struct* polynomial,
                   const FlintRationalContext& context,
                   const std::vector<std::uint32_t>& kinematic_indices)
{
  if (fmpz_mpoly_is_zero(polynomial, context.raw())) return std::int64_t{0};
  std::vector<ulong> powers(context.variable_names().size());
  std::optional<std::int64_t> expected;
  for (slong term = 0; term < fmpz_mpoly_length(polynomial, context.raw()); ++term) {
    fmpz_mpoly_get_term_exp_ui(powers.data(), polynomial, term, context.raw());
    std::uint64_t degree = 0;
    for (const auto index : kinematic_indices) {
      if (index >= powers.size())
        throw std::logic_error("kinematic parameter index is out of range");
      if (degree >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) -
              powers[index])
        throw std::overflow_error("combination coefficient degree overflow");
      degree += powers[index];
    }
    const auto signed_degree = static_cast<std::int64_t>(degree);
    if (expected && *expected != signed_degree) return std::nullopt;
    expected = signed_degree;
  }
  return expected.value_or(0);
}

std::optional<std::int64_t> rational_degree(const FlintRational& value,
                                            const Config& config)
{
  const auto numerator =
      homogeneous_degree(fmpz_mpoly_q_numref(value.raw()), *value.context(),
                         config.kinematic_parameter_indices);
  const auto denominator =
      homogeneous_degree(fmpz_mpoly_q_denref(value.raw()), *value.context(),
                         config.kinematic_parameter_indices);
  if (!numerator || !denominator) return std::nullopt;
  std::int64_t result;
  if (__builtin_sub_overflow(*numerator, *denominator, &result))
    throw std::overflow_error("combination coefficient degree overflow");
  return result;
}

std::int64_t index_sum(const Integral& integral)
{
  std::int64_t result = 0;
  for (const int index : integral.indices) {
    if (__builtin_add_overflow(result, static_cast<std::int64_t>(index), &result))
      throw std::overflow_error("integral index sum overflow");
  }
  return result;
}

Integral parse_integral_node(const YAML::Node& node, const Config& config)
{
  if (!node.IsScalar())
    throw std::runtime_error("combination integral must be a headed scalar expression");
  try {
    Integral result =
        parse_integral_expression(node.as<std::string>(), config.integral_count);
    integral_layout::validate(config, result);
    return result;
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("invalid combination integral: ") +
                             error.what());
  }
}

} // namespace

std::vector<ParsedCombination> parse_combinations(const std::filesystem::path& path,
                                                  const Config& config)
{
  const YAML::Node root = YAML::LoadFile(path.string());
  require_mapping_fields(root, {"combinations"}, "combination file");
  const YAML::Node combinations = root["combinations"];
  if (!combinations || !combinations.IsSequence() || combinations.size() == 0)
    throw std::runtime_error("combination file requires a non-empty combinations list");

  auto context = std::make_shared<FlintRationalContext>(config.parameters);
  std::set<std::string> names_seen;
  std::vector<ParsedCombination> result;
  result.reserve(combinations.size());
  for (std::size_t combination_index = 0; combination_index < combinations.size();
       ++combination_index) {
    const auto node = combinations[combination_index];
    require_mapping_fields(node, {"name", "terms"}, "combination");
    if (!node["name"] || !node["name"].IsScalar())
      throw std::runtime_error("combination requires a scalar name");
    const std::string name = canonical_label(node["name"].as<std::string>());
    const auto bracket = name.find('[');
    const std::string head = name.substr(0, bracket);
    if (head == config.integral_header)
      throw std::runtime_error("combination name must not use integral_header");
    if (!names_seen.insert(name).second)
      throw std::runtime_error("duplicate combination name: " + name);
    const auto terms = node["terms"];
    if (!terms || !terms.IsSequence() || terms.size() == 0)
      throw std::runtime_error("combination requires a non-empty terms list");

    struct AccumulatedTerm {
      Integral integral;
      FlintRational coefficient;
    };
    std::vector<AccumulatedTerm> accumulated;
    for (std::size_t term_index = 0; term_index < terms.size(); ++term_index) {
      const auto term = terms[term_index];
      require_mapping_fields(term, {"coefficient", "integral"}, "combination term");
      if (!term["coefficient"] || !term["coefficient"].IsScalar())
        throw std::runtime_error("combination term requires a scalar coefficient");
      if (!term["integral"])
        throw std::runtime_error("combination term requires integral indices");
      Integral integral = parse_integral_node(term["integral"], config);
      FlintRational coefficient =
          parse_coefficient(context, term["coefficient"].as<std::string>(), config);
      const auto found = std::ranges::find_if(accumulated, [&](const auto& current) {
        return current.integral == integral;
      });
      if (found == accumulated.end()) {
        accumulated.push_back({std::move(integral), std::move(coefficient)});
      } else {
        fmpz_mpoly_q_add(found->coefficient.raw(), found->coefficient.raw(),
                         coefficient.raw(), context->raw());
        found->coefficient.canonicalise();
      }
    }
    std::erase_if(accumulated,
                  [](const auto& term) { return term.coefficient.is_zero(); });
    if (accumulated.empty())
      throw std::runtime_error("combination is identically zero after merging: " +
                               name);

    ParsedCombination parsed;
    parsed.output.named = true;
    parsed.output.name = name;
    std::optional<std::int64_t> common_scale_offset;
    for (auto& term : accumulated) {
      if (config.scale_homogeneous) {
        const auto degree = rational_degree(term.coefficient, config);
        if (!degree)
          throw std::runtime_error("combination coefficient is not homogeneous: " +
                                   name);
        std::int64_t offset;
        const auto sum = index_sum(term.integral);
        std::int64_t shifted_degree = 0;
        const auto half_shift = term.integral.dimension_shift / 2;
        if (__builtin_mul_overflow(static_cast<std::int64_t>(config.loop_count),
                                   static_cast<std::int64_t>(half_shift),
                                   &shifted_degree) ||
            __builtin_add_overflow(*degree, shifted_degree, &shifted_degree) ||
            __builtin_sub_overflow(shifted_degree, sum, &offset))
          throw std::overflow_error("combination scale degree overflow");
        if (common_scale_offset && *common_scale_offset != offset)
          throw std::runtime_error("combination terms have inconsistent scale: " +
                                   name);
        common_scale_offset = offset;
      }
      parsed.terms.push_back({std::move(term.integral), term.coefficient.to_string()});
    }
    parsed.output.scale_offset = common_scale_offset.value_or(0);
    result.push_back(std::move(parsed));
  }
  return result;
}

} // namespace compiler::detail
