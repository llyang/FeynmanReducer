#pragma once

#include "core/FlintRational.hpp"
#include "validation/ReductionValidator.hpp"
#include <memory>
#include <optional>
#include <set>
#include <string_view>

namespace validation::detail {

using IntegralKey = std::vector<int>;

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

struct RawLinearTerm {
  std::optional<IntegralKey> master;
  std::string coefficient;
};

RawReduction parse_raw_reduction(const std::filesystem::path& path);
RawBasis parse_raw_basis(const std::filesystem::path& path);
std::string integral_name(const std::string& header, const IntegralKey& indices);
std::vector<RawLinearTerm> linear_terms(std::string_view rhs,
                                        const std::string& target);
void collect_parameter_names(std::string_view expression, std::set<std::string>& names);
std::string
substitute_numerics(std::string_view expression,
                    const std::map<std::string, ExactRationalConstant>& numerics);
ParsedReduction
parse_coefficients(const RawReduction& raw,
                   const std::shared_ptr<const FlintRationalContext>& context);
void add_difference(ValidationReport& report, std::size_t limit,
                    std::string diagnostic);
const FlintRational* coefficient(const ParsedRule& rule, const IntegralKey& integral);
bool rational_equal(const FlintRational* lhs, const FlintRational* rhs,
                    const FlintRationalContext& context);
bool is_identity_rule(const ParsedRule& rule, const IntegralKey& target,
                      const FlintRationalContext& context);
std::string rational_text(const FlintRational* value);

} // namespace validation::detail
