#include "validation/ReductionValidator.hpp"
#include "validation/detail/ReductionParser.hpp"
#include <flint/fmpz_mpoly_q.h>
#include <memory>
#include <set>
#include <utility>

namespace validation {
using namespace detail;

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
