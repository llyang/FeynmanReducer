#include "reduction/DSeparationCheck.hpp"
#include "core/IntegralFormatting.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace reduction::detail {
namespace {
std::string output_name(const ReductionResult& result, std::size_t output)
{
  if (result.outputs.empty())
    return format_mathematica_integral(result.integral_header,
                                       result.targets.at(output));
  const auto& descriptor = result.outputs.at(output);
  return descriptor.named
             ? descriptor.name
             : format_mathematica_integral(result.integral_header, descriptor.integral);
}
} // namespace

void check_d_separation(ReductionResult& result, BasisSelectionPolicy policy,
                        const ReductionProgressCallback& progress)
{
  result.d_separation = {};
  if (std::ranges::find(result.parameters, "d") == result.parameters.end()) {
    result.d_separation.status = DSeparationStatus::Skipped;
    if (progress)
      progress("D-separation check skipped: d is not a free parameter",
               ReductionProgressEvent::info);
    return;
  }
  const std::size_t output_count =
      result.outputs.empty() ? result.targets.size() : result.outputs.size();
  if (result.coefficients.size() != output_count * result.basis.size())
    throw std::logic_error("D-separation result shape mismatch");
  for (std::size_t i = 0; i < result.coefficients.size(); ++i) {
    const auto factor = result.coefficients[i].mixed_denominator_factor();
    if (!factor) continue;
    ++result.d_separation.failed_coefficients;
    if (result.d_separation.witness.empty())
      result.d_separation.witness = std::format(
          "target={}, master={}, mixed factor=({})",
          output_name(result, i / result.basis.size()),
          format_mathematica_integral(result.integral_header,
                                      result.basis[i % result.basis.size()].indices),
          result.coefficients[i].factors().at(*factor).polynomial.to_string());
  }
  if (result.d_separation.failed_coefficients) {
    result.d_separation.status = DSeparationStatus::Failed;
    const auto message = std::format(
        "Reconstructed coefficients are not d-separating: {} coefficient(s); {}",
        result.d_separation.failed_coefficients, result.d_separation.witness);
    if (policy == BasisSelectionPolicy::DSeparating) throw std::runtime_error(message);
    if (progress) progress(message, ReductionProgressEvent::warning);
  } else {
    result.d_separation.status = DSeparationStatus::Passed;
    if (progress)
      progress("D-separation check passed for current targets and configured numerics",
               ReductionProgressEvent::info);
  }
}
} // namespace reduction::detail
