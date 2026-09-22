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
  if (descriptor.differential) {
    return "D[" +
           format_mathematica_integral(result.integral_header, descriptor.integral) +
           "," + descriptor.differential_parameter + "]";
  }
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
  std::size_t standalone_outputs = 0;
  for (std::size_t output = 0; output < output_count; ++output) {
    if (!result.outputs.empty()) {
      const auto& descriptor = result.outputs[output];
      if (descriptor.named || descriptor.differential) continue;
    }
    ++standalone_outputs;
    for (std::size_t master = 0; master < result.basis.size(); ++master) {
      const std::size_t coefficient = output * result.basis.size() + master;
      const auto factor = result.coefficients[coefficient].mixed_denominator_factor();
      if (!factor) continue;
      ++result.d_separation.failed_coefficients;
      if (result.d_separation.witness.empty())
        result.d_separation.witness = std::format(
            "target={}, master={}, mixed factor=({})", output_name(result, output),
            format_mathematica_integral(result.integral_header,
                                        result.basis[master].indices),
            result.coefficients[coefficient]
                .factors()
                .at(*factor)
                .polynomial.to_string());
    }
  }
  if (standalone_outputs == 0) {
    if (policy == BasisSelectionPolicy::DSeparating) {
      result.d_separation.status = DSeparationStatus::Passed;
      if (progress)
        progress("D-separation source-integral validation passed during basis "
                 "selection; no standalone integral outputs to recheck",
                 ReductionProgressEvent::info);
    } else {
      result.d_separation.status = DSeparationStatus::Skipped;
      if (progress)
        progress("D-separation check skipped: no standalone integral outputs",
                 ReductionProgressEvent::info);
    }
    return;
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
    if (progress) {
      progress(policy == BasisSelectionPolicy::DSeparating
                   ? "D-separation check passed for standalone integral outputs; "
                     "source integrals were validated during basis selection"
                   : "D-separation check passed for standalone integral outputs "
                     "and configured numerics",
               ReductionProgressEvent::info);
    }
  }
}
} // namespace reduction::detail
