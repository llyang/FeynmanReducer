#include "reduction/DifferentialEquations.hpp"

#include "reduction/RequestedOutputBlackBox.hpp"
#include "topology/IntegralLayout.hpp"

#include <gmpxx.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace reduction::detail {
namespace {

std::uint32_t intern_target(Config& config, const Integral& integral)
{
  const auto found = std::ranges::find(config.targets, integral);
  if (found != config.targets.end())
    return static_cast<std::uint32_t>(found - config.targets.begin());
  if (config.targets.size() >= std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("too many distinct differential-equation targets");
  config.targets.push_back(integral);
  return static_cast<std::uint32_t>(config.targets.size() - 1);
}

std::string rational_text(const mpq_class& value)
{
  return value.get_den() == 1
             ? value.get_num().get_str()
             : value.get_num().get_str() + "/" + value.get_den().get_str();
}

std::int64_t derivative_scale_offset(const Integral& integral)
{
  std::int64_t sum = 0;
  for (const int index : integral.indices) {
    if (__builtin_add_overflow(sum, static_cast<std::int64_t>(index), &sum))
      throw std::overflow_error("differentiated master index sum overflow");
  }
  std::int64_t result = 0;
  if (__builtin_sub_overflow(std::int64_t{-1}, sum, &result))
    throw std::overflow_error("differential-equation scale degree overflow");
  return result;
}

void ensure_explicit_requests(Config& config)
{
  if (!config.reduction_requests.empty() || config.targets.empty()) return;
  config.reduction_requests = resolved_reduction_requests(config);
}

std::vector<ReductionRequestTerm>
derivative_terms(Config& config, const Integral& master, std::size_t parameter,
                 const mpq_class& inverse_polynomial_scale)
{
  integral_layout::validate(config, master);
  if (master.dimension_shift != 0 ||
      std::ranges::any_of(master.indices, [](int index) { return index < 0; })) {
    throw std::invalid_argument(
        "differential equations require a nonnegative d-dimensional master basis");
  }
  std::map<std::uint32_t, mpq_class> combined;
  for (const auto& polynomial_term : config.polynomial_terms) {
    if (polynomial_term.powers.size() != config.propagator_count ||
        polynomial_term.weights.size() != config.kinematic_parameters.size() + 1) {
      throw std::logic_error("compiled LP polynomial metadata is inconsistent");
    }
    const std::int64_t weight = polynomial_term.weights.at(parameter + 1);
    if (weight == 0) continue;

    Integral shifted = master;
    shifted.dimension_shift = 2;
    mpz_class rising = 1;
    for (std::size_t variable = 0; variable < polynomial_term.powers.size();
         ++variable) {
      const std::size_t slot = config.propagator_slots.at(variable);
      const int index = master.indices.at(slot);
      const unsigned power = polynomial_term.powers[variable];
      for (unsigned step = 0; step < power; ++step)
        rising *= static_cast<long>(index) + static_cast<long>(step);
      if (rising == 0) break;
      if (power > static_cast<unsigned>(std::numeric_limits<int>::max() -
                                        shifted.indices.at(slot))) {
        throw std::overflow_error("differential-equation target index exceeds int");
      }
      shifted.indices[slot] += static_cast<int>(power);
    }
    if (rising == 0) continue;

    // For |b|=L+1 the LP normalization ratio cancels the explicit -d/2:
    //   -d/2 * N(a,d)/N(a+b,d+2)
    //     = (-1)^L product_i (a_i)_{b_i}.
    // The compiled coefficient is w=S*dg/ds, hence the remaining 1/S.
    mpq_class coefficient(weight);
    coefficient *= rising;
    coefficient *= inverse_polynomial_scale;
    if ((config.loop_count & 1U) != 0) coefficient = -coefficient;
    coefficient.canonicalize();
    if (coefficient == 0) continue;
    combined[intern_target(config, shifted)] += coefficient;
  }

  std::vector<ReductionRequestTerm> result;
  result.reserve(combined.size());
  for (auto& [target, coefficient] : combined) {
    coefficient.canonicalize();
    if (coefficient != 0) result.push_back({target, rational_text(coefficient)});
  }
  return result;
}

} // namespace

DifferentialEquationSourcePlan prepare_differential_equation_sources(
    Config& config, std::span<const Integral> differentiated_integrals)
{
  if (!config.differential_equations)
    throw std::invalid_argument("differential-equation generation is not enabled");
  if (differentiated_integrals.empty())
    throw std::invalid_argument("differential equations require a nonempty basis");
  if (config.kinematic_parameters.empty())
    throw std::invalid_argument(
        "differential equations require a free kinematic parameter");
  ensure_explicit_requests(config);

  mpq_class polynomial_scale(mpz_class(config.lp_polynomial_scale.numerator),
                             mpz_class(config.lp_polynomial_scale.denominator));
  polynomial_scale.canonicalize();
  if (polynomial_scale == 0)
    throw std::logic_error("compiled LP polynomial scale is zero");
  const mpq_class inverse_polynomial_scale = 1 / polynomial_scale;

  DifferentialEquationSourcePlan result;
  result.differentiated_integrals.assign(differentiated_integrals.begin(),
                                         differentiated_integrals.end());
  result.parameters = config.kinematic_parameters;
  result.terms.reserve(result.parameters.size() *
                       result.differentiated_integrals.size());
  for (std::size_t parameter = 0; parameter < result.parameters.size(); ++parameter) {
    for (const auto& integral : result.differentiated_integrals) {
      result.terms.push_back(
          derivative_terms(config, integral, parameter, inverse_polynomial_scale));
    }
  }

  // Keep the native reduction shape nonempty even when every requested
  // derivative vanishes identically on its sector boundary.
  if (config.targets.empty()) intern_target(config, result.differentiated_integrals[0]);
  return result;
}

void append_differential_equation_requests(Config& config,
                                           const DifferentialEquationSourcePlan& source,
                                           std::span<const Integral> final_basis)
{
  if (std::ranges::any_of(config.reduction_requests,
                          [](const ReductionRequest& request) {
                            return request.output.differential;
                          })) {
    throw std::logic_error("differential-equation requests were materialized twice");
  }
  const std::size_t expected =
      source.parameters.size() * source.differentiated_integrals.size();
  if (source.terms.size() != expected)
    throw std::logic_error("differential-equation source plan has the wrong shape");

  for (std::size_t parameter = 0; parameter < source.parameters.size(); ++parameter) {
    for (const auto& master : final_basis) {
      const auto found = std::ranges::find(source.differentiated_integrals, master);
      if (found == source.differentiated_integrals.end())
        throw std::logic_error("final master is absent from differential source plan");
      const std::size_t integral =
          static_cast<std::size_t>(found - source.differentiated_integrals.begin());
      ReductionRequest request;
      request.output.integral = master;
      request.output.scale_offset = derivative_scale_offset(master);
      request.output.differential = true;
      request.output.differential_parameter = source.parameters[parameter];
      request.terms =
          source.terms[parameter * source.differentiated_integrals.size() + integral];
      config.reduction_requests.push_back(std::move(request));
    }
  }
}

void materialize_differential_equations(Config& config)
{
  if (!config.differential_equations) return;
  const auto source = prepare_differential_equation_sources(config, config.basis);
  append_differential_equation_requests(config, source, config.basis);
}

} // namespace reduction::detail
