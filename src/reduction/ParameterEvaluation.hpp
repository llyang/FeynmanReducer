#pragma once

#include "core/Config.hpp"

#include <firefly/FFInt.hpp>
#include <firefly/ReconstHelper.hpp>

#include <gmpxx.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace reduction::detail {

[[nodiscard]] inline std::size_t
coefficient_parameter_count(const TopologyConfig& config) noexcept
{
  return config.kinematic_parameters.size() + 1;
}

inline void validate_parameter_values(const TopologyConfig& config,
                                      std::span<const firefly::FFInt> values)
{
  if (values.size() != config.parameters.size()) {
    throw std::invalid_argument("finite-field parameter vector has the wrong size");
  }
}

[[nodiscard]] inline firefly::FFInt
evaluate_exact_rational(const ExactRationalConstant& value)
{
  const firefly::FFInt numerator(mpz_class(value.numerator));
  const firefly::FFInt denominator(mpz_class(value.denominator));
  if (denominator == firefly::FFInt(0)) {
    throw std::runtime_error("exact rational denominator vanishes in the finite field");
  }
  return numerator / denominator;
}

[[nodiscard]] inline bool
rational_is_usable_at_prime(const ExactRationalConstant& value, std::uint64_t prime)
{
  const mpz_class denominator(value.denominator);
  return mpz_divisible_ui_p(denominator.get_mpz_t(), prime) == 0;
}

[[nodiscard]] inline bool configuration_is_usable_at_prime(const TopologyConfig& config,
                                                           std::uint64_t prime)
{
  return !config.dimension_value.has_value() ||
         rational_is_usable_at_prime(*config.dimension_value, prime);
}

[[nodiscard]] inline std::vector<std::uint64_t>
usable_firefly_primes(const TopologyConfig& config, std::size_t count)
{
  std::vector<std::uint64_t> result;
  result.reserve(count);
  for (const std::uint64_t prime : firefly::primes()) {
    if (configuration_is_usable_at_prime(config, prime)) result.push_back(prime);
    if (result.size() == count) break;
  }
  if (result.size() != count) {
    throw std::runtime_error("not enough FireFly primes are usable for fixed numerics");
  }
  return result;
}

[[nodiscard]] inline firefly::FFInt
evaluate_dimension(const TopologyConfig& config, std::span<const firefly::FFInt> values)
{
  validate_parameter_values(config, values);
  if (config.dimension_value.has_value()) {
    return evaluate_exact_rational(*config.dimension_value);
  }
  if (!config.dimension_parameter_index.has_value() ||
      *config.dimension_parameter_index >= values.size()) {
    throw std::logic_error("free dimension parameter index is unavailable");
  }
  return values[*config.dimension_parameter_index];
}

[[nodiscard]] inline firefly::FFInt
evaluate_kinematic_parameter(const TopologyConfig& config,
                             std::span<const firefly::FFInt> values,
                             std::size_t parameter)
{
  validate_parameter_values(config, values);
  if (parameter >= config.kinematic_parameter_indices.size()) {
    throw std::out_of_range("kinematic parameter index is unavailable");
  }
  const std::size_t runtime = config.kinematic_parameter_indices[parameter];
  if (runtime >= values.size()) {
    throw std::logic_error("kinematic runtime parameter index is invalid");
  }
  return values[runtime];
}

[[nodiscard]] inline firefly::FFInt
evaluate_polynomial_coefficient(const TopologyConfig& config,
                                const PolynomialTerm& term,
                                std::span<const firefly::FFInt> values)
{
  if (term.weights.size() != coefficient_parameter_count(config)) {
    throw std::logic_error("compiled LP coefficient has the wrong parameter shape");
  }
  firefly::FFInt result(term.weights.front());
  for (std::size_t parameter = 0; parameter < config.kinematic_parameters.size();
       ++parameter) {
    result = result + firefly::FFInt(term.weights[parameter + 1]) *
                          evaluate_kinematic_parameter(config, values, parameter);
  }
  return result;
}

} // namespace reduction::detail
