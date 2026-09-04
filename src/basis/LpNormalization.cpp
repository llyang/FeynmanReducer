#include "LpNormalization.hpp"

#include <stdexcept>

namespace basis {
namespace {

std::uint64_t factorial(const PrimeField& field, unsigned argument)
{
  std::uint64_t result = 1;
  for (unsigned value = 2; value <= argument; ++value)
    result = field.multiply(result, value);
  return result;
}

std::uint64_t integer_power(const PrimeField& field, std::uint64_t value, int exponent)
{
  if (exponent >= 0) return field.power(value, static_cast<std::uint64_t>(exponent));
  return field.divide(1, field.power(value, static_cast<std::uint64_t>(-exponent)));
}

} // namespace

LpNormalizationData lp_normalization_data(const Config& config,
                                          const Integral& integral)
{
  if (integral.indices.size() != config.integral_count)
    throw std::invalid_argument("integral size does not match the topology");

  std::vector<bool> active(config.integral_count, false);
  for (const auto slot : config.propagator_slots) {
    if (slot >= integral.indices.size())
      throw std::invalid_argument("propagator slot is outside the integral");
    active[slot] = true;
  }

  LpNormalizationData result;
  for (std::size_t slot = 0; slot < integral.indices.size(); ++slot) {
    const int index = integral.indices[slot];
    if (!active[slot]) {
      if (index != 0)
        throw std::invalid_argument(
            "LP normalization currently supports denominator-only integrals");
      continue;
    }
    if (index < 0)
      throw std::invalid_argument(
          "LP normalization does not support negative propagator indices");
    if (index == 0)
      ++result.pinch_count;
    else
      result.positive_sum += static_cast<unsigned>(index);
  }
  return result;
}

std::uint64_t lp_normalization_and_sign_ratio(const PrimeField& field,
                                              const Config& config,
                                              const Integral& target,
                                              const Integral& basis,
                                              std::uint64_t dimension)
{
  const auto target_data = lp_normalization_data(config, target);
  const auto basis_data = lp_normalization_data(config, basis);
  const auto lambda = field.divide(dimension, 2);

  std::uint64_t result = integer_power(field, lambda,
                                       static_cast<int>(basis_data.pinch_count) -
                                           static_cast<int>(target_data.pinch_count));

  // Gamma((L+1) lambda - A_target) /
  // Gamma((L+1) lambda - A_basis).
  const auto scaled_lambda =
      field.multiply(static_cast<std::uint64_t>(config.loop_count + 1), lambda);
  if (basis_data.positive_sum >= target_data.positive_sum) {
    const auto start = field.subtract(scaled_lambda, basis_data.positive_sum);
    const auto count = basis_data.positive_sum - target_data.positive_sum;
    for (unsigned offset = 0; offset < count; ++offset)
      result = field.multiply(result, field.add(start, offset));
  } else {
    const auto start = field.subtract(scaled_lambda, target_data.positive_sum);
    const auto count = target_data.positive_sum - basis_data.positive_sum;
    for (unsigned offset = 0; offset < count; ++offset)
      result = field.divide(result, field.add(start, offset));
  }

  for (const auto slot : config.propagator_slots) {
    const int target_index = target.indices[slot];
    const int basis_index = basis.indices[slot];
    if (target_index > 0)
      result = field.multiply(
          result, factorial(field, static_cast<unsigned>(target_index - 1)));
    if (basis_index > 0)
      result = field.divide(result,
                            factorial(field, static_cast<unsigned>(basis_index - 1)));
  }

  if (((target_data.positive_sum + basis_data.positive_sum) & 1U) != 0)
    result = field.subtract(0, result);
  return result;
}

} // namespace basis
