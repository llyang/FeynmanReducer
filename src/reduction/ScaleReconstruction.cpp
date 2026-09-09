#include "reduction/ScaleReconstruction.hpp"

#include "core/ProbeValues.hpp"
#include "core/detail/FactorizedRationalBuilder.hpp"
#include "reduction/ParameterEvaluation.hpp"

#include <flint/fmpz_mpoly.h>

#include <algorithm>
#include <format>
#include <limits>
#include <stdexcept>

namespace reduction::detail {
namespace {
std::int64_t checked_sum(std::int64_t a, std::int64_t b)
{
  std::int64_t result;
  if (__builtin_add_overflow(a, b, &result))
    throw std::overflow_error("reconstruction scale exponent overflow");
  return result;
}
std::int64_t degree(const std::vector<ulong>& powers,
                    const std::vector<std::string>& names)
{
  std::int64_t result = 0;
  for (std::size_t i = 0; i < powers.size(); ++i) {
    if (names[i] == "d") continue;
    if (powers[i] > static_cast<ulong>(std::numeric_limits<std::int64_t>::max()))
      throw std::overflow_error("reconstruction scale degree overflow");
    result = checked_sum(result, static_cast<std::int64_t>(powers[i]));
  }
  return result;
}
std::int64_t max_degree(const fmpz_mpoly_struct* polynomial,
                        const FlintRationalContext& context)
{
  std::vector<ulong> powers(context.variable_names().size());
  std::int64_t result = 0;
  for (slong i = 0; i < fmpz_mpoly_length(polynomial, context.raw()); ++i) {
    fmpz_mpoly_get_term_exp_ui(powers.data(), polynomial, i, context.raw());
    result = std::max(result, degree(powers, context.variable_names()));
  }
  return result;
}
void lift_polynomial(fmpz_mpoly_struct* dest, const fmpz_mpoly_struct* source,
                     const FlintRationalContext& from, const FlintRationalContext& to,
                     std::size_t scale, std::int64_t maximum)
{
  fmpz_mpoly_zero(dest, to.raw());
  std::vector<ulong> powers(from.variable_names().size());
  std::vector<ulong> lifted(to.variable_names().size());
  fmpz_t coefficient;
  fmpz_init(coefficient);
  try {
    for (slong i = 0; i < fmpz_mpoly_length(source, from.raw()); ++i) {
      fmpz_mpoly_get_term_exp_ui(powers.data(), source, i, from.raw());
      for (std::size_t j = 0; j < powers.size(); ++j)
        lifted[j < scale ? j : j + 1] = powers[j];
      lifted[scale] =
          static_cast<ulong>(maximum - degree(powers, from.variable_names()));
      fmpz_mpoly_get_term_coeff_fmpz(coefficient, source, i, from.raw());
      fmpz_mpoly_push_term_fmpz_ui(dest, coefficient, lifted.data(), to.raw());
    }
    fmpz_mpoly_sort_terms(dest, to.raw());
    fmpz_mpoly_combine_like_terms(dest, to.raw());
  } catch (...) {
    fmpz_clear(coefficient);
    throw;
  }
  fmpz_clear(coefficient);
}
} // namespace

std::size_t select_reconstruction_scale(const Config& config,
                                        std::span<const std::uint32_t> degrees)
{
  if (degrees.size() != config.parameters.size())
    throw std::logic_error("scale scan returned the wrong number of degrees");
  std::optional<std::size_t> best;
  for (const auto& name : config.reconstruction_scale_candidates) {
    const auto it = std::ranges::find(config.parameters, name);
    if (name == "d" || it == config.parameters.end())
      throw std::logic_error("invalid free reconstruction scale candidate");
    const auto index = static_cast<std::size_t>(it - config.parameters.begin());
    if (!best || degrees[index] > degrees[*best] ||
        (degrees[index] == degrees[*best] && name < config.parameters[*best]))
      best = index;
  }
  if (!best) throw std::logic_error("no reconstruction scale candidate");
  return *best;
}

FactorizedRational restore_reconstruction_scale(
    const FactorizedRational& reduced,
    const std::shared_ptr<const FlintRationalContext>& full_context, std::size_t scale,
    const Integral& target, const Integral& master)
{
  auto names = full_context->variable_names();
  if (scale >= names.size() || names[scale] == "d")
    throw std::logic_error("invalid scale restoration variable");
  names.erase(names.begin() + static_cast<std::ptrdiff_t>(scale));
  if (names != reduced.context()->variable_names() ||
      target.indices.size() != master.indices.size())
    throw std::logic_error("factorized scale restoration mapping mismatch");
  if (reduced.is_zero()) return FactorizedRational(full_context);
  factorized_detail::Builder result(full_context);
  std::int64_t power = 0;
  for (std::size_t i = 0; i < target.indices.size(); ++i)
    power = checked_sum(power, static_cast<std::int64_t>(master.indices[i]) -
                                   target.indices[i]);
  const auto lift = [&](const FlintRational& polynomial) {
    const auto maximum =
        max_degree(fmpz_mpoly_q_numref(polynomial.raw()), *reduced.context());
    FlintRational lifted(full_context);
    lift_polynomial(fmpz_mpoly_q_numref(lifted.raw()),
                    fmpz_mpoly_q_numref(polynomial.raw()), *reduced.context(),
                    *full_context, scale, maximum);
    return std::pair{std::move(lifted), maximum};
  };
  auto [numerator, degree] = lift(reduced.numerator());
  power = checked_sum(power, -degree);
  result.set_numerator(std::move(numerator));
  FlintRational scalar(full_context);
  fmpz_t value;
  fmpz_init(value);
  fmpz_mpoly_get_fmpz(value, fmpz_mpoly_q_numref(reduced.scalar().raw()),
                      reduced.context()->raw());
  fmpz_mpoly_set_fmpz(fmpz_mpoly_q_numref(scalar.raw()), value, full_context->raw());
  fmpz_mpoly_get_fmpz(value, fmpz_mpoly_q_denref(reduced.scalar().raw()),
                      reduced.context()->raw());
  fmpz_mpoly_set_fmpz(fmpz_mpoly_q_denref(scalar.raw()), value, full_context->raw());
  fmpz_clear(value);
  result.set_scalar(std::move(scalar));
  for (const auto& factor : reduced.factors()) {
    auto [polynomial, factor_degree] = lift(factor.polynomial);
    std::int64_t weighted;
    if (__builtin_mul_overflow(factor.power, factor_degree, &weighted) ||
        weighted == std::numeric_limits<std::int64_t>::min())
      throw std::overflow_error("factor scale degree overflow");
    power = checked_sum(power, -weighted);
    result.add_irreducible_factor(std::move(polynomial), factor.power);
  }
  if (power == std::numeric_limits<std::int64_t>::min())
    throw std::overflow_error("factor scale exponent overflow");
  if (power) {
    FlintRational scale_factor(full_context);
    fmpz_mpoly_gen(fmpz_mpoly_q_numref(scale_factor.raw()), static_cast<slong>(scale),
                   full_context->raw());
    result.add_irreducible_factor(std::move(scale_factor), power);
  }
  return std::move(result).finish();
}

void validate_restored_scale(const Config& config, const ReductionResult& result,
                             std::span<const std::uint32_t> outputs,
                             const std::function<void()>& prime_changed,
                             const std::function<std::vector<firefly::FFInt>(
                                 const std::vector<firefly::FFInt>&)>& evaluate)
{
  for (const auto prime : usable_firefly_primes(config, 2)) {
    firefly::FFInt::set_new_prime(prime);
    prime_changed();
    unsigned valid = 0;
    for (std::size_t attempt = 0; attempt < 16 && valid < 2; ++attempt) {
      std::vector<ulong> coordinates;
      std::vector<firefly::FFInt> values;
      for (std::size_t j = 0; j < config.parameters.size(); ++j) {
        const auto value =
            probe_values::field_value(prime, j, 0x4000000000000000ULL + attempt);
        coordinates.push_back(value);
        values.emplace_back(value);
      }
      const auto expected_values = evaluate(values);
      if (expected_values.size() != outputs.size())
        throw std::logic_error("scale verification output count mismatch");
      bool regular = true;
      for (std::size_t j = 0; j < outputs.size(); ++j) {
        const auto& coefficient = result.coefficients.at(outputs[j]);
        const auto value = coefficient.evaluate_mod(coordinates, prime);
        if (!value) {
          regular = false;
          break;
        }
        if (*value != expected_values[j].n)
          throw std::runtime_error(std::format(
              "restored scale verification failed for output {}", outputs[j]));
      }
      if (regular) ++valid;
    }
    if (valid != 2)
      throw std::runtime_error("not enough regular scale verification points");
  }
}
} // namespace reduction::detail
