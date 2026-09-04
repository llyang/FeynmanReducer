#include "UnivariateSeparation.hpp"

#include <firefly/FFInt.hpp>
#include <firefly/FFThieleInterpolator.hpp>

#include <algorithm>
#include <array>
#include <map>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace basis {
namespace {

using ThieleSampleOrders = std::array<std::vector<std::size_t>, 4>;

const ThieleSampleOrders& thiele_sample_orders(std::size_t training)
{
  // D-separating interpolation runs many functions with the same adaptive
  // sample count. Keep deterministic permutations once per worker instead of
  // allocating four index vectors for every coefficient.
  thread_local std::map<std::size_t, ThieleSampleOrders> cache;
  const auto found = cache.find(training);
  if (found != cache.end()) return found->second;

  ThieleSampleOrders orders;
  for (auto& order : orders)
    order.resize(training);
  std::iota(orders[0].begin(), orders[0].end(), 0);
  orders[1] = orders[0];
  std::ranges::reverse(orders[1]);
  for (std::size_t index = 0; index < training; ++index)
    orders[2][index] = index % 2 == 0 ? index / 2 : training - 1 - index / 2;
  auto stride = std::size_t{17};
  while (std::gcd(stride, training) != 1)
    stride += 2;
  for (std::size_t index = 0; index < training; ++index)
    orders[3][index] = (5 + stride * index) % training;
  return cache.emplace(training, std::move(orders)).first->second;
}

void trim_polynomial(FieldVector& polynomial)
{
  while (polynomial.size() > 1 && polynomial.back() == 0)
    polynomial.pop_back();
  if (polynomial.empty()) polynomial.push_back(0);
}

FieldVector powers(const PrimeField& field, std::uint64_t value, std::size_t degree)
{
  FieldVector result(degree + 1, 1);
  for (std::size_t index = 1; index <= degree; ++index)
    result[index] = field.multiply(result[index - 1], value);
  return result;
}

FieldVector monic_polynomial(const PrimeField& field, FieldVector polynomial)
{
  trim_polynomial(polynomial);
  if (polynomial.size() == 1 && polynomial.front() == 0) return polynomial;
  const auto scale = field.inverse(polynomial.back());
  for (auto& coefficient : polynomial)
    coefficient = field.multiply(coefficient, scale);
  return polynomial;
}

FieldVector polynomial_remainder(const PrimeField& field, FieldVector dividend,
                                 const FieldVector& divisor)
{
  trim_polynomial(dividend);
  if (divisor.size() == 1 && divisor.front() == 0)
    throw std::invalid_argument("polynomial division by zero");
  while (!(dividend.size() == 1 && dividend.front() == 0) &&
         dividend.size() >= divisor.size()) {
    const auto degree = dividend.size() - divisor.size();
    const auto factor = field.divide(dividend.back(), divisor.back());
    for (std::size_t index = 0; index < divisor.size(); ++index) {
      dividend[index + degree] = field.subtract(dividend[index + degree],
                                                field.multiply(factor, divisor[index]));
    }
    trim_polynomial(dividend);
  }
  return dividend;
}

FieldVector polynomial_gcd(const PrimeField& field, FieldVector lhs, FieldVector rhs)
{
  trim_polynomial(lhs);
  trim_polynomial(rhs);
  while (!(rhs.size() == 1 && rhs.front() == 0)) {
    auto remainder = polynomial_remainder(field, std::move(lhs), rhs);
    lhs = std::move(rhs);
    rhs = std::move(remainder);
  }
  return monic_polynomial(field, std::move(lhs));
}

FieldVector polynomial_quotient_exact(const PrimeField& field, FieldVector dividend,
                                      const FieldVector& divisor)
{
  trim_polynomial(dividend);
  if (divisor.size() == 1 && divisor.front() == 0)
    throw std::invalid_argument("polynomial division by zero");
  if (dividend.size() < divisor.size())
    throw std::runtime_error("non-exact polynomial quotient");
  FieldVector quotient(dividend.size() - divisor.size() + 1, 0);
  while (!(dividend.size() == 1 && dividend.front() == 0) &&
         dividend.size() >= divisor.size()) {
    const auto degree = dividend.size() - divisor.size();
    const auto factor = field.divide(dividend.back(), divisor.back());
    quotient[degree] = factor;
    for (std::size_t index = 0; index < divisor.size(); ++index)
      dividend[index + degree] = field.subtract(dividend[index + degree],
                                                field.multiply(factor, divisor[index]));
    trim_polynomial(dividend);
  }
  if (!(dividend.size() == 1 && dividend.front() == 0))
    throw std::runtime_error("non-exact polynomial quotient");
  trim_polynomial(quotient);
  return quotient;
}

FieldVector dense_polynomial(const firefly::ff_map& sparse)
{
  std::size_t maximum_degree = 0;
  for (const auto& [degrees, coefficient] : sparse) {
    if (degrees.size() > 1)
      throw std::runtime_error("Thiele returned a multivariate polynomial");
    if (coefficient != firefly::FFInt(0))
      maximum_degree =
          std::max(maximum_degree, degrees.empty() ? std::size_t{0} : degrees.front());
  }
  FieldVector result(maximum_degree + 1, 0);
  for (const auto& [degrees, coefficient] : sparse) {
    const auto degree = degrees.empty() ? std::size_t{0} : degrees.front();
    result[degree] = coefficient.n;
  }
  trim_polynomial(result);
  return result;
}

RationalFunction normalize_rational_function(const PrimeField& field,
                                             RationalFunction function)
{
  trim_polynomial(function.numerator);
  trim_polynomial(function.denominator);
  if (function.denominator.size() == 1 && function.denominator.front() == 0)
    throw std::runtime_error("Thiele returned a zero denominator");
  if (function.numerator.size() == 1 && function.numerator.front() == 0)
    return {.numerator = {0}, .denominator = {1}};
  const auto common = polynomial_gcd(field, function.numerator, function.denominator);
  if (common.size() > 1) {
    function.numerator =
        polynomial_quotient_exact(field, std::move(function.numerator), common);
    function.denominator =
        polynomial_quotient_exact(field, std::move(function.denominator), common);
  }
  const auto scale = field.inverse(function.denominator.back());
  for (auto& coefficient : function.numerator)
    coefficient = field.multiply(coefficient, scale);
  for (auto& coefficient : function.denominator)
    coefficient = field.multiply(coefficient, scale);
  return function;
}

bool validates(const PrimeField& field, const RationalFunction& function,
               std::span<const std::uint64_t> arguments,
               std::span<const std::uint64_t> values)
{
  for (std::size_t sample = 0; sample < arguments.size(); ++sample) {
    try {
      if (evaluate(field, function, arguments[sample]) != values[sample]) return false;
    } catch (const std::domain_error&) {
      return false;
    }
  }
  return true;
}

} // namespace

std::optional<RationalFunction>
interpolate_rational_monic(const PrimeField& field,
                           std::span<const std::uint64_t> arguments,
                           std::span<const std::uint64_t> values,
                           std::size_t maximum_total_degree, std::size_t holdout_count)
{
  if (arguments.size() != values.size() || holdout_count == 0 ||
      arguments.size() <= holdout_count) {
    throw std::invalid_argument("invalid monic rational interpolation samples");
  }
  if (std::ranges::all_of(values,
                          [&](const auto value) { return value == values.front(); })) {
    return RationalFunction{.numerator = {values.front()}, .denominator = {1}};
  }
  const std::size_t training = arguments.size() - holdout_count;
  for (std::size_t total = 0; total <= maximum_total_degree; ++total) {
    for (std::size_t denominator_degree = 0; denominator_degree <= total;
         ++denominator_degree) {
      const std::size_t numerator_degree = total - denominator_degree;
      const std::size_t unknowns = numerator_degree + 1 + denominator_degree;
      if (training < unknowns) continue;
      FieldMatrix matrix(training, FieldVector(unknowns, 0));
      FieldVector rhs(training, 0);
      for (std::size_t sample = 0; sample < training; ++sample) {
        const auto sample_powers = powers(field, arguments[sample], total);
        for (std::size_t degree = 0; degree <= numerator_degree; ++degree)
          matrix[sample][degree] = sample_powers[degree];
        for (std::size_t degree = 0; degree < denominator_degree; ++degree) {
          matrix[sample][numerator_degree + 1 + degree] =
              field.subtract(0, field.multiply(values[sample], sample_powers[degree]));
        }
        rhs[sample] =
            denominator_degree == 0
                ? values[sample]
                : field.multiply(values[sample], sample_powers[denominator_degree]);
      }
      const auto solved = solve_linear_system(field, std::move(matrix), std::move(rhs));
      if (!solved.consistent || !solved.unique) continue;
      RationalFunction candidate;
      candidate.numerator.assign(
          solved.solution.begin(),
          solved.solution.begin() +
              static_cast<FieldVector::difference_type>(numerator_degree + 1));
      candidate.denominator.assign(denominator_degree + 1, 0);
      if (denominator_degree == 0) {
        candidate.denominator[0] = 1;
      } else {
        for (std::size_t degree = 0; degree < denominator_degree; ++degree)
          candidate.denominator[degree] =
              solved.solution[numerator_degree + 1 + degree];
        candidate.denominator.back() = 1;
      }
      trim_polynomial(candidate.numerator);
      trim_polynomial(candidate.denominator);
      bool valid = true;
      for (std::size_t sample = training; sample < arguments.size(); ++sample) {
        try {
          if (evaluate(field, candidate, arguments[sample]) != values[sample]) {
            valid = false;
            break;
          }
        } catch (const std::domain_error&) {
          valid = false;
          break;
        }
      }
      if (valid) return candidate;
    }
  }
  return std::nullopt;
}

std::optional<RationalFunction> interpolate_rational_fixed_degrees(
    const PrimeField& field, std::span<const std::uint64_t> arguments,
    std::span<const std::uint64_t> values, std::size_t numerator_degree,
    std::size_t denominator_degree, std::size_t holdout_count)
{
  if (arguments.size() != values.size() || holdout_count == 0 ||
      arguments.size() <= holdout_count) {
    throw std::invalid_argument("invalid fixed-degree rational interpolation samples");
  }
  const std::size_t training = arguments.size() - holdout_count;
  const std::size_t unknowns = numerator_degree + 1 + denominator_degree;
  if (training < unknowns) return std::nullopt;

  const std::size_t maximum_degree = std::max(numerator_degree, denominator_degree);
  FieldMatrix matrix(training, FieldVector(unknowns, 0));
  FieldVector rhs(training, 0);
  for (std::size_t sample = 0; sample < training; ++sample) {
    const auto sample_powers = powers(field, arguments[sample], maximum_degree);
    for (std::size_t degree = 0; degree <= numerator_degree; ++degree)
      matrix[sample][degree] = sample_powers[degree];
    for (std::size_t degree = 0; degree < denominator_degree; ++degree) {
      matrix[sample][numerator_degree + 1 + degree] =
          field.subtract(0, field.multiply(values[sample], sample_powers[degree]));
    }
    rhs[sample] =
        denominator_degree == 0
            ? values[sample]
            : field.multiply(values[sample], sample_powers[denominator_degree]);
  }
  const auto solved = solve_linear_system(field, std::move(matrix), std::move(rhs));
  if (!solved.consistent || !solved.unique) return std::nullopt;

  RationalFunction candidate;
  candidate.numerator.assign(
      solved.solution.begin(),
      solved.solution.begin() +
          static_cast<FieldVector::difference_type>(numerator_degree + 1));
  candidate.denominator.assign(denominator_degree + 1, 0);
  if (denominator_degree == 0) {
    candidate.denominator[0] = 1;
  } else {
    for (std::size_t degree = 0; degree < denominator_degree; ++degree)
      candidate.denominator[degree] = solved.solution[numerator_degree + 1 + degree];
    candidate.denominator.back() = 1;
  }
  trim_polynomial(candidate.numerator);
  trim_polynomial(candidate.denominator);
  if (candidate.numerator.size() - 1 > numerator_degree ||
      candidate.denominator.size() - 1 != denominator_degree) {
    return std::nullopt;
  }
  if (!validates(field, candidate, arguments, values)) return std::nullopt;
  return candidate;
}

std::optional<RationalFunction> interpolate_rational_thiele_monic(
    const PrimeField& field, std::span<const std::uint64_t> arguments,
    std::span<const std::uint64_t> values, std::size_t holdout_count,
    RationalInterpolationDiagnostics* diagnostics)
{
  RationalInterpolationDiagnostics local;
  auto& report = diagnostics ? *diagnostics : local;
  report = {};
  if (arguments.size() != values.size() || holdout_count == 0 ||
      arguments.size() <= holdout_count) {
    throw std::invalid_argument("invalid Thiele rational interpolation samples");
  }
  if (firefly::FFInt::p != field.prime())
    throw std::logic_error("FireFly prime is not configured for Thiele interpolation");
  if (std::ranges::all_of(values,
                          [&](const auto value) { return value == values.front(); })) {
    report.samples_fed = 1;
    return RationalFunction{.numerator = {values.front()}, .denominator = {1}};
  }

  firefly::ThieleInterpolator interpolator;
  const auto training = arguments.size() - holdout_count;
  bool produced_candidate = false;
  const auto& orders = thiele_sample_orders(training);

  for (const auto& order : orders) {
    ++report.attempts;
    interpolator = firefly::ThieleInterpolator{};
    for (std::size_t position = 0; position < training; ++position) {
      report.samples_fed = std::max(report.samples_fed, position + 1);
      const auto sample = order[position];
      const auto completed = interpolator.add_point(firefly::FFInt(values[sample]),
                                                    firefly::FFInt(arguments[sample]));
      if (!completed) continue;
      produced_candidate = true;
      try {
        auto [numerator, denominator] = interpolator.get_result();
        auto candidate = normalize_rational_function(
            field, {.numerator = dense_polynomial(numerator),
                    .denominator = dense_polynomial(denominator)});
        report.numerator_degree = candidate.numerator.size() - 1;
        report.denominator_degree = candidate.denominator.size() - 1;
        if (!validates(field, candidate, arguments, values)) continue;
        return candidate;
      } catch (const std::exception&) {
        // Try the remaining samples and deterministic input permutations. The
        // independent holdouts remain the acceptance certificate.
      }
    }
  }
  report.failure = produced_candidate ? "thiele_candidate_failed_holdouts"
                                      : "thiele_did_not_terminate";
  return std::nullopt;
}

std::size_t moving_polynomial_degree(const PrimeField& field,
                                     std::span<const FieldVector> polynomials)
{
  std::optional<FieldVector> common;
  std::size_t maximum_degree = 0;
  for (const auto& polynomial : polynomials) {
    if (polynomial.size() <= 1) continue;
    auto normalized = monic_polynomial(field, polynomial);
    maximum_degree = std::max(maximum_degree, normalized.size() - 1);
    common = common ? polynomial_gcd(field, std::move(*common), normalized)
                    : std::optional<FieldVector>(std::move(normalized));
  }
  if (!common) return 0;
  return maximum_degree - (common->size() - 1);
}

bool stable_polynomial_signatures(const PrimeField& field,
                                  std::span<const FieldVector> polynomials)
{
  std::optional<FieldVector> reference;
  for (const auto& polynomial : polynomials) {
    if (polynomial.size() <= 1) continue;
    auto normalized = monic_polynomial(field, polynomial);
    if (!reference) {
      reference = std::move(normalized);
    } else if (*reference != normalized) {
      return false;
    }
  }
  return true;
}

} // namespace basis
