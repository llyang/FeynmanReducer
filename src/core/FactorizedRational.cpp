#include "core/FactorizedRational.hpp"

#include <flint/fmpz_mpoly_factor.h>
#include <flint/nmod.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {
std::int64_t add_power(std::int64_t a, std::int64_t b)
{
  std::int64_t result;
  if (__builtin_add_overflow(a, b, &result) ||
      result == std::numeric_limits<std::int64_t>::min())
    throw std::overflow_error("factor multiplicity overflow");
  return result;
}
} // namespace

FactorizedRational::FactorizedRational(
    std::shared_ptr<const FlintRationalContext> context)
    : numerator_(context), scalar_(context)
{
  fmpz_mpoly_q_one(scalar_.raw(), context->raw());
}

FactorizedRational::FactorizedRational(FlintRational residual)
    : FactorizedRational(residual.context())
{
  set_residual(std::move(residual));
  normalize();
}

void FactorizedRational::set_residual(FlintRational residual)
{
  residual.canonicalise();
  fmpz_mpoly_set(fmpz_mpoly_q_numref(numerator_.raw()),
                 fmpz_mpoly_q_numref(residual.raw()), context()->raw());
  if (!is_zero()) factor_polynomial(fmpz_mpoly_q_denref(residual.raw()), -1);
}

void FactorizedRational::set_numerator(FlintRational numerator)
{
  if (numerator.context().get() != context().get() ||
      !fmpz_mpoly_is_one(fmpz_mpoly_q_denref(numerator.raw()), context()->raw()))
    throw std::logic_error("invalid factorized numerator");
  numerator_ = std::move(numerator);
}

void FactorizedRational::set_scalar(FlintRational scalar)
{
  if (scalar.context().get() != context().get() ||
      !fmpz_mpoly_is_fmpz(fmpz_mpoly_q_numref(scalar.raw()), context()->raw()) ||
      !fmpz_mpoly_is_fmpz(fmpz_mpoly_q_denref(scalar.raw()), context()->raw()))
    throw std::logic_error("invalid factorized scalar");
  scalar.canonicalise();
  scalar_ = std::move(scalar);
}

void FactorizedRational::factor_polynomial(const fmpz_mpoly_struct* polynomial,
                                           std::int64_t sign)
{
  const auto* ctx = context()->raw();
  fmpz_mpoly_factor_t decomposition;
  fmpz_mpoly_factor_init(decomposition, ctx);
  try {
    if (!fmpz_mpoly_factor(decomposition, polynomial, ctx))
      throw std::runtime_error("polynomial factorization failed");
    FlintRational constant(context());
    fmpz_t value;
    fmpz_init(value);
    fmpz_mpoly_factor_get_constant_fmpz(value, decomposition, ctx);
    fmpz_mpoly_q_one(constant.raw(), ctx);
    fmpz_mpoly_set_fmpz(sign > 0 ? fmpz_mpoly_q_numref(constant.raw())
                                 : fmpz_mpoly_q_denref(constant.raw()),
                        value, ctx);
    fmpz_clear(value);
    if (fmpz_mpoly_is_zero(fmpz_mpoly_q_denref(constant.raw()), ctx))
      throw std::runtime_error("zero factor denominator");
    constant.canonicalise();
    fmpz_mpoly_q_mul(scalar_.raw(), scalar_.raw(), constant.raw(), ctx);
    for (slong i = 0; i < fmpz_mpoly_factor_length(decomposition, ctx); ++i) {
      FlintRational base(context());
      fmpz_mpoly_factor_get_base(fmpz_mpoly_q_numref(base.raw()), decomposition, i,
                                 ctx);
      add_irreducible_factor(
          std::move(base), sign * fmpz_mpoly_factor_get_exp_si(decomposition, i, ctx));
    }
  } catch (...) {
    fmpz_mpoly_factor_clear(decomposition, ctx);
    throw;
  }
  fmpz_mpoly_factor_clear(decomposition, ctx);
}

void FactorizedRational::add_scanned_factor(const FlintRational& factor)
{
  if (factor.context().get() != context().get())
    throw std::logic_error("factor context mismatch");
  if (is_zero()) return;
  if (factor.is_zero()) {
    fmpz_mpoly_q_zero(numerator_.raw(), context()->raw());
    factors_.clear();
    return;
  }
  factor_polynomial(fmpz_mpoly_q_numref(factor.raw()), 1);
  factor_polynomial(fmpz_mpoly_q_denref(factor.raw()), -1);
}

void FactorizedRational::add_irreducible_factor(FlintRational polynomial,
                                                std::int64_t power)
{
  const auto* ctx = context()->raw();
  if (polynomial.context().get() != context().get() || polynomial.is_zero() ||
      !fmpz_mpoly_is_one(fmpz_mpoly_q_denref(polynomial.raw()), ctx) ||
      power == std::numeric_limits<std::int64_t>::min())
    throw std::logic_error("invalid polynomial factor");
  if (power == 0) return;
  fmpz_t leading;
  fmpz_init(leading);
  fmpz_mpoly_get_term_coeff_fmpz(leading, fmpz_mpoly_q_numref(polynomial.raw()), 0,
                                 ctx);
  const bool negative = fmpz_sgn(leading) < 0;
  fmpz_clear(leading);
  if (negative) {
    fmpz_mpoly_neg(fmpz_mpoly_q_numref(polynomial.raw()),
                   fmpz_mpoly_q_numref(polynomial.raw()), ctx);
    if (power % 2) fmpz_mpoly_q_neg(scalar_.raw(), scalar_.raw(), ctx);
  }
  if (!fmpz_mpoly_is_one(fmpz_mpoly_q_numref(polynomial.raw()), ctx))
    factors_.push_back({std::move(polynomial), power});
}

void FactorizedRational::normalize()
{
  const auto* ctx = context()->raw();
  if (is_zero() || scalar_.is_zero()) {
    fmpz_mpoly_q_zero(numerator_.raw(), ctx);
    fmpz_mpoly_q_one(scalar_.raw(), ctx);
    factors_.clear();
    return;
  }
  // Move integer content and the residual sign into the scalar without
  // factoring (or multiplying out) the residual numerator.
  fmpz_t content, coefficient;
  fmpz_init(content);
  fmpz_init(coefficient);
  auto* polynomial = fmpz_mpoly_q_numref(numerator_.raw());
  for (slong i = 0; i < fmpz_mpoly_length(polynomial, ctx) && !fmpz_is_one(content);
       ++i) {
    fmpz_mpoly_get_term_coeff_fmpz(coefficient, polynomial, i, ctx);
    fmpz_gcd(content, content, coefficient);
  }
  fmpz_mpoly_get_term_coeff_fmpz(coefficient, polynomial, 0, ctx);
  if (fmpz_sgn(coefficient) < 0) fmpz_neg(content, content);
  if (!fmpz_is_one(content)) {
    fmpz_mpoly_scalar_divexact_fmpz(polynomial, polynomial, content, ctx);
    fmpz_mpoly_scalar_mul_fmpz(fmpz_mpoly_q_numref(scalar_.raw()),
                               fmpz_mpoly_q_numref(scalar_.raw()), content, ctx);
    scalar_.canonicalise();
  }
  fmpz_clear(content);
  fmpz_clear(coefficient);
  std::ranges::sort(factors_, [&](const Factor& a, const Factor& b) {
    return fmpz_mpoly_cmp(fmpz_mpoly_q_numref(a.polynomial.raw()),
                          fmpz_mpoly_q_numref(b.polynomial.raw()), ctx) < 0;
  });
  std::size_t kept = 0;
  for (std::size_t index = 0; index < factors_.size(); ++index) {
    auto& factor = factors_[index];
    if (kept != 0 &&
        fmpz_mpoly_equal(fmpz_mpoly_q_numref(factors_[kept - 1].polynomial.raw()),
                         fmpz_mpoly_q_numref(factor.polynomial.raw()), ctx))
      factors_[kept - 1].power = add_power(factors_[kept - 1].power, factor.power);
    else {
      if (kept != index) factors_[kept] = std::move(factor);
      ++kept;
    }
  }
  factors_.erase(factors_.begin() + static_cast<std::ptrdiff_t>(kept), factors_.end());
  FlintRational quotient(context());
  for (auto& factor : factors_) {
    while (factor.power < 0 &&
           fmpz_mpoly_divides(fmpz_mpoly_q_numref(quotient.raw()),
                              fmpz_mpoly_q_numref(numerator_.raw()),
                              fmpz_mpoly_q_numref(factor.polynomial.raw()), ctx)) {
      fmpz_mpoly_swap(fmpz_mpoly_q_numref(numerator_.raw()),
                      fmpz_mpoly_q_numref(quotient.raw()), ctx);
      ++factor.power;
    }
  }
  std::erase_if(factors_, [](const Factor& f) { return f.power == 0; });
}

std::string FactorizedRational::to_string() const
{
  if (is_zero()) return "0";
  if (factors_.empty()) {
    FlintRational value(context());
    fmpz_mpoly_q_mul(value.raw(), scalar_.raw(), numerator_.raw(), context()->raw());
    return value.to_string();
  }
  std::string numerator;
  if (!fmpz_mpoly_q_is_one(scalar_.raw(), context()->raw()))
    numerator = "(" + scalar_.to_string() + ")";
  if (!fmpz_mpoly_q_is_one(numerator_.raw(), context()->raw())) {
    if (!numerator.empty()) numerator += '*';
    numerator += "(" + numerator_.to_string() + ")";
  }
  std::string denominator;
  for (const auto& factor : factors_) {
    auto& text = factor.power > 0 ? numerator : denominator;
    if (!text.empty()) text += '*';
    text += '(' + factor.polynomial.to_string() + ')';
    const auto power = factor.power > 0 ? factor.power : -factor.power;
    if (power != 1) text += '^' + std::to_string(power);
  }
  if (numerator.empty()) numerator = "1";
  return denominator.empty() ? numerator : "(" + numerator + ")/(" + denominator + ")";
}

FlintRational FactorizedRational::expand() const
{
  FlintRational result(context());
  const auto* ctx = context()->raw();
  fmpz_mpoly_q_mul(result.raw(), scalar_.raw(), numerator_.raw(), ctx);
  for (const auto& factor : factors_) {
    FlintRational power(context());
    if (!fmpz_mpoly_pow_ui(
            fmpz_mpoly_q_numref(power.raw()),
            fmpz_mpoly_q_numref(factor.polynomial.raw()),
            static_cast<ulong>(factor.power > 0 ? factor.power : -factor.power), ctx))
      throw std::runtime_error("factor expansion failed");
    if (factor.power < 0) fmpz_mpoly_q_inv(power.raw(), power.raw(), ctx);
    fmpz_mpoly_q_mul(result.raw(), result.raw(), power.raw(), ctx);
  }
  return result;
}

std::optional<ulong> FactorizedRational::evaluate_mod(std::span<const ulong> values,
                                                      ulong prime) const
{
  if (values.size() != context()->variable_names().size())
    throw std::logic_error("factor evaluation coordinate mismatch");
  nmod_t mod;
  nmod_init(&mod, prime);
  const auto eval = [&](const fmpz_mpoly_struct* p) {
    return fmpz_mpoly_evaluate_all_nmod(p, values.data(), context()->raw(), mod);
  };
  ulong numerator = nmod_mul(eval(fmpz_mpoly_q_numref(numerator_.raw())),
                             eval(fmpz_mpoly_q_numref(scalar_.raw())), mod);
  ulong denominator = eval(fmpz_mpoly_q_denref(scalar_.raw()));
  for (const auto& factor : factors_) {
    const auto base = eval(fmpz_mpoly_q_numref(factor.polynomial.raw()));
    const auto power =
        static_cast<ulong>(factor.power > 0 ? factor.power : -factor.power);
    auto& target = factor.power > 0 ? numerator : denominator;
    target = nmod_mul(target, nmod_pow_ui(base, power, mod), mod);
  }
  if (denominator == 0) return std::nullopt;
  return nmod_div(numerator, denominator, mod);
}

std::optional<std::size_t> FactorizedRational::mixed_denominator_factor() const
{
  const auto& names = context()->variable_names();
  const auto it = std::ranges::find(names, "d");
  if (it == names.end() || is_zero()) return std::nullopt;
  const auto d = static_cast<std::size_t>(it - names.begin());
  std::vector<slong> degrees(names.size());
  for (std::size_t index = 0; index < factors_.size(); ++index) {
    const auto& factor = factors_[index];
    if (factor.power >= 0) continue;
    fmpz_mpoly_degrees_si(degrees.data(), fmpz_mpoly_q_numref(factor.polynomial.raw()),
                          context()->raw());
    if (degrees[d] <= 0) continue;
    for (std::size_t i = 0; i < degrees.size(); ++i)
      if (i != d && degrees[i] > 0) return index;
  }
  return std::nullopt;
}
