#include "reduction/FireflyResultImport.hpp"
#include "core/detail/FactorizedRationalBuilder.hpp"

#include <firefly/Polynomial.hpp>
#include <firefly/RationalFunction.hpp>

#include <flint/fmpq.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mpoly.h>
#include <flint/fmpz_mpoly_q.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

class FmpzValue {
public:
  FmpzValue()
  {
    fmpz_init(value_);
  }
  ~FmpzValue()
  {
    fmpz_clear(value_);
  }

  FmpzValue(const FmpzValue&) = delete;
  FmpzValue& operator=(const FmpzValue&) = delete;

  [[nodiscard]] fmpz* raw() noexcept
  {
    return value_;
  }
  [[nodiscard]] const fmpz* raw() const noexcept
  {
    return value_;
  }

private:
  fmpz_t value_;
};

class FmpqValue {
public:
  FmpqValue()
  {
    fmpq_init(value_);
  }
  ~FmpqValue()
  {
    fmpq_clear(value_);
  }

  FmpqValue(const FmpqValue&) = delete;
  FmpqValue& operator=(const FmpqValue&) = delete;

  [[nodiscard]] fmpq* raw() noexcept
  {
    return value_;
  }
  [[nodiscard]] const fmpq* raw() const noexcept
  {
    return value_;
  }

private:
  fmpq_t value_;
};

using VariableOrder = std::unordered_map<std::uint32_t, std::uint32_t>;

void validate_variable_order(const VariableOrder& order, std::size_t variable_count)
{
  if (order.empty()) return;
  if (order.size() != variable_count) {
    throw std::runtime_error("FireFly returned an incomplete variable-order map");
  }
  std::vector<bool> targets(variable_count, false);
  for (const auto& [source, target] : order) {
    if (source >= variable_count || target >= variable_count) {
      throw std::runtime_error("FireFly returned an out-of-range variable-order map");
    }
    if (targets[target]) {
      throw std::runtime_error("FireFly returned a non-permutation variable-order map");
    }
    targets[target] = true;
  }
}

void set_rational(FmpqValue& destination, const firefly::RationalNumber& coefficient)
{
  fmpz_set_mpz(fmpq_numref(destination.raw()), coefficient.numerator.get_mpz_t());
  fmpz_set_mpz(fmpq_denref(destination.raw()), coefficient.denominator.get_mpz_t());
  if (fmpz_is_zero(fmpq_denref(destination.raw())))
    throw std::runtime_error("FireFly returned a coefficient with zero denominator");
  fmpq_canonicalise(destination.raw());
}

void set_term_powers(std::vector<ulong>& destination, const firefly::Monomial& term,
                     int variable_position, const VariableOrder& order)
{
  std::ranges::fill(destination, 0);
  const std::size_t variable_count = destination.size();
  if (variable_position >= 0) {
    const auto target = static_cast<std::size_t>(variable_position);
    if (target >= variable_count) {
      throw std::runtime_error("FireFly factor refers to an out-of-range variable");
    }
    if (term.powers.size() > 1) {
      throw std::runtime_error(
          "FireFly factor marked as univariate has multivariate powers");
    }
    if (!term.powers.empty()) {
      destination[target] = static_cast<ulong>(term.powers.front());
    }
    return;
  }

  if (term.powers.size() > variable_count) {
    throw std::runtime_error(
        "FireFly polynomial has more variables than the FLINT context");
  }
  for (std::size_t source = 0; source < term.powers.size(); ++source) {
    const std::size_t target =
        order.empty() ? source : order.at(static_cast<std::uint32_t>(source));
    destination[target] = static_cast<ulong>(term.powers[source]);
  }
}

void import_polynomial(fmpz_mpoly_struct* destination,
                       const firefly::Polynomial& source,
                       const FlintRationalContext& context, const VariableOrder& order,
                       fmpz* common_denominator)
{
  fmpz_mpoly_zero(destination, context.raw());
  fmpz_one(common_denominator);
  if (source.coefs.empty() || source.zero()) return;

  FmpqValue rational;
  for (const auto& term : source.coefs) {
    set_rational(rational, term.coef);
    fmpz_lcm(common_denominator, common_denominator, fmpq_denref(rational.raw()));
  }

  std::vector<ulong> powers(context.variable_names().size(), 0);
  FmpzValue coefficient;
  FmpzValue multiplier;
  const auto insert_terms = [&](bool bulk) {
    for (const auto& term : source.coefs) {
      set_rational(rational, term.coef);
      fmpz_divexact(multiplier.raw(), common_denominator, fmpq_denref(rational.raw()));
      fmpz_mul(coefficient.raw(), fmpq_numref(rational.raw()), multiplier.raw());
      set_term_powers(powers, term, source.get_var_pos(), order);
      if (bulk)
        fmpz_mpoly_push_term_fmpz_ui(destination, coefficient.raw(), powers.data(),
                                     context.raw());
      else
        fmpz_mpoly_set_coeff_fmpz_ui(destination, coefficient.raw(), powers.data(),
                                     context.raw());
    }
  };
  fmpz_mpoly_fit_length(destination, static_cast<slong>(source.coefs.size()),
                        context.raw());
  insert_terms(true);
  fmpz_mpoly_sort_terms(destination, context.raw());
  if (!fmpz_mpoly_is_canonical(destination, context.raw())) {
    // Public FireFly coefficient vectors can contain repeated exponents (also
    // after padding short power vectors). Preserve the previous last-write wins
    // behavior, including explicit zeros, rather than summing duplicate terms.
    fmpz_mpoly_zero(destination, context.raw());
    insert_terms(false);
  }
}

[[nodiscard]] FlintRational
import_rational_impl(const firefly::RationalFunction& source,
                     const std::shared_ptr<const FlintRationalContext>& context,
                     bool is_factor)
{
  const VariableOrder order = source.get_order_map();
  validate_variable_order(order, context->variable_names().size());

  FlintRational result(context);
  FmpzValue numerator_denominator;
  FmpzValue denominator_denominator;
  if (is_factor && (source.numerator.coefs.empty() || source.numerator.zero())) {
    // Denominator-only scanned factors use an empty numerator as the
    // implicit unit numerator. A top-level empty numerator remains zero.
    fmpz_mpoly_one(fmpz_mpoly_q_numref(result.raw()), context->raw());
    fmpz_one(numerator_denominator.raw());
  } else {
    import_polynomial(fmpz_mpoly_q_numref(result.raw()), source.numerator, *context,
                      order, numerator_denominator.raw());
  }
  if (source.denominator.coefs.empty() || source.denominator.zero()) {
    // FireFly uses an empty denominator polynomial as the implicit unit
    // denominator for numerator-only factors.
    fmpz_mpoly_one(fmpz_mpoly_q_denref(result.raw()), context->raw());
    fmpz_one(denominator_denominator.raw());
  } else {
    import_polynomial(fmpz_mpoly_q_denref(result.raw()), source.denominator, *context,
                      order, denominator_denominator.raw());
  }
  fmpz_mpoly_scalar_mul_fmpz(fmpz_mpoly_q_numref(result.raw()),
                             fmpz_mpoly_q_numref(result.raw()),
                             denominator_denominator.raw(), context->raw());
  fmpz_mpoly_scalar_mul_fmpz(fmpz_mpoly_q_denref(result.raw()),
                             fmpz_mpoly_q_denref(result.raw()),
                             numerator_denominator.raw(), context->raw());
  // Leave the local rational unreduced: the residual builder canonicalizes it
  // once, while scanned factors are factored separately and cancelled at finish.

  return result;
}

} // namespace

namespace reduction_detail {

FactorizedRational
import_firefly_rational(const firefly::RationalFunction& source,
                        const std::shared_ptr<const FlintRationalContext>& context)
{
  if (!context) {
    throw std::runtime_error("FireFly result import requires a FLINT context");
  }
  factorized_detail::Builder result(import_rational_impl(source, context, false));
  const auto add_factors = [&](const auto& self,
                               const firefly::RationalFunction& parent) -> void {
    for (const auto& factor : parent.get_factors()) {
      result.add_scanned_factor(import_rational_impl(factor, context, true));
      self(self, factor);
    }
  };
  add_factors(add_factors, source);
  return std::move(result).finish();
}

} // namespace reduction_detail
