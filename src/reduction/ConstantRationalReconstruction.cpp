#include "reduction/ConstantRationalReconstruction.hpp"

#include <flint/fmpq.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mpoly_q.h>
#include <flint/ulong_extras.h>

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

class IntegerValue {
public:
  IntegerValue()
  {
    fmpz_init(value_);
  }
  IntegerValue(const IntegerValue&) = delete;
  IntegerValue& operator=(const IntegerValue&) = delete;
  ~IntegerValue()
  {
    fmpz_clear(value_);
  }
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

class RationalValue {
public:
  RationalValue()
  {
    fmpq_init(value_);
  }
  RationalValue(const RationalValue&) = delete;
  RationalValue& operator=(const RationalValue&) = delete;
  ~RationalValue()
  {
    fmpq_clear(value_);
  }
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

bool reconstruct_all(const std::vector<IntegerValue>& residues,
                     const IntegerValue& modulus, std::vector<RationalValue>& rationals)
{
  for (std::size_t index = 0; index < residues.size(); ++index) {
    if (fmpq_reconstruct_fmpz(rationals[index].raw(), residues[index].raw(),
                              modulus.raw()) == 0) {
      return false;
    }
  }
  return true;
}

bool verify_all(const std::vector<RationalValue>& rationals,
                std::span<const std::uint64_t> values, std::uint64_t prime)
{
  for (std::size_t index = 0; index < rationals.size(); ++index) {
    const ulong denominator = fmpz_fdiv_ui(fmpq_denref(rationals[index].raw()), prime);
    if (denominator == 0) return false;
    const ulong numerator = fmpz_fdiv_ui(fmpq_numref(rationals[index].raw()), prime);
    const ulong inverse = n_invmod(denominator, prime);
    const ulong expected =
        n_mulmod2_preinv(numerator, inverse, prime, n_preinvert_limb(prime));
    if (expected != values[index]) return false;
  }
  return true;
}

std::vector<FlintRational>
materialize(const std::vector<RationalValue>& rationals,
            const std::shared_ptr<const FlintRationalContext>& context)
{
  std::vector<FlintRational> result;
  result.reserve(rationals.size());
  for (const auto& rational : rationals) {
    FlintRational value(context);
    fmpz_mpoly_q_set_fmpq(value.raw(), rational.raw(),
                          const_cast<fmpz_mpoly_ctx_struct*>(context->raw()));
    result.push_back(std::move(value));
  }
  return result;
}

} // namespace

namespace reduction::detail {

std::vector<FlintRational> reconstruct_constant_rationals(
    std::size_t output_count,
    const std::shared_ptr<const FlintRationalContext>& context,
    const std::vector<std::uint64_t>& primes,
    const ConstantFiniteFieldEvaluator& evaluate)
{
  if (!context || !context->variable_names().empty()) {
    throw std::invalid_argument(
        "constant rational reconstruction requires a zero-variable context");
  }
  if (!evaluate) {
    throw std::invalid_argument(
        "constant rational reconstruction requires an evaluator");
  }
  if (output_count == 0) return {};

  std::vector<IntegerValue> residues(output_count);
  IntegerValue modulus;
  fmpz_one(modulus.raw());
  std::vector<RationalValue> rationals(output_count);
  bool candidate_available = false;
  std::size_t accepted_primes = 0;
  std::size_t skipped_primes = 0;

  for (const std::uint64_t prime : primes) {
    const auto values = evaluate(prime);
    if (values.empty()) {
      ++skipped_primes;
      continue;
    }
    if (values.size() != output_count) {
      throw std::runtime_error(
          "constant finite-field evaluation returned the wrong output count");
    }
    if (candidate_available && verify_all(rationals, values, prime)) {
      return materialize(rationals, context);
    }

    for (std::size_t index = 0; index < output_count; ++index) {
      fmpz_CRT_ui(residues[index].raw(), residues[index].raw(), modulus.raw(),
                  static_cast<ulong>(values[index]), static_cast<ulong>(prime), 0);
    }
    fmpz_mul_ui(modulus.raw(), modulus.raw(), static_cast<ulong>(prime));
    ++accepted_primes;
    candidate_available =
        accepted_primes >= 2 && reconstruct_all(residues, modulus, rationals);
  }

  throw std::runtime_error(
      "constant rational reconstruction exhausted FireFly primes: accepted=" +
      std::to_string(accepted_primes) + ", skipped=" + std::to_string(skipped_primes) +
      ", modulus_bits=" + std::to_string(fmpz_bits(modulus.raw())));
}

} // namespace reduction::detail
