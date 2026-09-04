#include "core/FlintRational.hpp"

#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mpoly.h>

#include <stdexcept>
#include <utility>

FlintRationalContext::FlintRationalContext(std::vector<std::string> variable_names)
    : variable_names_(std::move(variable_names))
{
  fmpz_mpoly_ctx_init(context_, static_cast<slong>(variable_names_.size()), ORD_LEX);
}

FlintRationalContext::~FlintRationalContext()
{
  fmpz_mpoly_ctx_clear(context_);
}

FlintRational::FlintRational(std::shared_ptr<const FlintRationalContext> context)
    : context_(std::move(context))
{
  if (!context_) {
    throw std::runtime_error("FLINT rational requires a context");
  }
  fmpz_mpoly_q_init(value_, context_->raw());
  fmpz_mpoly_q_zero(value_, context_->raw());
}

FlintRational::FlintRational(const FlintRational& other) : context_(other.context_)
{
  fmpz_mpoly_q_init(value_, context_->raw());
  fmpz_mpoly_q_set(value_, other.value_, context_->raw());
}

FlintRational& FlintRational::operator=(const FlintRational& other)
{
  if (this == &other) {
    return *this;
  }
  if (context_.get() != other.context_.get()) {
    throw std::runtime_error("cannot assign across FLINT contexts");
  }
  fmpz_mpoly_q_set(value_, other.value_, context_->raw());
  return *this;
}

FlintRational::FlintRational(FlintRational&& other) noexcept
    : context_(other.context_) // NOLINT(performance-move-constructor-init)
{
  // The moved-from value still needs the context to clear its swapped FLINT
  // object in its destructor, so the shared_ptr must deliberately be copied.
  fmpz_mpoly_q_init(value_, context_->raw());
  fmpz_mpoly_q_swap(value_, other.value_, context_->raw());
}

FlintRational& FlintRational::operator=(FlintRational&& other) noexcept
{
  if (this != &other) {
    if (context_.get() == other.context_.get()) {
      fmpz_mpoly_q_swap(value_, other.value_, context_->raw());
    } else {
      fmpz_mpoly_q_clear(value_, context_->raw());
      context_ = other.context_;
      fmpz_mpoly_q_init(value_, context_->raw());
      fmpz_mpoly_q_swap(value_, other.value_, context_->raw());
    }
  }
  return *this;
}

FlintRational::~FlintRational()
{
  fmpz_mpoly_q_clear(value_, context_->raw());
}

void FlintRational::canonicalise()
{
  fmpz_mpoly_q_canonicalise(value_, context_->raw());
}

std::string FlintRational::to_string() const
{
  std::vector<const char*> names;
  names.reserve(context_->variable_names().size());
  for (const auto& name : context_->variable_names()) {
    names.push_back(name.c_str());
  }
  char* raw = fmpz_mpoly_q_get_str_pretty(value_, names.data(), context_->raw());
  if (raw == nullptr) {
    throw std::runtime_error("FLINT failed to format a rational function");
  }
  std::string result(raw);
  flint_free(raw);
  return result;
}

bool FlintRational::is_zero() const noexcept
{
  return fmpz_mpoly_is_zero(fmpz_mpoly_q_numref(value_), context_->raw()) != 0;
}
