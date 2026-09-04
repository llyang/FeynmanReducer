#pragma once

#include <gmp.h>

#include <flint/fmpz_mpoly_q.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class FlintRationalContext {
public:
  explicit FlintRationalContext(std::vector<std::string> variable_names);
  ~FlintRationalContext();

  FlintRationalContext(const FlintRationalContext&) = delete;
  FlintRationalContext& operator=(const FlintRationalContext&) = delete;

  [[nodiscard]] const fmpz_mpoly_ctx_struct* raw() const noexcept
  {
    return context_;
  }
  [[nodiscard]] const std::vector<std::string>& variable_names() const noexcept
  {
    return variable_names_;
  }

private:
  std::vector<std::string> variable_names_;
  fmpz_mpoly_ctx_t context_;
};

class FlintRational {
public:
  explicit FlintRational(std::shared_ptr<const FlintRationalContext> context);
  FlintRational(const FlintRational& other);
  FlintRational& operator=(const FlintRational& other);
  FlintRational(FlintRational&& other) noexcept;
  FlintRational& operator=(FlintRational&& other) noexcept;
  ~FlintRational();

  [[nodiscard]] const fmpz_mpoly_q_struct* raw() const noexcept
  {
    return value_;
  }
  [[nodiscard]] fmpz_mpoly_q_struct* raw() noexcept
  {
    return value_;
  }
  [[nodiscard]] const std::shared_ptr<const FlintRationalContext>&
  context() const noexcept
  {
    return context_;
  }
  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] bool is_zero() const noexcept;
  void canonicalise();

private:
  std::shared_ptr<const FlintRationalContext> context_;
  fmpz_mpoly_q_t value_;
};
