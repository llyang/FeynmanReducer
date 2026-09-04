#pragma once

#include "FiniteField.hpp"
#include "core/AtomicSharedPtr.hpp"
#include "core/Config.hpp"
#include "reduction/ReductionOptions.hpp"
#include "reduction/ReductionProgress.hpp"

#include <firefly/FFInt.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

class BlackBoxFeynman;

namespace basis {

struct NativeOracleEvaluation {
  FieldMatrix conventional;
  FieldMatrix quotient_normalized;
  std::size_t selected_reconstructed_outputs = 0;
  bool replay_performed = false;
};

// Prepared finite-field reduction oracle for one target batch and one fixed
// denominator-only physical basis.  The object owns Config because the native
// replay kernel keeps a reference to it.
class NativeFixedBasisOracle {
public:
  class PreparedRows {
  public:
    PreparedRows(const PreparedRows&) = delete;
    PreparedRows& operator=(const PreparedRows&) = delete;
    ~PreparedRows() = default;

  private:
    friend class NativeFixedBasisOracle;

    PreparedRows() = default;

    std::weak_ptr<const int> owner;
    std::vector<std::size_t> target_rows;
    std::vector<std::uint32_t> active_outputs;
    std::vector<std::pair<std::size_t, std::size_t>> destinations;
  };

  [[nodiscard]] static std::unique_ptr<NativeFixedBasisOracle>
  prepare(Config config, std::vector<Integral> basis,
          ReductionProgressCallback progress = {},
          ReplayOrientationPreference replay_orientation =
              ReplayOrientationPreference::Auto,
          std::span<const std::uint32_t> relation_source_sectors = {},
          bool compute_quotient_normalized = true,
          ReductionOptions reduction_options = {});

  NativeFixedBasisOracle(const NativeFixedBasisOracle&) = delete;
  NativeFixedBasisOracle& operator=(const NativeFixedBasisOracle&) = delete;
  ~NativeFixedBasisOracle();

  void set_prime(std::uint64_t prime);

  [[nodiscard]] std::optional<NativeOracleEvaluation>
  evaluate(std::uint64_t dimension, std::span<const std::uint64_t> kinematic_values);

  // Evaluates only the requested target rows. Rows must be unique; the result
  // preserves their supplied order. Probabilistic-zero output components are
  // materialized as exact finite-field zeros.
  [[nodiscard]] std::optional<NativeOracleEvaluation>
  evaluate_rows(std::uint64_t dimension,
                std::span<const std::uint64_t> kinematic_values,
                std::span<const std::size_t> target_rows);

  // Prepare the structural output selection once and reuse it across finite-field
  // points and primes. The returned handle is immutable and thread-safe.
  [[nodiscard]] std::shared_ptr<const PreparedRows>
  prepare_rows(std::span<const std::size_t> target_rows);

  [[nodiscard]] std::optional<NativeOracleEvaluation>
  evaluate_rows(std::uint64_t dimension,
                std::span<const std::uint64_t> kinematic_values,
                const PreparedRows& prepared_rows);

  // Evaluates a complete parameter vector that already follows Config::parameters.
  // The vector overload is the zero-copy hot path used by DSeparatingReduction;
  // the span overload is provided for other prepared callers.
  [[nodiscard]] std::optional<NativeOracleEvaluation>
  evaluate_rows(const std::vector<firefly::FFInt>& values,
                const PreparedRows& prepared_rows);

  [[nodiscard]] std::optional<NativeOracleEvaluation>
  evaluate_rows(std::span<const firefly::FFInt> values,
                const PreparedRows& prepared_rows);

  [[nodiscard]] const Config& config() const noexcept
  {
    return config_;
  }
  [[nodiscard]] std::span<const Integral> basis() const noexcept
  {
    return config_.basis;
  }
  [[nodiscard]] std::span<const Integral> targets() const noexcept
  {
    return config_.targets;
  }
  [[nodiscard]] std::span<const std::uint32_t> reconstructed_outputs() const noexcept;
  [[nodiscard]] std::size_t total_output_count() const noexcept;

private:
  explicit NativeFixedBasisOracle(Config config, bool compute_quotient_normalized);

  [[nodiscard]] std::optional<NativeOracleEvaluation>
  evaluate_complete_values(const std::vector<firefly::FFInt>& values,
                           const PreparedRows& prepared_rows);

  Config config_;
  bool compute_quotient_normalized_ = true;
  std::vector<std::optional<FieldVector>> unit_rows_;
  std::unique_ptr<BlackBoxFeynman> black_box_;
  const std::shared_ptr<const int> identity_ = std::make_shared<const int>(0);
  std::shared_ptr<const PreparedRows> all_rows_;
  core::AtomicSharedPtr<const PreparedRows> last_rows_;
  std::optional<std::uint64_t> prime_;
};

} // namespace basis

namespace quotient {
using basis::NativeFixedBasisOracle;
using basis::NativeOracleEvaluation;
} // namespace quotient
