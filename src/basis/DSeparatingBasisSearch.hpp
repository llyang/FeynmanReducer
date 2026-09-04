#pragma once

#include "NativeFixedBasisOracle.hpp"
#include "core/Config.hpp"
#include "reduction/ReductionProgress.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace basis {

struct DSeparatingBasisSearchOptions {
  unsigned maximum_candidate_dots = 2;
  std::size_t screening_target_limit = 8;
  std::size_t maximum_basis_states = 100000;
  std::size_t basis_beam_width = 8;
  std::size_t swap_shortlist = 64;
  std::size_t initial_dimension_samples = 16;
  std::size_t dimension_sample_batch = 8;
  std::size_t maximum_dimension_samples = 64;
  std::size_t dimension_holdouts = 8;
  std::size_t kinematic_training_points = 3;
  std::size_t kinematic_holdout_points = 2;
  std::size_t progress_interval_seconds = 10;
};

using DSeparatingBasisProgressCallback = std::function<void(std::string_view message)>;

struct DDenominatorSignature {
  FieldVector coefficients;

  bool operator==(const DDenominatorSignature&) const = default;
};

// Re-expresses one coefficient row after replacing basis[slot] by an integral
// whose coordinates in the old basis are pivot_coordinates.
[[nodiscard]] FieldVector rebase_after_basis_swap(
    const PrimeField& field, std::span<const std::uint64_t> coefficients,
    std::span<const std::uint64_t> pivot_coordinates, std::size_t slot);

[[nodiscard]] unsigned integral_dot_count(const TopologyConfig& topology,
                                          const Integral& integral);

struct BasisIntegralPool {
  std::vector<Integral> integrals;
  std::vector<unsigned> dot_counts;

  [[nodiscard]] static BasisIntegralPool build(const TopologyConfig& topology,
                                               std::span<const Integral> initial_basis,
                                               unsigned maximum_candidate_dots);
};

enum class DSeparatingBasisSearchStatus {
  Passed,
  NotFound,
  SearchBudgetExhausted,
  SamplingFailure,
};

struct DSeparatingBasisSwap {
  std::size_t slot = 0;
  Integral removed;
  Integral inserted;
  std::uint32_t sector = 0;
  bool repairs_primary = false;
  bool pivot_numerator_stable = false;
  std::size_t child_mixed_degree = 0;
  std::size_t pivot_mixed_degree = 0;
  std::size_t pivot_numerator_degree = 0;
};

struct DMovingPoleWitness {
  Integral target;
  std::size_t component = 0;
  std::uint32_t sector = 0;
  std::size_t mixed_degree = 0;
};

struct DSeparatingBasisProbeStatistics {
  std::size_t logical_requests = 0;
  std::size_t full_cache_hits = 0;
  std::size_t requested_target_rows = 0;
  std::size_t cached_target_rows = 0;
  std::size_t replayed_target_rows = 0;
  std::size_t selected_replay_calls = 0;
  std::size_t selected_compact_outputs = 0;
  std::size_t skipped_compact_outputs = 0;
  std::size_t rational_interpolation_attempts = 0;
  std::size_t rational_interpolation_cache_hits = 0;
  std::size_t fixed_degree_attempts = 0;
  std::size_t thiele_attempts = 0;
  std::size_t exhaustive_fallbacks = 0;
  std::size_t completed_target_rows = 0;
  std::size_t avoided_basis_factorizations = 0;
};

struct DSeparatingBasisTiming {
  double total_seconds = 0.0;
  double native_prepare_seconds = 0.0;
  double rank_filter_seconds = 0.0;
  double screening_seconds = 0.0;
  double pivot_scoring_seconds = 0.0;
  double strict_validation_seconds = 0.0;
  double probe_evaluation_seconds = 0.0;
  double pointwise_rebase_seconds = 0.0;
  double rational_interpolation_seconds = 0.0;
  double signature_comparison_seconds = 0.0;
};

struct DSeparatingBasisSearchReport {
  DSeparatingBasisSearchStatus status = DSeparatingBasisSearchStatus::NotFound;
  std::vector<Integral> selected_basis;
  unsigned total_dots = 0;
  bool selected_basis_d_separating = false;
  std::size_t candidate_pool_size = 0;
  std::size_t basis_states_tested = 0;
  std::size_t swaps_scored = 0;
  std::size_t swaps_shortlisted = 0;
  std::size_t maximum_beam_depth = 0;
  std::size_t strict_validations = 0;
  std::size_t rank_deficient_rejects = 0;
  std::size_t moving_d_pole_rejects = 0;
  std::size_t sampling_rejects = 0;
  std::vector<Integral> screening_targets;
  std::vector<Integral> validation_targets;
  std::vector<std::uint64_t> primes;
  std::optional<DMovingPoleWitness> primary_witness;
  std::vector<DSeparatingBasisSwap> swap_path;
  DSeparatingBasisProbeStatistics probe_statistics;
  DSeparatingBasisTiming timing;
  std::string message;
};

// In-memory result used by the production reduction path. The retained oracle
// owns the one prepared fixed-basis kernel used by both discovery probes and
// the subsequent selected-basis reconstruction adapter.
struct PreparedDSeparatingBasisSearch {
  DSeparatingBasisSearchReport report;
  std::unique_ptr<NativeFixedBasisOracle> oracle;
  // Full target-major/selected-basis-major positions retained after two-prime
  // support validation. These positions define the final FireFly output shape.
  std::vector<std::uint32_t> final_output_support;
};

// A pointwise basis-change tape.  Each pivot vector is expressed in the basis
// produced by all preceding swaps.  It can therefore be replayed on arbitrary
// coefficient rows without factoring a dense basis-change matrix.
struct BasisSwapCoordinateTape {
  std::vector<std::size_t> slots;
  FieldMatrix pivot_coordinates;
  // Inverses of pivot_coordinates[step][slots[step]].  Computing them while
  // the tape is built turns replay into multiplication-only coordinate updates.
  FieldVector pivot_inverses;
};

[[nodiscard]] std::optional<BasisSwapCoordinateTape> build_basis_swap_coordinate_tape(
    const PrimeField& field, std::span<const FieldVector> inserted_initial_coordinates,
    std::span<const std::size_t> slots);

[[nodiscard]] FieldVector
rebase_after_basis_swaps(const PrimeField& field,
                         std::span<const std::uint64_t> coefficients,
                         const BasisSwapCoordinateTape& tape);

[[nodiscard]] PreparedDSeparatingBasisSearch prepare_d_separating_basis_search(
    Config config, std::vector<Integral> initial_basis,
    std::span<const std::uint32_t> relation_source_sectors,
    const DSeparatingBasisSearchOptions& options = {},
    DSeparatingBasisProgressCallback progress = {},
    ReductionOptions reduction_options = {});

} // namespace basis

namespace quotient {
using basis::BasisIntegralPool;
using basis::BasisSwapCoordinateTape;
using basis::build_basis_swap_coordinate_tape;
using basis::DDenominatorSignature;
using basis::DMovingPoleWitness;
using basis::DSeparatingBasisProbeStatistics;
using basis::DSeparatingBasisProgressCallback;
using basis::DSeparatingBasisSearchOptions;
using basis::DSeparatingBasisSearchReport;
using basis::DSeparatingBasisSearchStatus;
using basis::DSeparatingBasisSwap;
using basis::DSeparatingBasisTiming;
using basis::integral_dot_count;
using basis::prepare_d_separating_basis_search;
using basis::PreparedDSeparatingBasisSearch;
using basis::rebase_after_basis_swap;
using basis::rebase_after_basis_swaps;
} // namespace quotient
