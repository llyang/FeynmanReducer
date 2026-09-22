#pragma once

#include "DSeparatingSearchStrategy.hpp"
#include "FactorSaturation.hpp"
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
  DSeparatingSearchStrategy strategy = DSeparatingSearchStrategy::SingleSlotThenScored;
  unsigned maximum_candidate_dots = 2;
  std::size_t screening_target_limit = 8;
  std::size_t maximum_basis_states = 100000;
  // First-stage cap; both stages still share maximum_basis_states.
  std::size_t simple_search_maximum_basis_states = 64;
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

// Builds the deterministic target set used exclusively for D-separating basis
// discovery: all symmetry-inequivalent d-dimensional top-sector integrals with
// both dots on one propagator (exactly one index equal to three). User-requested
// source integrals are validated only after a basis has been selected.
[[nodiscard]] std::vector<Integral>
build_d_separating_search_targets(const TopologyConfig& topology);

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
  std::size_t witnesses_considered = 0;
  std::size_t witnesses_repaired = 0;
  std::size_t child_mixed_degree = 0;
  std::size_t child_total_mixed_degree = 0;
  std::size_t pivot_mixed_degree = 0;
  std::size_t pivot_numerator_degree = 0;
};

struct DMovingPoleWitness {
  Integral target;
  std::size_t component = 0;
  std::uint32_t sector = 0;
  std::size_t mixed_degree = 0;
  std::uint32_t master_sector = 0;
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
  double rank_data_seconds = 0.0;
  double rank_coordinates_seconds = 0.0;
  double rank_pivot_seconds = 0.0;
};

struct DSeparatingSearchStageReport {
  DSeparatingSearchStrategy strategy = DSeparatingSearchStrategy::SingleSlot;
  DSeparatingBasisSearchStatus status = DSeparatingBasisSearchStatus::NotFound;
  std::string reason;
  std::size_t budget = 0;
  // Beam states in SingleSlot; attempted exchanges (including rollback) in Scored.
  std::size_t states_used = 0;
  DSeparatingBasisProbeStatistics probe_statistics;
  DSeparatingBasisTiming timing;
};

struct DSeparatingBasisSearchReport {
  std::vector<DSeparatingSearchStageReport> stages;
  DSeparatingBasisSearchStatus status = DSeparatingBasisSearchStatus::NotFound;
  std::vector<Integral> selected_basis;
  unsigned total_dots = 0;
  bool selected_basis_d_separating = false;
  std::size_t candidate_pool_size = 0;
  std::size_t basis_states_tested = 0;
  std::size_t raw_pairs = 0;
  std::size_t scored_pairs = 0;
  std::size_t pair_sampling_failures = 0;
  std::size_t maximum_moving_witnesses = 0;
  std::size_t maximum_moving_witness_slots = 0;
  std::size_t beam_discarded = 0;
  std::size_t seen_suppressed = 0;
  std::vector<std::size_t> shortlisted_by_slot;
  std::vector<factor_saturation::SequenceResult> factor_sequences;
  std::vector<std::uint64_t> sequence_primes;
  std::vector<std::size_t> sequence_anchors;
  std::vector<FieldVector> sequence_kinematics;
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
  // Canonical published basis position -> pointwise swap coordinate slot.
  std::vector<std::size_t> selected_basis_to_swap_slot;
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
    std::span<const std::size_t> slots, DSeparatingBasisTiming* timing = nullptr);

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

// Strictly validates one fixed basis and retains the exact oracle used for the
// validation as the final reduction kernel. No alternative masters or swaps
// are considered.
[[nodiscard]] PreparedDSeparatingBasisSearch
prepare_d_separating_fixed_basis_validation(
    Config config, std::vector<Integral> basis,
    std::span<const std::uint32_t> relation_source_sectors,
    const DSeparatingBasisSearchOptions& options = {},
    DSeparatingBasisProgressCallback progress = {},
    ReductionOptions reduction_options = {});

} // namespace basis
