#pragma once

#include "FiniteField.hpp"
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>

namespace basis::factor_saturation {
using RationalRow = std::vector<RationalFunction>;
constexpr int infinite_valuation = std::numeric_limits<int>::max();
RationalFunction normalize(const PrimeField&, RationalFunction);
RationalFunction subtract(const PrimeField&, const RationalFunction&,
                          const RationalFunction&);
RationalFunction multiply(const PrimeField&, const RationalFunction&,
                          const RationalFunction&);
RationalFunction divide(const PrimeField&, const RationalFunction&,
                        const RationalFunction&);
std::vector<FieldVector> factors(const PrimeField&, const FieldVector&);
FieldVector noncommon_part(const PrimeField&, std::span<const FieldVector>,
                           std::size_t);
int valuation(const PrimeField&, const RationalFunction&, const FieldVector&);
RationalRow rebase(const PrimeField&, const RationalRow&, const RationalRow&,
                   std::size_t);

struct Step {
  std::size_t candidate = 0, slot = 0;
  int pivot_valuation = 0;
  std::size_t regular_rows = 0;
};
struct BlockedRow {
  std::size_t candidate = 0;
  int minimum = 0;
  std::vector<std::size_t> minimum_slots;
};
struct PivotPair {
  std::size_t candidate = 0, slot = 0;
  int valuation = 0;
  std::size_t reintroduced = 0, reintroduced_degree = 0;
};
struct NumeratorScore {
  // 0: stable, 1: moving, 2: incomplete (never a separation certificate).
  unsigned classification = 2;
  std::size_t moving_degree = 0, degree = 0;
};
struct PairDecision {
  std::size_t legal_pairs = 0, cycle_excluded = 0, scored_pairs = 0;
  std::vector<FieldVector> protected_factors;
  // (reintroduced count, degree sum) -> number of fresh pairs.
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> risk_groups;
  std::optional<PivotPair> selected;
  NumeratorScore numerator;
  std::size_t incomplete_scores = 0;
  double scoring_seconds = 0, coordinate_seconds = 0, data_seconds = 0;
};
struct PairScores {
  std::vector<NumeratorScore> values;
  double data_seconds = 0, coordinate_seconds = 0;
};
using PairScorer =
    std::function<PairScores(std::span<const PivotPair>, std::span<const Step>)>;
struct SaturationResult {
  std::string status = "interpolation_incomplete";
  std::vector<std::size_t> selected;
  std::vector<Step> steps;
  std::vector<BlockedRow> blocked;
  std::vector<PairDecision> pair_decisions;
  std::size_t covered = 0, regular = 0;
};
using RowPrefetch = std::function<void(std::size_t, std::size_t)>;
using RowProvider = std::function<std::optional<RationalRow>(std::size_t)>;
RationalRow apply_steps(const PrimeField&, RationalRow, std::span<const Step>,
                        const RowProvider&);

struct FactorScan {
  std::string status = "complete";
  std::vector<FieldVector> factors;
};
struct FactorObservation {
  FieldVector factor;
  std::size_t candidate_poles = 0, target_poles = 0;
  bool reintroduced = false;
  std::size_t first_seen_round = 0;
};
struct SequenceRound {
  FieldVector factor;
  std::string status;
  SaturationResult saturation;
  std::vector<Step> cumulative_steps;
  std::vector<FactorObservation> observations;
};
struct SequenceResult {
  std::string status;
  std::vector<std::size_t> selected;
  std::vector<Step> steps;
  std::vector<SequenceRound> rounds;
  std::size_t states_used = 0;
};
using FactorFinder = std::function<FactorScan(std::span<const Step>)>;
using SequenceValidator =
    std::function<std::string(std::span<const std::size_t>, std::span<const Step>)>;
// Exact algebra over the supplied F_p(d) model; local regularity is not a
// certificate for unspecialized kinematics. A nonempty PairScorer is required.
// rows [0,sectors.size()) are candidates, remaining rows are original targets.
// The provider always returns coordinates in the original basis; a committed
// prefix is applied lazily. Local saturation never mutates committed coordinates.
SequenceResult sequential(const PrimeField&, std::span<const std::size_t> initial,
                          std::span<const std::uint32_t> sectors,
                          std::size_t target_count, const RowProvider&,
                          const FactorFinder&, const SequenceValidator&,
                          std::size_t maximum_states, const PairScorer& score_pairs,
                          const RowPrefetch& prefetch = {});
} // namespace basis::factor_saturation
