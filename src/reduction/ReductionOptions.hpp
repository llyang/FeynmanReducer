#pragma once

#include <cstddef>

enum class NumeratorReductionStrategy {
  Projected,
  Direct,
};

enum class ReplayOrientationPreference {
  Auto,
  Target,
  Master,
};

enum class AnsatzDotOrdering {
  Auto,
  Markowitz,
  LowFirst,
  HighFirst,
};

enum class DirectGroupOrdering {
  Auto,
  ExactJet,
  RankSector,
  SectorRank,
};

enum class DirectRankOrdering {
  LowFirst,
  HighFirst,
};

[[nodiscard]] constexpr AnsatzDotOrdering
resolve_ansatz_dot_ordering(AnsatzDotOrdering requested, bool multi_layer,
                            std::size_t expansion_rounds) noexcept
{
  if (requested != AnsatzDotOrdering::Auto) return requested;
  return !multi_layer && expansion_rounds != 0 ? AnsatzDotOrdering::HighFirst
                                               : AnsatzDotOrdering::LowFirst;
}

struct ReductionOptions {
  NumeratorReductionStrategy numerator_strategy = NumeratorReductionStrategy::Projected;
  ReplayOrientationPreference replay_orientation = ReplayOrientationPreference::Auto;
  AnsatzDotOrdering ansatz_dot_ordering = AnsatzDotOrdering::Auto;
  bool force_direct_positive_targets = false;
  unsigned direct_pinched_dot_halo = 1;
  DirectGroupOrdering direct_group_ordering = DirectGroupOrdering::Auto;
  DirectRankOrdering direct_rank_ordering = DirectRankOrdering::LowFirst;
  // Internal provenance: global selection already checked master independence.
  bool master_basis_globally_selected = false;
};
