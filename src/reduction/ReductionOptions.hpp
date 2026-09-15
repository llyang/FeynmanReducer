#pragma once

#include "basis/DSeparatingSearchStrategy.hpp"

#include <cstddef>

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

[[nodiscard]] constexpr AnsatzDotOrdering
resolve_ansatz_dot_ordering(AnsatzDotOrdering requested, bool multi_layer,
                            std::size_t expansion_rounds) noexcept
{
  if (requested != AnsatzDotOrdering::Auto) return requested;
  return !multi_layer && expansion_rounds != 0 ? AnsatzDotOrdering::HighFirst
                                               : AnsatzDotOrdering::LowFirst;
}

enum class DSeparatingKernelStrategy {
  Auto,
  SharedOracle,
  RebuildFinalBasis,
  RecompactFinalBasis,
  ReuseProvisionalFinalBasis,
  ReuseProvisionalReselectFinalBasis,
};

[[nodiscard]] constexpr bool
uses_provisional_checkpoint(DSeparatingKernelStrategy strategy)
{
  return strategy == DSeparatingKernelStrategy::ReuseProvisionalFinalBasis ||
         strategy == DSeparatingKernelStrategy::ReuseProvisionalReselectFinalBasis;
}

[[nodiscard]] constexpr bool uses_recompact_source(DSeparatingKernelStrategy strategy)
{
  return strategy == DSeparatingKernelStrategy::RecompactFinalBasis ||
         uses_provisional_checkpoint(strategy);
}

enum class DSeparatingSharedOptimization {
  None,
  TrimKernel,
};

enum class TargetReplayPolicy { Exact = 0, TargetRows = 1, TargetRowsCached = 2 };

// Master-oriented grouping policy; no extra replay cache.
enum class MasterReplayGrouping { Exact, TargetRows, MasterColumns };

struct ReductionOptions {
  ReplayOrientationPreference replay_orientation = ReplayOrientationPreference::Auto;
  AnsatzDotOrdering ansatz_dot_ordering = AnsatzDotOrdering::Auto;
  // Internal provenance: global selection already checked master independence.
  bool master_basis_globally_selected = false;
  // Auto resolves only at the reduction entrypoint; low-level prepare stays shared.
  // Internal search policy override; not exposed by YAML or the production CLI.
  basis::DSeparatingSearchStrategy d_separating_search =
      basis::DSeparatingSearchStrategy::SingleSlotThenScored;
  DSeparatingKernelStrategy d_separating_kernel = DSeparatingKernelStrategy::Auto;
  TargetReplayPolicy d_separating_replay = TargetReplayPolicy::TargetRowsCached;
  // Default-basis replay policy; independent of D-separating search options.
  TargetReplayPolicy default_replay = TargetReplayPolicy::TargetRows;
  MasterReplayGrouping default_master_replay = MasterReplayGrouping::MasterColumns;
  DSeparatingSharedOptimization d_separating_shared =
      DSeparatingSharedOptimization::None;
};
