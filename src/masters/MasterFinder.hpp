#pragma once

#include "core/Config.hpp"
#include "core/MasterCandidates.hpp"

#include <vector>

namespace masters {

// Discovers deterministic local candidates. Positive-dimensional sectors are
// returned for relation selection by the reduction kernel.
[[nodiscard]] MasterCandidateSet
find_master_candidates(const MasterFinderConfig& config);

// Resolves a positive-dimensional candidate pool with the deterministic global
// quotient-rank selector.  The caller can reuse an already discovered pool
// without repeating the Singular sector scan.
[[nodiscard]] std::vector<Integral>
select_global_master_basis(const MasterFinderConfig& config,
                           const MasterCandidateSet& candidates);

// Standalone wrapper used by find_masters and callers that need a final basis. Global
// candidates are selected after quotient closure, including local residual expansion.
[[nodiscard]] std::vector<Integral>
find_master_integrals(const MasterFinderConfig& config);

} // namespace masters
