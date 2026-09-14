#pragma once

#include <span>

#include "core/Config.hpp"
#include "core/MasterCandidates.hpp"
#include "core/ReductionResult.hpp"
#include "reduction/ReductionOptions.hpp"
#include "reduction/ReductionProgress.hpp"

[[nodiscard]] ReductionResult perform_reduction(const Config& config,
                                                ReductionProgressCallback progress = {},
                                                ReductionOptions options = {});

// Ownership-taking entry used by the CLI after configuration/master discovery.
// The distinct name avoids ambiguous braced calls with the candidate-set overload.
[[nodiscard]] ReductionResult
perform_reduction_owned(Config config, ReductionProgressCallback progress = {},
                        ReductionOptions options = {});

[[nodiscard]] ReductionResult perform_reduction(Config config,
                                                masters::MasterCandidateSet candidates,
                                                ReductionProgressCallback progress = {},
                                                ReductionOptions options = {});

namespace reduction::research {
// Fixed-basis reconstruction for independent change-of-basis experiments. The
// caller supplies the verified basis and preserves its relation-source anchors.
[[nodiscard]] ReductionResult reconstruct_fixed_basis(
    Config config, std::span<const std::uint32_t> relation_source_sectors,
    ReductionProgressCallback progress = {}, ReductionOptions options = {});
}
