#pragma once

#include "core/Config.hpp"
#include "core/ReductionResult.hpp"
#include "reduction/ReductionProgress.hpp"
#include <memory>

class BlackBoxFeynman;

namespace reduction::detail {
// Consumes a prepared, validated kernel after the caller has configured replay.
// Shares scale selection, FireFly reconstruction and final checks with production.
[[nodiscard]] ReductionResult
reconstruct_prepared_kernel(Config& config, std::unique_ptr<BlackBoxFeynman> kernel,
                            ReductionProgressCallback progress);
} // namespace reduction::detail
