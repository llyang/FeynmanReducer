#pragma once
#include "core/ReductionResult.hpp"
#include "reduction/ReductionProgress.hpp"

namespace reduction::detail {
void check_d_separation(ReductionResult& result, BasisSelectionPolicy policy,
                        const ReductionProgressCallback& progress = {});
}
