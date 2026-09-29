#pragma once

#include "core/Config.hpp"
#include "reduction/ReductionProgress.hpp"

#include <vector>

namespace basis {

// Complete a required preferred set against an already selected physical basis.
[[nodiscard]] std::vector<Integral>
complete_preferred_basis(const Config& config, ReductionProgressCallback progress = {});

} // namespace basis
