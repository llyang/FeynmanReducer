#pragma once

#include "core/Config.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace masters::detail {

[[nodiscard]] std::vector<Integral>
select_global_master_basis(const MasterFinderConfig& config,
                           std::span<const Integral> ordered_candidates,
                           std::span<const std::uint32_t> relation_source_sectors);

} // namespace masters::detail
