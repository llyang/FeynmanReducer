#pragma once

#include "core/Config.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace integral_layout {

[[nodiscard]] std::vector<std::uint32_t>
default_variable_slots(const TopologyConfig& topology);

[[nodiscard]] std::vector<std::uint32_t> active_variables(std::uint32_t sector,
                                                          std::size_t variable_count);

void validate(const TopologyConfig& topology, const Integral& integral);

[[nodiscard]] Integral project_active(const TopologyConfig& topology,
                                      const Integral& integral);

[[nodiscard]] Integral expand_active(const TopologyConfig& topology,
                                     const Integral& integral);

[[nodiscard]] Integral sector_corner(const TopologyConfig& topology,
                                     std::uint32_t sector);

} // namespace integral_layout
