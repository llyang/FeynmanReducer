#include "topology/IntegralLayout.hpp"

#include <bit>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace integral_layout {

std::vector<std::uint32_t> default_variable_slots(const TopologyConfig& topology)
{
  if (!topology.propagator_slots.empty()) return topology.propagator_slots;
  std::vector<std::uint32_t> slots(topology.propagator_count);
  std::iota(slots.begin(), slots.end(), 0U);
  return slots;
}

std::vector<std::uint32_t> active_variables(std::uint32_t sector,
                                            std::size_t variable_count)
{
  constexpr auto sector_bits = std::numeric_limits<std::uint32_t>::digits;
  if (variable_count > sector_bits) {
    throw std::invalid_argument("sector mask cannot represent all variables");
  }
  std::vector<std::uint32_t> result;
  result.reserve(static_cast<std::size_t>(std::popcount(sector)));
  for (std::size_t variable = 0; variable < variable_count; ++variable) {
    if (((sector >> variable) & 1U) != 0) {
      result.push_back(static_cast<std::uint32_t>(variable));
    }
  }
  return result;
}

void validate(const TopologyConfig& topology, const Integral& integral)
{
  if (integral.indices.size() != topology.integral_count) {
    throw std::invalid_argument("integral length does not match propagator+ISP layout");
  }
  if (topology.top_sector.size() != topology.integral_count ||
      topology.propagator_slots.size() != topology.propagator_count) {
    throw std::invalid_argument("topology integral layout is inconsistent");
  }
  for (std::size_t slot = 0; slot < topology.top_sector.size(); ++slot) {
    if (topology.top_sector[slot] == 0 && integral.indices[slot] > 0) {
      throw std::invalid_argument("positive ISP indices are not supported");
    }
  }
}

Integral project_active(const TopologyConfig& topology, const Integral& integral)
{
  validate(topology, integral);
  Integral result;
  result.indices.reserve(topology.propagator_slots.size());
  for (const auto slot : topology.propagator_slots) {
    if (slot >= integral.indices.size()) {
      throw std::invalid_argument("active propagator slot is out of range");
    }
    result.indices.push_back(integral.indices[slot]);
  }
  return result;
}

Integral expand_active(const TopologyConfig& topology, const Integral& integral)
{
  if (integral.indices.size() != topology.propagator_count ||
      topology.propagator_slots.size() != topology.propagator_count) {
    throw std::invalid_argument("active integral length does not match the topology");
  }
  Integral result;
  result.indices.assign(topology.integral_count, 0);
  for (std::size_t variable = 0; variable < topology.propagator_slots.size();
       ++variable) {
    const auto slot = topology.propagator_slots[variable];
    if (slot >= result.indices.size()) {
      throw std::invalid_argument("active propagator slot is out of range");
    }
    result.indices[slot] = integral.indices[variable];
  }
  return result;
}

Integral sector_corner(const TopologyConfig& topology, std::uint32_t sector)
{
  constexpr auto sector_bits = std::numeric_limits<std::uint32_t>::digits;
  if (topology.propagator_count > sector_bits) {
    throw std::invalid_argument("sector mask cannot represent all propagators");
  }
  if (topology.propagator_count < sector_bits &&
      (sector >> topology.propagator_count) != 0) {
    throw std::invalid_argument("sector contains an inactive propagator bit");
  }
  Integral active_corner;
  active_corner.indices.resize(topology.propagator_count);
  for (std::size_t variable = 0; variable < topology.propagator_count; ++variable) {
    active_corner.indices[variable] = ((sector >> variable) & 1U) != 0 ? 1 : 0;
  }
  return expand_active(topology, active_corner);
}

} // namespace integral_layout
