#pragma once

#include "core/Config.hpp"

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

/// Topology-level non-zero-sector classifier shared by symmetry and reduction.
class SectorUtils {
public:
  explicit SectorUtils(const TopologyConfig& config);
  SectorUtils(const TopologyConfig& config,
              std::span<const std::uint32_t> variable_slots);

  [[nodiscard]] std::uint32_t sector_from_powers(std::span<const int> powers) const;
  [[nodiscard]] bool is_valid_sector(std::uint32_t sector) const;
  [[nodiscard]] std::vector<std::uint32_t> enumerate_nonzero_sectors() const;

private:
  [[nodiscard]] bool compute_valid_sector(std::uint32_t sector) const;

  const TopologyConfig& config_;
  std::vector<std::uint32_t> variable_slots_;
  std::vector<int> sector_bit_by_slot_;
  std::uint32_t propagator_mask_ = 0;
  std::vector<std::uint32_t> term_support_masks_;
  mutable std::unordered_map<std::uint32_t, bool> valid_sectors_;
};
