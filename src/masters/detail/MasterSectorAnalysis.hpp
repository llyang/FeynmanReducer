#pragma once

#include "core/MasterCandidates.hpp"
#include "masters/detail/SectorDimensionCounter.hpp"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace masters::detail {
struct CriticalSectorBasis {
  std::uint32_t sector = 0;
  int critical_dimension = 0; // Before quotienting by internal symmetry.
  bool nonisolated = false;
  std::vector<std::vector<int>> monomials;
  std::optional<std::vector<std::vector<int>>> symmetry_monomials;
};

enum class DimensionComparisonStatus { Match, Mismatch, Unresolved };
struct SectorDimensionComparison {
  std::uint32_t sector = 0;
  int critical_dimension = 0;
  std::optional<std::int64_t> regulated_total;
  std::optional<std::int64_t> regulated_net;
  DimensionComparisonStatus status = DimensionComparisonStatus::Unresolved;
  std::string detail;
};

// Transient discovery data, never retained by a reduction kernel. A missing
// regulated report means that the nonisolated/global route skipped counting.
struct MasterSectorAnalysis {
  std::vector<std::uint32_t> nonzero_sectors;
  std::vector<CriticalSectorBasis> critical;
  // Nonisolated critical data, or a complete isolated dimension mismatch.
  bool needs_global_selection = false;
  std::optional<sector_count::Report> regulated;
  std::vector<SectorDimensionComparison> comparisons;
};

[[nodiscard]] std::vector<SectorDimensionComparison>
compare_sector_dimensions(const MasterFinderConfig& config,
                          std::span<const CriticalSectorBasis> critical,
                          const sector_count::Report& regulated);
[[nodiscard]] MasterSectorAnalysis
analyze_master_sectors(const MasterFinderConfig& config);
// Consumes analysis only after checking every comparison. Incomplete analysis
// must never publish even a partial physical basis. Complete mismatches produce
// a global-selection candidate pool with a one-dot border in affected sectors.
[[nodiscard]] MasterCandidateSet
assemble_master_candidates(const MasterFinderConfig& config,
                           MasterSectorAnalysis analysis);
} // namespace masters::detail
