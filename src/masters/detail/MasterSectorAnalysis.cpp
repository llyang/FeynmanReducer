#include "masters/detail/MasterSectorAnalysis.hpp"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace masters::detail {
std::vector<SectorDimensionComparison>
compare_sector_dimensions(const MasterFinderConfig& config,
                          std::span<const CriticalSectorBasis> critical,
                          const sector_count::Report& regulated)
{
  if (!config.symmetry) throw std::invalid_argument("missing symmetry analysis");
  std::unordered_map<std::uint32_t, int> dimensions;
  for (const auto& basis : critical) {
    if (basis.nonisolated || basis.critical_dimension < 0)
      throw std::invalid_argument(
          "dimension comparison requires isolated critical data");
    if (!dimensions.emplace(basis.sector, basis.critical_dimension).second)
      throw std::logic_error("duplicate critical sector");
  }
  // Euler inversion is on labelled sectors, before either symmetry quotient.
  for (const auto& symmetry_class : config.symmetry->sector_classes) {
    const auto representative = dimensions.find(symmetry_class.representative);
    if (representative == dimensions.end())
      throw std::logic_error("symmetry representative has no critical dimension");
    const int dimension = representative->second;
    for (const auto& relation : symmetry_class.relations)
      if (!dimensions.emplace(relation.target_sector, dimension).second)
        throw std::logic_error("duplicate labelled critical sector");
  }
  if (dimensions.size() != regulated.rows.size())
    throw std::logic_error("critical and regulated sector coverage differs");
  std::unordered_set<std::uint32_t> seen;
  std::vector<SectorDimensionComparison> result;
  result.reserve(regulated.rows.size());
  for (const auto& row : regulated.rows) {
    const auto found = dimensions.find(row.sector);
    if (found == dimensions.end() || !seen.insert(row.sector).second)
      throw std::logic_error("regulated sector has no unique critical dimension");
    SectorDimensionComparison comparison;
    comparison.sector = row.sector;
    comparison.critical_dimension = found->second;
    comparison.regulated_total = row.dimension;
    comparison.regulated_net = row.net_count;
    if (!row.dimension || *row.dimension < 0 || !row.net_count) {
      comparison.detail = "net count is unresolved";
      if (!row.attempts.empty() && !row.attempts.back().error.empty())
        comparison.detail = row.attempts.back().error;
    } else {
      comparison.status = found->second == *row.net_count
                              ? DimensionComparisonStatus::Match
                              : DimensionComparisonStatus::Mismatch;
    }
    result.push_back(std::move(comparison));
  }
  return result;
}
} // namespace masters::detail
