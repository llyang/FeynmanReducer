#pragma once

#include "core/Config.hpp"

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sector_count {
enum class Method { Regulated, Critical };
struct Probe {
  std::uint32_t prime = 0;
  std::vector<std::uint32_t> kinematics;
  std::vector<std::uint64_t> regulators;
  std::optional<std::int64_t> dimension;
  std::optional<double> process_wall_ms; // Only for singleton batches.
  std::size_t batch_id = 0;
  std::string error;
};
struct Attempt {
  std::vector<Probe> probes;
  std::optional<double> process_wall_ms; // Only if both probes have exclusive processes.
  double wall_ms = 0; // First task start to last task completion.
  std::string error;
};
struct Row {
  std::uint32_t sector = 0;
  std::vector<int> indices;
  std::optional<std::int64_t> dimension;
  std::optional<std::int64_t> net_count;
  std::string status = "unresolved";
  std::vector<Attempt> attempts;
  double wall_ms = 0;
};
struct BatchTask {
  std::uint32_t sector = 0;
  std::size_t probe = 0;
};
struct Batch {
  std::size_t round = 0;
  std::vector<BatchTask> tasks;
  double process_wall_ms = 0;
  std::string error;
};
// Deterministic scheduler; sectors must be unique.
[[nodiscard]] std::vector<Batch> make_batches(Method method,
    std::vector<std::uint32_t> sectors, unsigned threads, std::size_t round = 0);
struct Report {
  Method method = Method::Regulated;
  std::uint32_t root_sector = 0;
  unsigned threads = 0;
  std::size_t workers = 0;
  std::uint64_t zero_sector_count = 0; // Includes the empty sector.
  bool complete = true;
  double count_wall_ms = 0;
  std::vector<Row> rows;
  std::vector<Batch> batches;
};
[[nodiscard]] Report count(const MasterFinderConfig& config, Method method,
                           std::optional<std::uint32_t> root = std::nullopt);
// Internal entry for a complete, sorted labelled inventory from the same topology.
[[nodiscard]] Report count_sectors(const MasterFinderConfig& config, Method method,
                                   std::span<const std::uint32_t> nonzero_sectors,
                                   std::optional<std::uint32_t> root = std::nullopt);
// Möbius inversion on the original labelled sector poset. Negative values are
// valid Euler net counts, not physical basis dimensions. No symmetry merging.
void compute_net_counts(std::vector<Row>& rows);
} // namespace sector_count
