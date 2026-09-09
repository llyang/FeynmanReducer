#include "SectorDimensionCounter.hpp"
#include "masters/detail/SingularProbeSupport.hpp"

#include "core/ParallelForExecutor.hpp"
#include "core/ProbeValues.hpp"
#include "masters/detail/SingularProcess.hpp"
#include "topology/SectorPolynomial.hpp"
#include "topology/SectorUtils.hpp"

#include <sys/wait.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace sector_count {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
constexpr auto primes = masters::detail::singular_probe_primes;

std::string script(const MasterFinderConfig& config, std::uint32_t sector,
                   Method method, Probe& probe, std::size_t round, std::size_t p,
                   std::size_t task_id)
{
  std::ostringstream out;
  probe.prime = primes[p];
  const auto point = p + 2 * round;
  probe.kinematics =
      masters::detail::singular_kinematics(config.kinematic_parameters.size(), point);
  for (std::size_t j = 0; j < config.propagator_count; ++j)
    if (method == Method::Regulated)
      probe.regulators.push_back(probe_values::field_value(
          probe.prime, config.kinematic_parameters.size() +
                           config.propagator_slots.at(j), point));
  out << "ring r" << p << '=' << probe.prime << ",(h";
  for (unsigned j = 0; j < config.propagator_count; ++j)
    if (sector & (1U << j)) out << ",x" << config.propagator_slots.at(j) + 1;
  out << "),dp;\npoly G="
      << masters::detail::singular_sector_polynomial(config, sector, probe.prime,
                                                     probe.kinematics)
      << ";\n"
      << "if(G==0){print(\"FR_COUNT|" << task_id << "|-2\");}\nelse{\n"
      << "ideal J=1-h*G";
  for (unsigned j = 0; j < config.propagator_count; ++j) {
    if (!(sector & (1U << j))) continue;
    const auto name = "x" + std::to_string(config.propagator_slots.at(j) + 1);
    out << ',';
    if (method == Method::Regulated) out << name << '*';
    out << "diff(G," << name << ')';
    if (method == Method::Regulated) out << '-' << probe.regulators[j] << "*G";
  }
  out << ";\nideal GB=slimgb(J);\nprint(\"FR_COUNT|" << task_id
      << "|\"+string(vdim(GB)));\n}\n";
  out << "kill r" << p << ";\n";
  return out.str();
}

void parse(const std::string& text, const std::unordered_map<std::size_t, Probe*>& probes)
{
  std::istringstream input(text);
  std::string line;
  while (std::getline(input, line)) {
    constexpr std::string_view prefix = "FR_COUNT|";
    if (!line.starts_with(prefix)) continue;
    const auto split = line.find('|', prefix.size());
    if (split == std::string::npos) throw std::runtime_error("malformed count marker");
    std::size_t index = 0;
    std::int64_t dimension = 0;
    const auto a = std::from_chars(line.data() + prefix.size(), line.data() + split, index);
    const auto b = std::from_chars(line.data() + split + 1, line.data() + line.size(), dimension);
    if (a.ec != std::errc{} || a.ptr != line.data() + split ||
        b.ec != std::errc{} || b.ptr != line.data() + line.size() ||
        !probes.contains(index) || dimension < -2 || probes.at(index)->dimension)
      throw std::runtime_error("invalid or duplicate count marker");
    probes.at(index)->dimension = dimension;
  }
  for (const auto& [id, probe] : probes)
    if (!probe->dimension) throw std::runtime_error("missing Singular count marker");
}

void count_batch(const MasterFinderConfig& config, Method method, Batch& batch,
                 std::vector<Row>& rows,
                 const std::unordered_map<std::uint32_t, std::size_t>& row_indices)
{
  std::string source;
  std::unordered_map<std::size_t, Probe*> probes;
  for (const auto& task : batch.tasks) {
    auto& probe = rows[row_indices.at(task.sector)].attempts.back().probes[task.probe];
    const auto id = static_cast<std::size_t>(task.sector) * 2 + task.probe;
    probes.emplace(id, &probe);
    source += script(config, task.sector, method, probe, batch.round, task.probe, id);
  }
  source += "exit;\n";
  const auto started = Clock::now();
  try {
    const auto result = masters::detail::run_singular_process(config.singular_path, source);
    if (!WIFEXITED(result.status) || WEXITSTATUS(result.status) != 0)
      throw std::runtime_error("Singular process failed: " + result.stderr_text);
    parse(result.stdout_text, probes);
  } catch (const std::exception& error) {
    batch.error = error.what();
    // A batch is atomic: never accept partial output from a failed process.
    for (const auto& [id, probe] : probes) {
      probe->dimension.reset();
      probe->error = batch.error;
    }
  }
  batch.process_wall_ms = milliseconds(started);
  if (probes.size() == 1) probes.begin()->second->process_wall_ms = batch.process_wall_ms;
}

} // namespace

std::vector<Batch> make_batches(Method method, std::vector<std::uint32_t> sectors,
                                unsigned threads, std::size_t round)
{
  auto tasks = masters::detail::singular_probe_batches(std::move(sectors), threads,
                                                       method == Method::Regulated);
  std::vector<Batch> batches;
  batches.reserve(tasks.size());
  for (const auto& group : tasks) {
    auto& batch = batches.emplace_back();
    batch.round = round;
    for (const auto& task : group)
      batch.tasks.push_back({task.sector, task.probe});
  }
  return batches;
}

void compute_net_counts(std::vector<Row>& rows)
{
  std::ranges::sort(rows, [](const Row& a, const Row& b) {
    return std::pair{std::popcount(a.sector), a.sector} <
           std::pair{std::popcount(b.sector), b.sector};
  });
  for (std::size_t i = 0; i < rows.size(); ++i) {
    auto& row = rows[i];
    row.net_count.reset();
    if (!row.dimension || *row.dimension < 0) continue;
    auto net = *row.dimension;
    bool complete = true;
    for (std::size_t j = 0; j < i; ++j) {
      const auto& child = rows[j];
      if ((child.sector & row.sector) != child.sector) continue;
      if (!child.net_count) { complete = false; break; }
      const auto n = *child.net_count;
      if ((n > 0 && net < std::numeric_limits<std::int64_t>::min() + n) ||
          (n < 0 && net > std::numeric_limits<std::int64_t>::max() + n))
        throw std::overflow_error("sector net count overflow");
      net -= n;
    }
    if (complete) row.net_count = net;
  }
}

Report count(const MasterFinderConfig& config, Method method,
             std::optional<std::uint32_t> root)
{
  const auto started = Clock::now();
  const auto sectors = SectorUtils(config).enumerate_nonzero_sectors();
  auto report = count_sectors(config, method, sectors, root);
  report.count_wall_ms = milliseconds(started);
  return report;
}

Report count_sectors(const MasterFinderConfig& config, Method method,
                     std::span<const std::uint32_t> nonzero_sectors,
                     std::optional<std::uint32_t> root)
{
  const auto started = Clock::now();
  if (config.threads == 0 || config.singular_path.empty())
    throw std::invalid_argument("threads and Singular path must be set");
  if (config.propagator_count == 0 || config.propagator_count >= 32)
    throw std::invalid_argument("propagator count must be in [1, 31]");
  const auto top = (1U << config.propagator_count) - 1;
  Report report;
  report.method = method;
  report.root_sector = root.value_or(top);
  report.threads = config.threads;
  if (report.root_sector & ~top)
    throw std::invalid_argument("sector mask is outside active propagator slots");
  std::vector<std::uint32_t> sectors(nonzero_sectors.begin(), nonzero_sectors.end());
  if (!std::ranges::is_sorted(sectors) ||
      std::ranges::adjacent_find(sectors) != sectors.end() ||
      std::ranges::any_of(sectors, [top](auto s) { return s == 0 || (s & ~top); }))
    throw std::invalid_argument("invalid nonzero sector inventory");
  std::erase_if(sectors, [&](auto s) { return (s & report.root_sector) != s; });
  report.zero_sector_count = (std::uint64_t{1} << std::popcount(report.root_sector)) - sectors.size();
  // Start the largest sectors first; each prime consumes one shared worker.
  std::ranges::sort(sectors, [](auto a, auto b) {
    if (std::popcount(a) != std::popcount(b)) return std::popcount(a) > std::popcount(b);
    return a < b;
  });
  report.workers = std::min<std::size_t>(2 * sectors.size(), config.threads);
  report.rows.resize(sectors.size());
  std::vector<std::size_t> pending;
  std::unordered_map<std::uint32_t, std::size_t> row_indices;
  std::vector<Clock::time_point> row_started(sectors.size());
  for (std::size_t i = 0; i < sectors.size(); ++i) {
    auto& row = report.rows[i];
    row.sector = sectors[i];
    row_indices.emplace(row.sector, i);
    row.indices.assign(config.integral_count, 0);
    for (unsigned j = 0; j < config.propagator_count; ++j)
      if (row.sector & (1U << j)) row.indices.at(config.propagator_slots.at(j)) = 1;
    pending.push_back(i);
  }
  core::ParallelForExecutor executor(report.workers);
  for (std::size_t round = 0; round < 3 && !pending.empty(); ++round) {
    // Allocate all result slots before dispatch. Workers only write their probe.
    for (auto i : pending) {
      report.rows[i].attempts.emplace_back();
      report.rows[i].attempts.back().probes.resize(primes.size());
    }
    std::vector<std::uint32_t> pending_sectors;
    for (auto i : pending) pending_sectors.push_back(report.rows[i].sector);
    auto batches = make_batches(method, std::move(pending_sectors), config.threads, round);
    const auto offset = report.batches.size();
    for (auto& batch : batches) report.batches.push_back(std::move(batch));
    for (std::size_t i = offset; i < report.batches.size(); ++i)
      for (const auto& task : report.batches[i].tasks)
        report.rows[row_indices.at(task.sector)].attempts.back().probes[task.probe].batch_id = i;
    struct Interval { Clock::time_point start, end; };
    std::vector<Interval> intervals(batches.size());
    executor.run(intervals.size(), [&](std::size_t batch, std::size_t) {
      intervals[batch].start = Clock::now();
      count_batch(config, method, report.batches[offset + batch], report.rows, row_indices);
      intervals[batch].end = Clock::now();
    });
    std::vector<std::size_t> retry;
    for (std::size_t j = 0; j < pending.size(); ++j) {
      const auto i = pending[j];
      auto& row = report.rows[i];
      auto& attempt = row.attempts.back();
      const auto& a = intervals[attempt.probes[0].batch_id - offset];
      const auto& b = intervals[attempt.probes[1].batch_id - offset];
      const auto begin = std::min(a.start, b.start);
      const auto end = std::max(a.end, b.end);
      if (round == 0) row_started[i] = begin;
      attempt.wall_ms = std::chrono::duration<double, std::milli>(end - begin).count();
      row.wall_ms = std::chrono::duration<double, std::milli>(end - row_started[i]).count();
      if (attempt.probes[0].process_wall_ms && attempt.probes[1].process_wall_ms)
        attempt.process_wall_ms = *attempt.probes[0].process_wall_ms +
                                  *attempt.probes[1].process_wall_ms;
      for (const auto& probe : attempt.probes) {
        if (!probe.error.empty()) {
          if (!attempt.error.empty()) attempt.error += "; ";
          attempt.error += "prime " + std::to_string(probe.prime) + ": " + probe.error;
        }
      }
      if (!attempt.error.empty()) continue; // Process/protocol failures never retry.
      const auto first = *attempt.probes[0].dimension;
      const auto second = *attempt.probes[1].dimension;
      if (first == second && (first >= 0 || (method == Method::Critical && first == -1))) {
        row.dimension = first;
        row.status = first == -1 ? "nonisolated" : "ok";
      } else {
        attempt.error = "inconsistent or degenerate samples";
        retry.push_back(i);
      }
    }
    pending = std::move(retry);
  }
  for (const auto& row : report.rows)
    if (!row.dimension) report.complete = false;
  if (method == Method::Regulated) compute_net_counts(report.rows);
  else std::ranges::sort(report.rows, {}, &Row::sector);
  report.count_wall_ms = milliseconds(started);
  return report;
}

} // namespace sector_count
