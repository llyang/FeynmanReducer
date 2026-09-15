#include "DSeparatingBasisSearch.hpp"

#include "NativeFixedBasisOracle.hpp"
#include "RowBatchExecution.hpp"
#include "UnivariateSeparation.hpp"
#include "core/IntegralFormatting.hpp"
#include "core/ParallelForExecutor.hpp"
#include "core/ProbeValues.hpp"
#include "masters/detail/MasterIntegralOrdering.hpp"
#include "masters/detail/MonomialPreference.hpp"
#include "reduction/ParameterEvaluation.hpp"
#include "symmetry/Symmetry.hpp"
#include "topology/IntegralLayout.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>

namespace basis {
namespace {

void append_unique(std::vector<Integral>& values, const Integral& value)
{
  if (std::ranges::find(values, value) == values.end()) {
    values.push_back(value);
  }
}

std::vector<std::vector<int>> dot_monomials(std::size_t variables,
                                            unsigned maximum_dots)
{
  std::vector<std::vector<int>> result;
  std::vector<int> current(variables, 0);
  const auto enumerate = [&](auto&& self, std::size_t variable,
                             unsigned remaining) -> void {
    if (variable == variables) {
      result.push_back(current);
      return;
    }
    for (unsigned dots = 0; dots <= remaining; ++dots) {
      current[variable] = static_cast<int>(dots);
      self(self, variable + 1, remaining - dots);
    }
  };
  enumerate(enumerate, 0, maximum_dots);
  std::ranges::sort(result, [](const auto& lhs, const auto& rhs) {
    return masters::detail::monomial_preferred(lhs, rhs);
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

std::size_t find_row(std::span<const Integral> rows, const Integral& integral)
{
  const auto found = std::ranges::find(rows, integral);
  if (found == rows.end())
    throw std::logic_error("integral is absent from oracle batch");
  return static_cast<std::size_t>(found - rows.begin());
}

std::vector<Integral> automatic_screening_targets(const TopologyConfig& topology,
                                                  std::span<const Integral> targets,
                                                  std::size_t limit)
{
  if (limit == 0 || targets.empty()) return {};
  std::map<std::uint32_t, std::vector<Integral>> sectors;
  for (const auto& target : targets)
    sectors[masters::detail::integral_sector(topology, target)].push_back(target);

  std::vector<Integral> result;
  for (auto& [sector, values] : sectors) {
    (void)sector;
    std::ranges::sort(values, [&](const auto& lhs, const auto& rhs) {
      const auto lhs_dots = integral_dot_count(topology, lhs);
      const auto rhs_dots = integral_dot_count(topology, rhs);
      if (lhs_dots != rhs_dots) return lhs_dots > rhs_dots;
      return masters::detail::canonical_master_integral_preferred(topology, lhs, rhs);
    });
    append_unique(result, values.front());
    if (result.size() == limit) return result;
  }
  std::vector<Integral> remaining(targets.begin(), targets.end());
  std::ranges::sort(remaining, [&](const auto& lhs, const auto& rhs) {
    const auto lhs_dots = integral_dot_count(topology, lhs);
    const auto rhs_dots = integral_dot_count(topology, rhs);
    if (lhs_dots != rhs_dots) return lhs_dots > rhs_dots;
    return masters::detail::canonical_master_integral_preferred(topology, lhs, rhs);
  });
  for (const auto& target : remaining) {
    append_unique(result, target);
    if (result.size() == limit) break;
  }
  return result;
}

class BasisSearchProgress {
public:
  BasisSearchProgress(DSeparatingBasisProgressCallback callback,
                      std::size_t interval_seconds, std::size_t maximum_basis_states,
                      std::size_t expected_probes)
      : callback_(std::move(callback)), interval_seconds_(interval_seconds),
        maximum_basis_states_(maximum_basis_states), expected_probes_(expected_probes),
        started_(std::chrono::steady_clock::now())
  {
    if (callback_ && interval_seconds_ != 0) {
      heartbeat_ = std::thread([this] { heartbeat_loop(); });
    }
  }

  BasisSearchProgress(const BasisSearchProgress&) = delete;
  BasisSearchProgress& operator=(const BasisSearchProgress&) = delete;

  ~BasisSearchProgress()
  {
    {
      std::lock_guard lock(state_mutex_);
      stopping_ = true;
    }
    wakeup_.notify_all();
    if (heartbeat_.joinable()) heartbeat_.join();
  }

  void set_stage(std::string stage, std::optional<unsigned> depth = std::nullopt)
  {
    std::lock_guard lock(state_mutex_);
    stage_ = std::move(stage);
    message_.clear();
    depth_ = depth;
  }

  void event(std::string stage, std::string message,
             std::optional<unsigned> depth = std::nullopt)
  {
    {
      std::lock_guard lock(state_mutex_);
      stage_ = std::move(stage);
      message_ = std::move(message);
      depth_ = depth;
    }
    emit(false);
  }

  void set_swaps(std::size_t value) noexcept
  {
    swaps_.store(value, std::memory_order_relaxed);
  }

  void set_basis_states(std::size_t value) noexcept
  {
    basis_states_.store(value, std::memory_order_relaxed);
  }

  void set_rejects(std::size_t rank, std::size_t moving, std::size_t sampling) noexcept
  {
    rank_rejects_.store(rank, std::memory_order_relaxed);
    moving_rejects_.store(moving, std::memory_order_relaxed);
    sampling_rejects_.store(sampling, std::memory_order_relaxed);
  }

  void probe_completed() noexcept
  {
    probes_.fetch_add(1, std::memory_order_relaxed);
  }

private:
  [[nodiscard]] std::string elapsed() const
  {
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - started_)
                             .count();
    const auto hours = seconds / 3600;
    const auto minutes = (seconds / 60) % 60;
    return std::format("{:02}:{:02}:{:02}", hours, minutes, seconds % 60);
  }

  void emit(bool heartbeat)
  {
    if (!callback_) return;
    std::string stage;
    std::string message;
    std::optional<unsigned> depth;
    {
      std::lock_guard lock(state_mutex_);
      stage = stage_;
      message = message_;
      depth = depth_;
    }
    std::string line = std::format("[basis-search {}]{} stage={}", elapsed(),
                                   heartbeat ? " heartbeat" : "", stage);
    if (depth) line += std::format(" depth={}", *depth);
    line += std::format(" swaps={} states={}/{} rank_rejects={} moving_pole_rejects={} "
                        "sampling_rejects={} probes={}/{}",
                        swaps_.load(std::memory_order_relaxed),
                        basis_states_.load(std::memory_order_relaxed),
                        maximum_basis_states_,
                        rank_rejects_.load(std::memory_order_relaxed),
                        moving_rejects_.load(std::memory_order_relaxed),
                        sampling_rejects_.load(std::memory_order_relaxed),
                        probes_.load(std::memory_order_relaxed), expected_probes_);
    if (!message.empty()) line += " " + message;
    std::lock_guard output_lock(output_mutex_);
    callback_(line);
  }

  void heartbeat_loop()
  {
    std::unique_lock lock(state_mutex_);
    while (!wakeup_.wait_for(lock, std::chrono::seconds(interval_seconds_),
                             [this] { return stopping_; })) {
      lock.unlock();
      emit(true);
      lock.lock();
    }
  }

  DSeparatingBasisProgressCallback callback_;
  std::size_t interval_seconds_ = 0;
  std::size_t maximum_basis_states_ = 0;
  std::size_t expected_probes_ = 0;
  std::chrono::steady_clock::time_point started_;
  std::atomic<std::size_t> swaps_{0};
  std::atomic<std::size_t> basis_states_{0};
  std::atomic<std::size_t> rank_rejects_{0};
  std::atomic<std::size_t> moving_rejects_{0};
  std::atomic<std::size_t> sampling_rejects_{0};
  std::atomic<std::size_t> probes_{0};
  std::mutex state_mutex_;
  std::mutex output_mutex_;
  std::condition_variable wakeup_;
  bool stopping_ = false;
  std::string stage_ = "initializing";
  std::string message_;
  std::optional<unsigned> depth_;
  std::thread heartbeat_;
};

struct ProbeKey {
  std::size_t prime = 0;
  std::size_t kinematics = 0;
  std::size_t dimension = 0;

  auto operator<=>(const ProbeKey&) const = default;
};

class NativeBasisProbeDataset {
public:
  struct ProbeLocation {
    std::size_t kinematics = 0;
    std::size_t dimension = 0;
  };

  struct ProbeEvaluation {
    bool failed = false;
    std::vector<std::optional<FieldVector>> conventional;

    [[nodiscard]] explicit operator bool() const noexcept
    {
      return !failed;
    }

    [[nodiscard]] const FieldVector& at(std::size_t row) const
    {
      if (row >= conventional.size() || !conventional[row])
        throw std::logic_error("native basis probe row was not requested");
      return *conventional[row];
    }
  };

  NativeBasisProbeDataset(Config config, std::vector<Integral> initial_basis,
                          std::vector<std::uint64_t> primes,
                          std::size_t kinematic_points,
                          std::size_t maximum_dimension_samples,
                          std::span<const std::uint32_t> relation_source_sectors,
                          ReductionOptions reduction_options,
                          ReductionProgressCallback native_progress,
                          BasisSearchProgress& progress)
      : primes_(std::move(primes)), kinematic_points_(kinematic_points),
        maximum_dimension_samples_(maximum_dimension_samples),
        oracle_(NativeFixedBasisOracle::prepare(
            std::move(config), std::move(initial_basis), std::move(native_progress),
            ReplayOrientationPreference::Target, relation_source_sectors,
            reduction_options)),
        executor_(oracle_->config().threads), progress_(progress)
  {
    if (primes_.empty() || kinematic_points_ == 0 || maximum_dimension_samples_ == 0)
      throw std::invalid_argument("native basis probe grid must be nonempty");
  }

  [[nodiscard]] const Config& config() const noexcept
  {
    return oracle_->config();
  }

  [[nodiscard]] std::span<const std::uint64_t> primes() const noexcept
  {
    return primes_;
  }

  [[nodiscard]] std::vector<std::uint64_t> kinematics(std::size_t prime,
                                                      std::size_t point) const
  {
    std::vector<std::uint64_t> result(config().kinematic_parameters.size());
    for (std::size_t parameter = 0; parameter < result.size(); ++parameter) {
      result[parameter] =
          probe_values::field_value(primes_.at(prime), parameter, point);
    }
    return result;
  }

  [[nodiscard]] std::uint64_t dimension(std::size_t prime, std::size_t sample) const
  {
    return probe_values::field_value(primes_.at(prime),
                                     config().kinematic_parameters.size() + 11, sample);
  }

  [[nodiscard]] std::vector<const ProbeEvaluation*>
  evaluations(std::size_t prime, std::span<const ProbeLocation> locations,
              std::span<const std::size_t> requested_rows, bool prefetch = false)
  {
    if (prime >= primes_.size())
      throw std::out_of_range("native basis probe prime is out of range");

    std::vector<std::size_t> rows(requested_rows.begin(), requested_rows.end());
    std::ranges::sort(rows);
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    for (const auto row : rows) {
      if (row >= config().targets.size())
        throw std::out_of_range("requested native basis probe row is out of range");
    }

    struct PendingReplay {
      ProbeKey key;
      ProbeEvaluation* destination = nullptr;
      std::vector<std::size_t> missing_rows;
      detail::RowBatchResult result;
    };
    std::vector<PendingReplay> pending;
    std::vector<ProbeEvaluation*> destinations;
    destinations.reserve(locations.size());
    std::set<ProbeKey> unique_locations;
    for (const auto location : locations) {
      if (location.kinematics >= kinematic_points_ ||
          location.dimension >= maximum_dimension_samples_) {
        throw std::out_of_range("native basis probe index is out of range");
      }
      const ProbeKey key{prime, location.kinematics, location.dimension};
      if (!unique_locations.insert(key).second)
        throw std::invalid_argument("native basis probe batch contains duplicates");
      auto [entry, inserted] = cache_.try_emplace(key);
      if (inserted) entry->second.conventional.resize(config().targets.size());
      auto& destination = entry->second;
      destinations.push_back(&destination);
      if (!prefetch) {
        ++statistics_.logical_requests;
        statistics_.requested_target_rows += rows.size();
      }

      std::vector<std::size_t> missing_rows;
      missing_rows.reserve(rows.size());
      for (const auto row : rows) {
        if (destination.conventional[row]) {
          if (!prefetch) ++statistics_.cached_target_rows;
        } else {
          missing_rows.push_back(row);
        }
      }
      if (missing_rows.empty()) {
        if (!prefetch) ++statistics_.full_cache_hits;
        continue;
      }
      pending.push_back({key, &destination, std::move(missing_rows), {}});
    }

    if (!pending.empty() && (!active_prime_.has_value() || *active_prime_ != prime ||
                             firefly::FFInt::p != primes_[prime])) {
      oracle_->set_prime(primes_[prime]);
      active_prime_ = prime;
    }

    std::map<std::vector<std::size_t>, std::vector<std::size_t>> groups;
    for (std::size_t index = 0; index < pending.size(); ++index)
      groups[pending[index].missing_rows].push_back(index);
    for (const auto& [missing_rows, indices] : groups) {
      const auto prepared = oracle_->prepare_rows(missing_rows);
      executor_.run(indices.size(), [&](std::size_t local, std::size_t) {
        auto& job = pending[indices[local]];
        job.result = detail::evaluate_row_batch(missing_rows, [&](auto requested) {
          const auto selection = requested.size() == missing_rows.size()
                                     ? prepared
                                     : oracle_->prepare_rows(requested);
          return oracle_->evaluate_rows(this->dimension(prime, job.key.dimension),
                                        this->kinematics(prime, job.key.kinematics),
                                        *selection);
        });
      });
    }

    for (auto& job : pending) {
      progress_.probe_completed();
      statistics_.replayed_target_rows += job.result.attempted_rows;
      statistics_.selected_replay_calls += job.result.replay_calls;
      statistics_.selected_compact_outputs += job.result.selected_outputs;
      statistics_.skipped_compact_outputs +=
          job.result.replay_calls * oracle_->reconstructed_outputs().size() -
          job.result.selected_outputs;
      for (std::size_t index = 0; index < job.missing_rows.size(); ++index)
        if (job.result.rows[index])
          job.destination->conventional[job.missing_rows[index]] =
              std::move(job.result.rows[index]);
    }
    // Failure is a view of this request, not a property of the full point.
    // Successfully cached rows remain usable after another row fails.
    for (auto* destination : destinations)
      destination->failed = std::ranges::any_of(
          rows, [&](auto row) { return !destination->conventional[row].has_value(); });

    std::vector<const ProbeEvaluation*> result;
    result.reserve(destinations.size());
    for (const auto* destination : destinations)
      result.push_back(destination);
    return result;
  }

  [[nodiscard]] const DSeparatingBasisProbeStatistics& statistics() const noexcept
  {
    return statistics_;
  }

  [[nodiscard]] std::unique_ptr<NativeFixedBasisOracle> release_oracle()
  {
    return std::move(oracle_);
  }

  template <typename Function> void run_parallel(std::size_t count, Function&& function)
  {
    executor_.run(count, std::forward<Function>(function));
  }

private:
  std::vector<std::uint64_t> primes_;
  std::size_t kinematic_points_ = 0;
  std::size_t maximum_dimension_samples_ = 0;
  std::unique_ptr<NativeFixedBasisOracle> oracle_;
  core::ParallelForExecutor executor_;
  BasisSearchProgress& progress_;
  std::optional<std::size_t> active_prime_;
  std::map<ProbeKey, ProbeEvaluation> cache_;
  DSeparatingBasisProbeStatistics statistics_;
};

enum class SeparationResult { Passed, MovingPole, SamplingFailure };

struct MovingPoleInternal {
  std::size_t target_row = 0;
  std::size_t component = 0;
  std::uint32_t sector = 0;
  std::size_t mixed_degree = 0;
};

struct SeparationDiagnostics {
  SeparationResult result = SeparationResult::SamplingFailure;
  std::optional<MovingPoleInternal> witness;
  std::vector<std::uint32_t> nonzero_output_support;
};

struct PivotProposal {
  std::size_t slot = 0;
  std::size_t candidate = 0;
  bool repairs_primary = false;
  bool pivot_numerator_stable = false;
  std::size_t child_mixed_degree = 0;
  std::size_t pivot_mixed_degree = 0;
  std::size_t pivot_numerator_degree = 0;
};

void rebase_after_basis_swap_in_place(const PrimeField& field,
                                      FieldVector& coefficients,
                                      std::span<const std::uint64_t> pivot_coordinates,
                                      std::size_t slot, std::uint64_t pivot_inverse)
{
  if (coefficients.size() != pivot_coordinates.size() || slot >= coefficients.size())
    throw std::invalid_argument("invalid single-swap rebase dimensions");
  const auto replacement_coefficient =
      field.multiply(coefficients[slot], pivot_inverse);
  coefficients[slot] = replacement_coefficient;
  for (std::size_t component = 0; component < coefficients.size(); ++component) {
    if (component == slot) continue;
    coefficients[component] = field.subtract(
        coefficients[component],
        field.multiply(replacement_coefficient, pivot_coordinates[component]));
  }
}

class TrialBasisChecker {
public:
  TrialBasisChecker(NativeBasisProbeDataset& dataset,
                    std::span<const std::size_t> candidate_rows,
                    std::span<const std::size_t> screening_rows,
                    std::span<const std::size_t> validation_rows,
                    const DSeparatingBasisSearchOptions& options,
                    std::size_t basis_size)
      : dataset_(dataset),
        candidate_rows_(candidate_rows.begin(), candidate_rows.end()),
        screening_rows_(screening_rows.begin(), screening_rows.end()),
        validation_rows_(validation_rows.begin(), validation_rows.end()),
        options_(options), basis_size_(basis_size)
  {}

  [[nodiscard]] bool rank_complete(std::span<const std::size_t> selected,
                                   std::span<const DSeparatingBasisSwap> path)
  {
    if (selected.size() != basis_size_)
      throw std::invalid_argument("trial basis has the wrong size");
    if (path.empty()) return true;
    const std::size_t kinematic_points =
        options_.kinematic_training_points + options_.kinematic_holdout_points;
    const auto rows = swap_rows(path);
    const auto slots = swap_slots(path);
    for (std::size_t prime = 0; prime < dataset_.primes().size(); ++prime) {
      const std::array locations{
          NativeBasisProbeDataset::ProbeLocation{0, 0},
          NativeBasisProbeDataset::ProbeLocation{kinematic_points - 1, 0}};
      const auto data_started = std::chrono::steady_clock::now();
      const auto evaluations = dataset_.evaluations(prime, locations, rows);
      work_timing_.rank_data_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() - data_started)
              .count();
      for (const auto* evaluation : evaluations) {
        if (!*evaluation) return false;
        if (!coordinate_tape(PrimeField(dataset_.primes()[prime]), *evaluation, rows,
                             slots, &work_timing_))
          return false;
        ++analysis_statistics_.avoided_basis_factorizations;
      }
    }
    return true;
  }

  [[nodiscard]] SeparationDiagnostics
  diagnose(std::span<const std::size_t> selected,
           std::span<const DSeparatingBasisSwap> path, bool strict)
  {
    if (selected.size() != basis_size_)
      throw std::invalid_argument("trial basis has the wrong size");
    const auto& target_rows = strict ? validation_rows_ : screening_rows_;
    const std::size_t prime_count = strict ? dataset_.primes().size() : 1;
    const std::size_t kinematic_points =
        strict ? options_.kinematic_training_points + options_.kinematic_holdout_points
               : options_.kinematic_training_points;
    const auto path_rows = swap_rows(path);
    const auto path_slots = swap_slots(path);

    std::optional<std::vector<std::size_t>> reference_degrees;
    std::optional<std::vector<std::vector<bool>>> reference_supports;
    std::optional<std::vector<std::vector<std::pair<std::size_t, std::size_t>>>>
        reference_function_degrees;
    std::vector<std::uint8_t> output_nonzero(target_rows.size() * basis_size_, 0);
    SeparationDiagnostics first_prime_diagnostics;
    for (std::size_t prime = 0; prime < prime_count; ++prime) {
      const PrimeField field(dataset_.primes()[prime]);
      const auto flat_count = target_rows.size() * basis_size_;
      std::vector<std::vector<std::vector<std::optional<FieldVector>>>> rebased(
          kinematic_points,
          std::vector<std::vector<std::optional<FieldVector>>>(
              options_.maximum_dimension_samples,
              std::vector<std::optional<FieldVector>>(target_rows.size())));
      std::vector<std::vector<std::optional<RationalFunction>>> functions(
          flat_count, std::vector<std::optional<RationalFunction>>(kinematic_points));
      std::vector<std::optional<FieldVector>> signatures(flat_count);
      bool complete = false;
      SeparationDiagnostics diagnostics;
      std::size_t previous_sample_count = 0;
      for (std::size_t sample_count = options_.initial_dimension_samples;
           sample_count <= options_.maximum_dimension_samples;) {
        std::vector<std::size_t> active_targets;
        for (std::size_t target = 0; target < target_rows.size(); ++target) {
          bool target_complete = true;
          for (std::size_t component = 0; component < basis_size_; ++component) {
            const auto flat = target * basis_size_ + component;
            if (std::ranges::any_of(functions[flat],
                                    [](const auto& value) { return !value; })) {
              target_complete = false;
              break;
            }
          }
          if (!target_complete) active_targets.push_back(target);
        }
        if (active_targets.empty()) {
          complete = true;
          break;
        }

        std::vector<NativeBasisProbeDataset::ProbeLocation> locations;
        std::vector<std::pair<std::size_t, std::size_t>> location_indices;
        for (std::size_t point = 0; point < kinematic_points; ++point) {
          for (std::size_t dimension = previous_sample_count; dimension < sample_count;
               ++dimension) {
            locations.push_back({point, dimension});
            location_indices.emplace_back(point, dimension);
          }
        }
        std::vector<std::size_t> probe_rows = path_rows;
        for (const auto target : active_targets)
          probe_rows.push_back(target_rows[target]);
        std::ranges::sort(probe_rows);
        probe_rows.erase(std::unique(probe_rows.begin(), probe_rows.end()),
                         probe_rows.end());
        const auto probe_started = std::chrono::steady_clock::now();
        const auto raw_batch = dataset_.evaluations(prime, locations, probe_rows);
        work_timing_.probe_evaluation_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          probe_started)
                .count();
        const auto rebase_started = std::chrono::steady_clock::now();
        dataset_.run_parallel(raw_batch.size(), [&](std::size_t index, std::size_t) {
          const auto [point, dimension] = location_indices[index];
          const auto& raw = *raw_batch[index];
          if (!raw) return;
          const auto tape = coordinate_tape(field, raw, path_rows, path_slots);
          if (!tape) return;
          for (const auto target : active_targets) {
            rebased[point][dimension][target] =
                rebase_after_basis_swaps(field, raw.at(target_rows[target]), *tape);
          }
        });
        work_timing_.pointwise_rebase_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          rebase_started)
                .count();
        analysis_statistics_.avoided_basis_factorizations += raw_batch.size();

        const auto interpolation_started = std::chrono::steady_clock::now();
        std::atomic<std::size_t> attempts{0};
        std::atomic<std::size_t> cache_hits{0};
        std::atomic<std::size_t> fixed_attempts{0};
        std::atomic<std::size_t> thiele_attempts{0};
        std::atomic<std::size_t> exhaustive_fallbacks{0};
        firefly::FFInt::set_new_prime(field.prime());
        dataset_.run_parallel(
            active_targets.size() * basis_size_ * kinematic_points,
            [&](std::size_t job, std::size_t) {
              const auto point = job % kinematic_points;
              const auto component = (job / kinematic_points) % basis_size_;
              const auto target =
                  active_targets[job / (kinematic_points * basis_size_)];
              const auto flat = target * basis_size_ + component;
              FieldVector arguments;
              FieldVector values;
              for (std::size_t dimension = 0; dimension < sample_count; ++dimension) {
                if (!rebased[point][dimension][target]) continue;
                arguments.push_back(dataset_.dimension(prime, dimension));
                values.push_back((*rebased[point][dimension][target])[component]);
              }
              if (arguments.size() <= options_.dimension_holdouts) return;
              ++attempts;
              if (functions[flat][point] &&
                  validates_function(field, *functions[flat][point], arguments,
                                     values)) {
                ++cache_hits;
                return;
              }
              functions[flat][point].reset();
              if (prime != 0 && reference_function_degrees) {
                const auto [numerator_degree, denominator_degree] =
                    (*reference_function_degrees)[flat][point];
                ++fixed_attempts;
                functions[flat][point] = interpolate_rational_fixed_degrees(
                    field, arguments, values, numerator_degree, denominator_degree,
                    options_.dimension_holdouts);
              }
              if (!functions[flat][point]) {
                ++thiele_attempts;
                functions[flat][point] = interpolate_rational_thiele_monic(
                    field, arguments, values, options_.dimension_holdouts);
              }
              if (!functions[flat][point] &&
                  sample_count == options_.maximum_dimension_samples) {
                ++exhaustive_fallbacks;
                const auto maximum_degree =
                    arguments.size() - options_.dimension_holdouts - 1;
                functions[flat][point] =
                    interpolate_rational_monic(field, arguments, values, maximum_degree,
                                               options_.dimension_holdouts);
              }
            });
        analysis_statistics_.rational_interpolation_attempts += attempts.load();
        analysis_statistics_.rational_interpolation_cache_hits += cache_hits.load();
        analysis_statistics_.fixed_degree_attempts += fixed_attempts.load();
        analysis_statistics_.thiele_attempts += thiele_attempts.load();
        analysis_statistics_.exhaustive_fallbacks += exhaustive_fallbacks.load();
        work_timing_.rational_interpolation_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          interpolation_started)
                .count();

        diagnostics = {.result = SeparationResult::Passed,
                       .witness = std::nullopt,
                       .nonzero_output_support = {}};
        const auto signature_started = std::chrono::steady_clock::now();
        for (std::size_t flat = 0; flat < flat_count; ++flat) {
          if (std::ranges::any_of(functions[flat], [](const auto& value) {
                return !value.has_value();
              })) {
            continue;
          }
          std::optional<FieldVector> reference;
          std::vector<FieldVector> denominators;
          for (const auto& function : functions[flat]) {
            denominators.push_back(function->denominator);
            if (function->denominator.size() <= 1) continue;
            if (!reference) {
              reference = function->denominator;
            } else if (*reference != function->denominator) {
              if (!diagnostics.witness) {
                const auto target = flat / basis_size_;
                const auto component = flat % basis_size_;
                diagnostics.witness = MovingPoleInternal{
                    .target_row = target_rows[target],
                    .component = component,
                    .sector = masters::detail::integral_sector(
                        dataset_.config(),
                        dataset_.config().targets[target_rows[target]]),
                    .mixed_degree = moving_polynomial_degree(field, denominators)};
              }
              break;
            }
          }
        }
        work_timing_.signature_comparison_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          signature_started)
                .count();
        if (diagnostics.witness) {
          diagnostics.result = SeparationResult::MovingPole;
          return diagnostics;
        }
        std::size_t completed_now = 0;
        bool sampling_incomplete = false;
        for (std::size_t target = 0; target < target_rows.size(); ++target) {
          bool target_complete = true;
          for (std::size_t component = 0; component < basis_size_; ++component) {
            const auto flat = target * basis_size_ + component;
            if (std::ranges::any_of(functions[flat],
                                    [](const auto& value) { return !value; })) {
              target_complete = false;
              break;
            }
          }
          if (target_complete)
            ++completed_now;
          else
            sampling_incomplete = true;
        }
        analysis_statistics_.completed_target_rows =
            std::max(analysis_statistics_.completed_target_rows, completed_now);
        if (!sampling_incomplete) {
          complete = true;
          if (strict) {
            for (std::size_t flat = 0; flat < flat_count; ++flat) {
              for (const auto& function : functions[flat]) {
                if (function &&
                    std::ranges::any_of(function->numerator, [](std::uint64_t value) {
                      return value != 0;
                    })) {
                  output_nonzero[flat] = 1;
                  break;
                }
              }
            }
          }
          for (std::size_t flat = 0; flat < flat_count; ++flat) {
            for (const auto& function : functions[flat])
              if (function && function->denominator.size() > 1) {
                signatures[flat] = function->denominator;
                break;
              }
          }
          break;
        }
        if (sample_count == options_.maximum_dimension_samples) break;
        previous_sample_count = sample_count;
        sample_count = std::min(options_.maximum_dimension_samples,
                                sample_count + options_.dimension_sample_batch);
      }
      if (!complete) {
        diagnostics.result = SeparationResult::SamplingFailure;
        return diagnostics;
      }
      diagnostics.result = SeparationResult::Passed;
      if (prime == 0) first_prime_diagnostics = diagnostics;
      std::vector<std::size_t> degrees;
      std::vector<std::vector<bool>> supports;
      degrees.reserve(signatures.size());
      supports.reserve(signatures.size());
      for (const auto& signature : signatures) {
        degrees.push_back(signature ? signature->size() - 1 : 0);
        std::vector<bool> support;
        if (signature) {
          support.reserve(signature->size());
          for (const auto coefficient : *signature)
            support.push_back(coefficient != 0);
        } else {
          support.push_back(true);
        }
        supports.push_back(std::move(support));
      }
      if (!reference_degrees) {
        std::vector<std::vector<std::pair<std::size_t, std::size_t>>> function_degrees(
            flat_count,
            std::vector<std::pair<std::size_t, std::size_t>>(kinematic_points));
        for (std::size_t flat = 0; flat < flat_count; ++flat)
          for (std::size_t point = 0; point < kinematic_points; ++point) {
            const auto& function = *functions[flat][point];
            function_degrees[flat][point] = {function.numerator.size() - 1,
                                             function.denominator.size() - 1};
          }
        reference_degrees = std::move(degrees);
        reference_supports = std::move(supports);
        reference_function_degrees = std::move(function_degrees);
      } else if (*reference_degrees != degrees || *reference_supports != supports) {
        return {.result = SeparationResult::SamplingFailure,
                .witness = std::nullopt,
                .nonzero_output_support = {}};
      }
    }
    if (strict) {
      for (std::size_t position = 0; position < output_nonzero.size(); ++position)
        if (output_nonzero[position])
          first_prime_diagnostics.nonzero_output_support.push_back(
              static_cast<std::uint32_t>(position));
    }
    return first_prime_diagnostics;
  }

  [[nodiscard]] std::vector<PivotProposal>
  propose_pivots(std::span<const std::size_t> selected,
                 std::span<const DSeparatingBasisSwap> path,
                 const SeparationDiagnostics& diagnostics,
                 std::span<const Integral> pool, std::span<const unsigned> dot_counts)
  {
    if (!diagnostics.witness) return {};
    const auto witness_slot = diagnostics.witness->component;
    if (witness_slot >= selected.size()) return {};
    const auto slot = witness_slot;
    const auto sector =
        masters::detail::integral_sector(dataset_.config(), pool[selected[slot]]);
    const PrimeField field(dataset_.primes().front());
    const auto point_count = options_.kinematic_training_points;
    // Pivot coordinates can be more complicated in D than the first bad
    // coefficient.  Populate the shared probe cache through the configured
    // maximum once, then score every candidate from the same complete grid.
    const auto sample_count = options_.maximum_dimension_samples;

    const auto path_rows = swap_rows(path);
    const auto path_slots = swap_slots(path);

    std::set<std::size_t> selected_set(selected.begin(), selected.end());
    std::vector<std::size_t> proposal_candidates;
    std::vector<std::size_t> requested_rows;
    for (std::size_t candidate = 0; candidate < pool.size(); ++candidate) {
      if (selected_set.contains(candidate) ||
          masters::detail::integral_sector(dataset_.config(), pool[candidate]) !=
              sector) {
        continue;
      }
      proposal_candidates.push_back(candidate);
      requested_rows.push_back(candidate_rows_.at(candidate));
    }
    if (proposal_candidates.empty()) return {};
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    for (std::size_t candidate = 0; candidate < proposal_candidates.size(); ++candidate)
      pairs.emplace_back(candidate, witness_slot);
    raw_pairs_ += pairs.size();
    const auto witness_local = requested_rows.size();
    requested_rows.push_back(diagnostics.witness->target_row);
    std::vector<std::size_t> probe_rows = path_rows;
    probe_rows.insert(probe_rows.end(), requested_rows.begin(), requested_rows.end());
    std::ranges::sort(probe_rows);
    probe_rows.erase(std::unique(probe_rows.begin(), probe_rows.end()),
                     probe_rows.end());

    std::vector<std::vector<FieldMatrix>> coordinates(
        point_count, std::vector<FieldMatrix>(sample_count));
    std::vector<NativeBasisProbeDataset::ProbeLocation> locations;
    locations.reserve(point_count * sample_count);
    for (std::size_t point = 0; point < point_count; ++point)
      for (std::size_t dimension = 0; dimension < sample_count; ++dimension)
        locations.push_back({point, dimension});
    const auto probe_started = std::chrono::steady_clock::now();
    const auto raw_batch = dataset_.evaluations(0, locations, probe_rows);
    work_timing_.probe_evaluation_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - probe_started)
            .count();
    const auto rebase_started = std::chrono::steady_clock::now();
    dataset_.run_parallel(raw_batch.size(), [&](std::size_t index, std::size_t) {
      const auto point = index / sample_count;
      const auto dimension = index % sample_count;
      const auto& raw = *raw_batch[index];
      if (!raw) return;
      const auto tape = coordinate_tape(field, raw, path_rows, path_slots);
      if (!tape) return;
      auto& requested = coordinates[point][dimension];
      requested.reserve(requested_rows.size());
      for (const auto row : requested_rows)
        requested.push_back(rebase_after_basis_swaps(field, raw.at(row), *tape));
    });
    work_timing_.pointwise_rebase_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - rebase_started)
            .count();
    analysis_statistics_.avoided_basis_factorizations += raw_batch.size();

    std::vector<std::optional<PivotProposal>> scored(pairs.size());
    const auto interpolation_started = std::chrono::steady_clock::now();
    std::atomic<std::size_t> thiele_attempts{0};
    std::atomic<std::size_t> exhaustive_fallbacks{0};
    firefly::FFInt::set_new_prime(field.prime());
    dataset_.run_parallel(pairs.size(), [&](std::size_t pair_index, std::size_t) {
      const auto [proposal, pivot_slot] = pairs[pair_index];
      const auto slot = pivot_slot;
      const auto candidate = proposal_candidates[proposal];
      std::vector<RationalFunction> pivot_functions;
      std::vector<RationalFunction> child_functions;
      bool complete = true;
      for (std::size_t point = 0; point < point_count; ++point) {
        FieldVector arguments;
        FieldVector pivots;
        FieldVector children;
        for (std::size_t dimension = 0; dimension < sample_count; ++dimension) {
          if (coordinates[point][dimension].size() != requested_rows.size()) continue;
          const auto pivot = coordinates[point][dimension][proposal][slot];
          if (pivot == 0) continue;
          arguments.push_back(dataset_.dimension(0, dimension));
          pivots.push_back(pivot);
          children.push_back(rebase_after_basis_swap(
              field, coordinates[point][dimension][witness_local],
              coordinates[point][dimension][proposal], slot)[witness_slot]);
        }
        if (arguments.size() <= options_.dimension_holdouts) {
          complete = false;
          break;
        }
        thiele_attempts.fetch_add(2, std::memory_order_relaxed);
        auto pivot_function = interpolate_rational_thiele_monic(
            field, arguments, pivots, options_.dimension_holdouts);
        auto child_function = interpolate_rational_thiele_monic(
            field, arguments, children, options_.dimension_holdouts);
        const auto maximum_degree = arguments.size() - options_.dimension_holdouts - 1;
        if (!pivot_function) {
          exhaustive_fallbacks.fetch_add(1, std::memory_order_relaxed);
          pivot_function = interpolate_rational_monic(
              field, arguments, pivots, maximum_degree, options_.dimension_holdouts);
        }
        if (!child_function) {
          exhaustive_fallbacks.fetch_add(1, std::memory_order_relaxed);
          child_function = interpolate_rational_monic(
              field, arguments, children, maximum_degree, options_.dimension_holdouts);
        }
        if (!pivot_function || !child_function) {
          complete = false;
          break;
        }
        pivot_functions.push_back(std::move(*pivot_function));
        child_functions.push_back(std::move(*child_function));
      }
      if (!complete) return;

      std::vector<FieldVector> pivot_numerators;
      std::vector<FieldVector> child_denominators;
      std::size_t numerator_degree = 0;
      for (const auto& function : pivot_functions) {
        pivot_numerators.push_back(function.numerator);
        numerator_degree = std::max(numerator_degree, function.numerator.size() - 1);
      }
      for (const auto& function : child_functions)
        child_denominators.push_back(function.denominator);
      scored[pair_index] = PivotProposal{
          .slot = slot,
          .candidate = candidate,
          .repairs_primary = stable_polynomial_signatures(field, child_denominators),
          .pivot_numerator_stable =
              stable_polynomial_signatures(field, pivot_numerators),
          .child_mixed_degree = moving_polynomial_degree(field, child_denominators),
          .pivot_mixed_degree = moving_polynomial_degree(field, pivot_numerators),
          .pivot_numerator_degree = numerator_degree};
    });
    analysis_statistics_.rational_interpolation_attempts +=
        thiele_attempts.load(std::memory_order_relaxed);
    analysis_statistics_.thiele_attempts +=
        thiele_attempts.load(std::memory_order_relaxed);
    analysis_statistics_.exhaustive_fallbacks +=
        exhaustive_fallbacks.load(std::memory_order_relaxed);
    work_timing_.rational_interpolation_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      interpolation_started)
            .count();
    std::vector<PivotProposal> result;
    result.reserve(scored.size());
    for (auto& proposal : scored)
      if (proposal) result.push_back(std::move(*proposal));
    std::ranges::sort(result, [&](const auto& lhs, const auto& rhs) {
      return std::tuple{lhs.repairs_primary ? 0U : 1U,
                        lhs.child_mixed_degree,
                        lhs.pivot_numerator_stable ? 0U : 1U,
                        lhs.pivot_mixed_degree,
                        lhs.pivot_numerator_degree,
                        dot_counts[lhs.candidate],
                        lhs.candidate,
                        lhs.slot} < std::tuple{rhs.repairs_primary ? 0U : 1U,
                                               rhs.child_mixed_degree,
                                               rhs.pivot_numerator_stable ? 0U : 1U,
                                               rhs.pivot_mixed_degree,
                                               rhs.pivot_numerator_degree,
                                               dot_counts[rhs.candidate],
                                               rhs.candidate,
                                               rhs.slot};
    });
    scored_pairs_ += result.size();
    pair_sampling_failures_ += pairs.size() - result.size();
    if (result.size() > options_.swap_shortlist) result.resize(options_.swap_shortlist);
    if (shortlisted_by_slot_.empty()) shortlisted_by_slot_.resize(basis_size_);
    for (const auto& proposal : result)
      ++shortlisted_by_slot_[proposal.slot];
    return result;
  }

  std::size_t raw_pairs_ = 0, scored_pairs_ = 0, pair_sampling_failures_ = 0;
  std::vector<std::size_t> shortlisted_by_slot_;

  using RationalRow = factor_saturation::RationalRow;

  std::optional<std::vector<RationalRow>>
  recover_rows(std::size_t prime, std::size_t point, std::span<const std::size_t> rows,
               std::span<const DSeparatingBasisSwap> path)
  {
    const PrimeField field(dataset_.primes()[prime]);
    const auto path_rows = swap_rows(path), slots = swap_slots(path);
    auto requested = path_rows;
    requested.insert(requested.end(), rows.begin(), rows.end());
    std::ranges::sort(requested);
    requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
    std::vector<NativeBasisProbeDataset::ProbeLocation> locations;
    for (std::size_t d = 0; d < options_.maximum_dimension_samples; ++d)
      locations.push_back({point, d});
    auto started = std::chrono::steady_clock::now();
    const auto raw = dataset_.evaluations(prime, locations, requested);
    work_timing_.probe_evaluation_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    std::vector<FieldMatrix> coordinates(raw.size());
    started = std::chrono::steady_clock::now();
    dataset_.run_parallel(raw.size(), [&](std::size_t d, std::size_t) {
      if (!*raw[d]) return;
      auto tape = coordinate_tape(field, *raw[d], path_rows, slots);
      if (!tape) return;
      for (auto row : rows)
        coordinates[d].push_back(
            rebase_after_basis_swaps(field, raw[d]->at(row), *tape));
    });
    work_timing_.pointwise_rebase_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    started = std::chrono::steady_clock::now();
    std::vector<RationalRow> result(rows.size(), RationalRow(basis_size_));
    std::atomic<bool> complete{true};
    std::atomic<std::size_t> attempts{0}, fallbacks{0};
    firefly::FFInt::set_new_prime(field.prime());
    dataset_.run_parallel(rows.size() * basis_size_, [&](std::size_t flat,
                                                         std::size_t) {
      const auto row = flat / basis_size_, slot = flat % basis_size_;
      FieldVector arguments, values;
      for (std::size_t d = 0; d < raw.size(); ++d)
        if (coordinates[d].size() == rows.size()) {
          arguments.push_back(dataset_.dimension(prime, d));
          values.push_back(coordinates[d][row][slot]);
        }
      if (arguments.size() <= options_.dimension_holdouts) {
        complete = false;
        return;
      }
      ++attempts;
      auto function = interpolate_rational_thiele_monic(field, arguments, values,
                                                        options_.dimension_holdouts);
      if (!function) {
        ++fallbacks;
        function = interpolate_rational_monic(field, arguments, values,
                                              arguments.size() -
                                                  options_.dimension_holdouts - 1,
                                              options_.dimension_holdouts);
      }
      if (!function) {
        complete = false;
        return;
      }
      result[row][slot] = factor_saturation::normalize(field, std::move(*function));
    });
    analysis_statistics_.rational_interpolation_attempts += attempts;
    analysis_statistics_.thiele_attempts += attempts;
    analysis_statistics_.exhaustive_fallbacks += fallbacks;
    work_timing_.rational_interpolation_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    if (!complete) return std::nullopt;
    return result;
  }

  std::vector<DSeparatingBasisSwap>
  factor_path(std::span<const factor_saturation::Step> steps,
              std::span<const std::size_t> initial, std::span<const Integral> pool)
  {
    std::vector<DSeparatingBasisSwap> result;
    std::vector<std::size_t> selected(initial.begin(), initial.end());
    for (const auto& step : steps) {
      result.push_back({.slot = step.slot,
                        .removed = pool[selected[step.slot]],
                        .inserted = pool[step.candidate],
                        .sector = masters::detail::integral_sector(
                            dataset_.config(), pool[step.candidate])});
      selected[step.slot] = step.candidate;
    }
    return result;
  }

  void sequential_factors(
      std::span<const std::size_t> initial, std::span<const Integral> pool,
      DSeparatingBasisSearchReport& report,
      const std::function<std::string(std::span<const std::size_t>,
                                      std::span<const DSeparatingBasisSwap>)>& validate)
  {
    namespace fx = factor_saturation;
    std::vector<std::uint32_t> sectors;
    for (const auto& integral : pool)
      sectors.push_back(masters::detail::integral_sector(dataset_.config(), integral));
    std::size_t remaining = options_.maximum_basis_states;
    for (std::size_t prime = 0; prime < dataset_.primes().size(); ++prime) {
      const PrimeField field(dataset_.primes()[prime]);
      const auto scan_at = [&](std::span<const fx::Step> steps,
                               std::size_t anchor) -> fx::FactorScan {
        auto path = factor_path(steps, initial, pool);
        std::vector<std::vector<RationalRow>> functions;
        for (std::size_t point = 0; point < options_.kinematic_training_points;
             ++point) {
          auto rows = recover_rows(prime, point, validation_rows_, path);
          if (!rows) return {"interpolation_incomplete", {}};
          functions.push_back(std::move(*rows));
        }
        fx::FactorScan scan;
        bool moving = false;
        for (std::size_t target = 0; target < validation_rows_.size(); ++target)
          for (std::size_t slot = 0; slot < basis_size_; ++slot) {
            std::vector<FieldVector> denominators;
            for (const auto& point : functions)
              denominators.push_back(point[target][slot].denominator);
            moving |= moving_polynomial_degree(field, denominators) != 0;
            const auto part = fx::noncommon_part(field, denominators, anchor);
            for (auto h : fx::factors(field, part))
              if (std::ranges::find(scan.factors, h) == scan.factors.end())
                scan.factors.push_back(std::move(h));
          }
        if (moving && scan.factors.empty()) scan.status = "anchor_inconclusive";
        return scan;
      };
      std::size_t anchor = 0;
      auto first = scan_at({}, anchor);
      while (first.status == "anchor_inconclusive" &&
             anchor + 1 < options_.kinematic_training_points)
        first = scan_at({}, ++anchor);
      report.sequence_primes.push_back(field.prime());
      report.sequence_anchors.push_back(anchor);
      report.sequence_kinematics.push_back(dataset_.kinematics(prime, anchor));
      if (first.status != "complete") {
        fx::SequenceResult failed;
        failed.status = first.status;
        failed.selected.assign(initial.begin(), initial.end());
        report.factor_sequences.push_back(std::move(failed));
        continue;
      }
      double score_data_seconds = 0, score_coordinate_seconds = 0;
      std::map<std::pair<std::size_t, std::size_t>, std::optional<RationalRow>>
          originals;
      for (std::size_t point = 0; point < options_.kinematic_training_points; ++point)
        for (std::size_t slot = 0; slot < initial.size(); ++slot) {
          RationalRow identity(basis_size_, RationalFunction{{0}, {1}});
          identity[slot] = {{1}, {1}};
          originals.emplace(std::pair{point, candidate_rows_[initial[slot]]},
                            std::move(identity));
        }
      const auto prefetch_originals = [&](std::size_t point,
                                          std::span<const std::size_t> physical) {
        std::vector<std::size_t> missing;
        for (const auto row : physical)
          if (!originals.contains({point, row})) missing.push_back(row);
        if (missing.empty()) return;
        std::ranges::sort(missing);
        missing.erase(std::unique(missing.begin(), missing.end()), missing.end());
        std::vector<NativeBasisProbeDataset::ProbeLocation> locations;
        for (std::size_t d = 0; d < options_.maximum_dimension_samples; ++d)
          locations.push_back({point, d});
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t begin = 0; begin < missing.size(); begin += 32)
          (void)dataset_.evaluations(
              prime, locations,
              std::span<const std::size_t>(missing).subspan(
                  begin, std::min<std::size_t>(32, missing.size() - begin)),
              true);
        const auto elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();
        score_data_seconds += elapsed;
        work_timing_.probe_evaluation_seconds += elapsed;
      };
      const auto original_row =
          [&](std::size_t point, std::size_t physical) -> std::optional<RationalRow> {
        const auto key = std::pair{point, physical};
        auto it = originals.find(key);
        if (it != originals.end()) return it->second;
        const auto started = std::chrono::steady_clock::now();
        const std::array requested{physical};
        auto got = recover_rows(prime, point, requested, {});
        std::optional<RationalRow> value;
        if (got) value = std::move(got->front());
        originals.emplace(key, value);
        score_data_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();
        return value;
      };
      const fx::RowProvider provider =
          [&](std::size_t row) -> std::optional<RationalRow> {
        const auto physical = row < pool.size() ? candidate_rows_[row]
                                                : validation_rows_[row - pool.size()];
        return original_row(anchor, physical);
      };
      const fx::RowPrefetch prefetch = [&](std::size_t begin, std::size_t count) {
        std::vector<std::size_t> physical;
        for (auto row = begin; row < begin + count; ++row)
          physical.push_back(row < pool.size() ? candidate_rows_[row]
                                               : validation_rows_[row - pool.size()]);
        prefetch_originals(anchor, physical);
      };
      bool first_scan = true;
      const fx::FactorFinder finder = [&](std::span<const fx::Step> steps) {
        if (first_scan) {
          first_scan = false;
          return first;
        }
        return scan_at(steps, anchor);
      };
      struct ScoredRow {
        std::optional<RationalRow> value;
        std::size_t version = 0;
      };
      struct ScoredPoint {
        std::vector<fx::Step> path;
        std::vector<RationalRow> pivots;
        std::map<std::size_t, ScoredRow> rows;
      };
      std::vector<ScoredPoint> scored_points(options_.kinematic_training_points);
      const auto current_scored_row =
          [&](std::size_t point, std::size_t physical,
              std::span<const fx::Step> steps) -> std::optional<RationalRow> {
        auto& cache = scored_points[point];
        const auto same_step = [](const auto& a, const auto& b) {
          return a.candidate == b.candidate && a.slot == b.slot;
        };
        if (cache.path.size() > steps.size() ||
            !std::equal(cache.path.begin(), cache.path.end(), steps.begin(), same_step))
          cache = {};
        const auto advance = [&](std::size_t row) -> std::optional<RationalRow> {
          auto [it, inserted] = cache.rows.try_emplace(row);
          auto& entry = it->second;
          if (inserted) entry.value = original_row(point, row);
          if (!entry.value) return std::nullopt;
          const auto started = std::chrono::steady_clock::now();
          while (entry.version < cache.pivots.size()) {
            *entry.value = fx::rebase(field, *entry.value, cache.pivots[entry.version],
                                      cache.path[entry.version].slot);
            ++entry.version;
          }
          score_coordinate_seconds +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                  .count();
          return entry.value;
        };
        while (cache.path.size() < steps.size()) {
          const auto& step = steps[cache.path.size()];
          auto pivot = advance(candidate_rows_[step.candidate]);
          if (!pivot) return std::nullopt;
          if (pivot->at(step.slot).numerator == FieldVector{0}) return std::nullopt;
          cache.pivots.push_back(std::move(*pivot));
          cache.path.push_back(step);
        }
        return advance(physical);
      };
      const fx::PairScorer pair_scorer = [&](std::span<const fx::PivotPair> pairs,
                                             std::span<const fx::Step> steps) {
        const auto data_before = score_data_seconds;
        const auto coordinates_before = score_coordinate_seconds;
        fx::PairScores result;
        // Each unique candidate is recovered once per point; all slots share it.
        std::map<std::size_t, std::vector<RationalRow>> functions;
        std::vector<std::size_t> active;
        for (const auto& pair : pairs)
          if (functions.try_emplace(pair.candidate).second)
            active.push_back(pair.candidate);
        for (std::size_t point = 0; point < options_.kinematic_training_points;
             ++point) {
          std::vector<std::size_t> physical;
          // Lazy coordinate advancement may need prefix pivot rows as well.
          for (const auto& step : steps)
            physical.push_back(candidate_rows_[step.candidate]);
          for (auto candidate : active)
            physical.push_back(candidate_rows_[candidate]);
          if (active.empty()) break;
          prefetch_originals(point, physical);
          std::vector<std::size_t> remaining;
          for (auto candidate : active) {
            auto row = current_scored_row(point, candidate_rows_[candidate], steps);
            if (!row)
              functions[candidate].clear();
            else {
              functions[candidate].push_back(std::move(*row));
              remaining.push_back(candidate);
            }
          }
          active = std::move(remaining);
        }
        for (const auto& pair : pairs) {
          auto it = functions.find(pair.candidate);
          fx::NumeratorScore score;
          if (!it->second.empty()) {
            std::vector<FieldVector> numerators;
            bool nonzero = true;
            for (const auto& row : it->second) {
              numerators.push_back(row[pair.slot].numerator);
              nonzero &= numerators.back() != FieldVector{0};
              score.degree = std::max(score.degree, numerators.back().size() - 1);
            }
            if (nonzero) {
              score.classification =
                  stable_polynomial_signatures(field, numerators) ? 0 : 1;
              score.moving_degree = moving_polynomial_degree(field, numerators);
            }
          }
          result.values.push_back(score);
        }
        result.data_seconds = score_data_seconds - data_before;
        result.coordinate_seconds = score_coordinate_seconds - coordinates_before;
        return result;
      };
      auto sequence = fx::sequential(
          field, initial, sectors, validation_rows_.size(), provider, finder,
          [&](std::span<const std::size_t> selected, std::span<const fx::Step> steps) {
            auto path = factor_path(steps, initial, pool);
            return validate(selected, path);
          },
          remaining, pair_scorer, prefetch);
      remaining -= sequence.states_used;
      report.basis_states_tested += sequence.states_used;
      const auto status = sequence.status;
      report.factor_sequences.push_back(std::move(sequence));
      if (status == "passed" || status == "budget_exhausted") break;
    }
  }

  [[nodiscard]] const DSeparatingBasisProbeStatistics&
  analysis_statistics() const noexcept
  {
    return analysis_statistics_;
  }

  [[nodiscard]] const DSeparatingBasisTiming& work_timing() const noexcept
  {
    return work_timing_;
  }

private:
  [[nodiscard]] std::vector<std::size_t>
  swap_rows(std::span<const DSeparatingBasisSwap> path) const
  {
    std::vector<std::size_t> result;
    result.reserve(path.size());
    for (const auto& swap : path)
      result.push_back(find_row(dataset_.config().targets, swap.inserted));
    return result;
  }

  [[nodiscard]] static std::vector<std::size_t>
  swap_slots(std::span<const DSeparatingBasisSwap> path)
  {
    std::vector<std::size_t> result;
    result.reserve(path.size());
    for (const auto& swap : path)
      result.push_back(swap.slot);
    return result;
  }

  [[nodiscard]] static std::optional<BasisSwapCoordinateTape>
  coordinate_tape(const PrimeField& field,
                  const NativeBasisProbeDataset::ProbeEvaluation& evaluation,
                  std::span<const std::size_t> rows, std::span<const std::size_t> slots,
                  DSeparatingBasisTiming* timing = nullptr)
  {
    FieldMatrix inserted;
    inserted.reserve(rows.size());
    for (const auto row : rows)
      inserted.push_back(evaluation.at(row));
    return build_basis_swap_coordinate_tape(field, inserted, slots, timing);
  }

  [[nodiscard]] static bool validates_function(const PrimeField& field,
                                               const RationalFunction& function,
                                               std::span<const std::uint64_t> arguments,
                                               std::span<const std::uint64_t> values)
  {
    if (arguments.size() != values.size()) return false;
    for (std::size_t sample = 0; sample < arguments.size(); ++sample) {
      try {
        if (evaluate(field, function, arguments[sample]) != values[sample])
          return false;
      } catch (const std::domain_error&) {
        return false;
      }
    }
    return true;
  }

  NativeBasisProbeDataset& dataset_;
  std::vector<std::size_t> candidate_rows_;
  std::vector<std::size_t> screening_rows_;
  std::vector<std::size_t> validation_rows_;
  const DSeparatingBasisSearchOptions& options_;
  std::size_t basis_size_ = 0;
  DSeparatingBasisProbeStatistics analysis_statistics_;
  DSeparatingBasisTiming work_timing_;
};

std::vector<std::size_t> candidate_ids(std::span<const Integral> pool,
                                       std::span<const Integral> basis)
{
  std::vector<std::size_t> result;
  result.reserve(basis.size());
  for (const auto& integral : basis)
    result.push_back(find_row(pool, integral));
  return result;
}

std::vector<Integral> selected_integrals(const TopologyConfig& topology,
                                         std::span<const Integral> pool,
                                         std::span<const std::size_t> selected)
{
  std::vector<Integral> result;
  result.reserve(selected.size());
  for (const auto index : selected)
    result.push_back(pool[index]);
  masters::detail::canonical_sort_master_integrals(topology, result);
  return result;
}

} // namespace

FieldVector rebase_after_basis_swap(const PrimeField& field,
                                    std::span<const std::uint64_t> coefficients,
                                    std::span<const std::uint64_t> pivot_coordinates,
                                    std::size_t slot)
{
  if (coefficients.size() != pivot_coordinates.size() || slot >= coefficients.size())
    throw std::invalid_argument("invalid single-swap rebase dimensions");
  if (pivot_coordinates[slot] == 0) throw std::domain_error("basis-swap pivot is zero");
  FieldVector result(coefficients.begin(), coefficients.end());
  rebase_after_basis_swap_in_place(field, result, pivot_coordinates, slot,
                                   field.inverse(pivot_coordinates[slot]));
  return result;
}

std::optional<BasisSwapCoordinateTape> build_basis_swap_coordinate_tape(
    const PrimeField& field, std::span<const FieldVector> inserted_initial_coordinates,
    std::span<const std::size_t> slots, DSeparatingBasisTiming* timing)
{
  if (inserted_initial_coordinates.size() != slots.size())
    throw std::invalid_argument("basis-swap tape input shape mismatch");
  BasisSwapCoordinateTape tape;
  tape.slots.assign(slots.begin(), slots.end());
  tape.pivot_coordinates.reserve(inserted_initial_coordinates.size());
  tape.pivot_inverses.reserve(inserted_initial_coordinates.size());
  for (std::size_t step = 0; step < inserted_initial_coordinates.size(); ++step) {
    const auto coordinate_started = timing ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
    FieldVector pivot = inserted_initial_coordinates[step];
    if (pivot.empty() || slots[step] >= pivot.size())
      throw std::invalid_argument("basis-swap pivot slot is out of range");
    for (std::size_t previous = 0; previous < step; ++previous) {
      rebase_after_basis_swap_in_place(field, pivot, tape.pivot_coordinates[previous],
                                       tape.slots[previous],
                                       tape.pivot_inverses[previous]);
    }
    const auto pivot_started = timing ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
    if (timing)
      timing->rank_coordinates_seconds +=
          std::chrono::duration<double>(pivot_started - coordinate_started).count();
    const bool nonzero = pivot[slots[step]] != 0;
    if (nonzero) tape.pivot_inverses.push_back(field.inverse(pivot[slots[step]]));
    if (timing)
      timing->rank_pivot_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        pivot_started)
              .count();
    if (!nonzero) return std::nullopt;
    tape.pivot_coordinates.push_back(std::move(pivot));
  }
  return tape;
}

FieldVector rebase_after_basis_swaps(const PrimeField& field,
                                     std::span<const std::uint64_t> coefficients,
                                     const BasisSwapCoordinateTape& tape)
{
  if (tape.slots.size() != tape.pivot_coordinates.size() ||
      tape.slots.size() != tape.pivot_inverses.size())
    throw std::invalid_argument("basis-swap tape shape mismatch");
  FieldVector result(coefficients.begin(), coefficients.end());
  for (std::size_t step = 0; step < tape.slots.size(); ++step) {
    rebase_after_basis_swap_in_place(field, result, tape.pivot_coordinates[step],
                                     tape.slots[step], tape.pivot_inverses[step]);
  }
  return result;
}

unsigned integral_dot_count(const TopologyConfig& topology, const Integral& integral)
{
  integral_layout::validate(topology, integral);
  unsigned result = 0;
  for (const auto slot : topology.propagator_slots) {
    if (integral.indices[slot] > 1)
      result += static_cast<unsigned>(integral.indices[slot] - 1);
  }
  return result;
}

BasisIntegralPool BasisIntegralPool::build(const TopologyConfig& topology,
                                           std::span<const Integral> initial_basis,
                                           unsigned maximum_candidate_dots)
{
  if (initial_basis.empty()) throw std::invalid_argument("initial basis is empty");
  std::set<std::uint32_t> sectors;
  for (const auto& integral : initial_basis)
    sectors.insert(masters::detail::integral_sector(topology, integral));

  BasisIntegralPool result;
  for (const auto sector : sectors) {
    const auto active =
        integral_layout::active_variables(sector, topology.propagator_count);
    auto monomials = dot_monomials(active.size(), maximum_candidate_dots);
    std::vector<std::size_t> representatives(monomials.size());
    std::iota(representatives.begin(), representatives.end(), 0);
    if (topology.symmetry && topology.symmetry->backend != SymmetryBackend::None) {
      representatives = symmetry::find_sector_monomial_orbit_representatives(
          topology, sector, monomials, topology.symmetry->backend);
    }
    for (const auto representative : representatives) {
      Integral active_integral;
      active_integral.indices.assign(topology.propagator_count, 0);
      for (std::size_t variable = 0; variable < active.size(); ++variable) {
        active_integral.indices[active[variable]] =
            1 + monomials[representative][variable];
      }
      append_unique(result.integrals,
                    integral_layout::expand_active(topology, active_integral));
    }
  }
  for (const auto& integral : initial_basis)
    append_unique(result.integrals, integral);
  masters::detail::canonical_sort_master_integrals(topology, result.integrals);
  result.dot_counts.reserve(result.integrals.size());
  for (const auto& integral : result.integrals)
    result.dot_counts.push_back(integral_dot_count(topology, integral));
  return result;
}

namespace {

PreparedDSeparatingBasisSearch
run_search_stage(NativeBasisProbeDataset& dataset, const BasisIntegralPool& pool,
                 const std::vector<std::size_t>& initial_candidate_ids,
                 std::span<const std::size_t> candidate_rows,
                 std::span<const std::size_t> screening_rows,
                 std::span<const std::size_t> validation_rows,
                 const DSeparatingBasisSearchOptions& options,
                 BasisSearchProgress& progress)
{
  const auto search_started = std::chrono::steady_clock::now();
  const auto initial_statistics = dataset.statistics();
  const auto& prepared_config = dataset.config();
  const auto basis_size = initial_candidate_ids.size();
  DSeparatingBasisSearchReport report;
  report.candidate_pool_size = pool.integrals.size();
  for (const auto row : screening_rows)
    report.screening_targets.push_back(prepared_config.targets[row]);
  report.validation_targets.reserve(validation_rows.size());
  for (const auto row : validation_rows)
    report.validation_targets.push_back(prepared_config.targets[row]);
  report.primes.assign(dataset.primes().begin(), dataset.primes().end());

  TrialBasisChecker checker(dataset, candidate_rows, screening_rows, validation_rows,
                            options, basis_size);

  const auto strict_validate = [&](std::span<const std::size_t> selected,
                                   std::span<const DSeparatingBasisSwap> path) {
    progress.set_stage("strict-validation");
    return checker.diagnose(selected, path, true);
  };

  const auto finalize_statistics = [&] {
    report.probe_statistics = dataset.statistics();
    report.probe_statistics.logical_requests -= initial_statistics.logical_requests;
    report.probe_statistics.full_cache_hits -= initial_statistics.full_cache_hits;
    report.probe_statistics.requested_target_rows -=
        initial_statistics.requested_target_rows;
    report.probe_statistics.cached_target_rows -= initial_statistics.cached_target_rows;
    report.probe_statistics.replayed_target_rows -=
        initial_statistics.replayed_target_rows;
    report.probe_statistics.selected_replay_calls -=
        initial_statistics.selected_replay_calls;
    report.probe_statistics.selected_compact_outputs -=
        initial_statistics.selected_compact_outputs;
    report.probe_statistics.skipped_compact_outputs -=
        initial_statistics.skipped_compact_outputs;

    const auto& analysis = checker.analysis_statistics();
    report.probe_statistics.rational_interpolation_attempts =
        analysis.rational_interpolation_attempts;
    report.probe_statistics.rational_interpolation_cache_hits =
        analysis.rational_interpolation_cache_hits;
    report.probe_statistics.fixed_degree_attempts = analysis.fixed_degree_attempts;
    report.probe_statistics.thiele_attempts = analysis.thiele_attempts;
    report.probe_statistics.exhaustive_fallbacks = analysis.exhaustive_fallbacks;
    report.probe_statistics.completed_target_rows = analysis.completed_target_rows;
    report.probe_statistics.avoided_basis_factorizations =
        analysis.avoided_basis_factorizations;
    const auto& work = checker.work_timing();
    report.timing.probe_evaluation_seconds = work.probe_evaluation_seconds;
    report.timing.pointwise_rebase_seconds = work.pointwise_rebase_seconds;
    report.timing.rational_interpolation_seconds = work.rational_interpolation_seconds;
    report.timing.signature_comparison_seconds = work.signature_comparison_seconds;
    report.timing.rank_data_seconds = work.rank_data_seconds;
    report.timing.rank_coordinates_seconds = work.rank_coordinates_seconds;
    report.timing.rank_pivot_seconds = work.rank_pivot_seconds;
    report.raw_pairs = checker.raw_pairs_;
    report.scored_pairs = checker.scored_pairs_;
    report.pair_sampling_failures = checker.pair_sampling_failures_;
    report.shortlisted_by_slot = checker.shortlisted_by_slot_;

    report.timing.total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - search_started)
            .count();
  };

  struct SearchState {
    std::vector<std::size_t> selected;
    std::vector<DSeparatingBasisSwap> path;
    bool repairs_primary = true;
    bool pivot_numerator_stable = true;
    std::size_t child_mixed_degree = 0;
    std::size_t pivot_mixed_degree = 0;
    std::size_t pivot_numerator_degree = 0;
  };
  const auto state_key = [](const SearchState& state) {
    auto key = state.selected;
    std::ranges::sort(key);
    return key;
  };
  const auto state_dots = [&](const SearchState& state) {
    unsigned result = 0;
    for (const auto candidate : state.selected)
      result += pool.dot_counts[candidate];
    return result;
  };
  const auto state_less = [&](const SearchState& lhs, const SearchState& rhs) {
    return std::tuple{lhs.repairs_primary ? 0U : 1U,
                      lhs.child_mixed_degree,
                      lhs.pivot_numerator_stable ? 0U : 1U,
                      lhs.pivot_mixed_degree,
                      lhs.pivot_numerator_degree,
                      lhs.path.size(),
                      state_dots(lhs),
                      state_key(lhs)} < std::tuple{rhs.repairs_primary ? 0U : 1U,
                                                   rhs.child_mixed_degree,
                                                   rhs.pivot_numerator_stable ? 0U : 1U,
                                                   rhs.pivot_mixed_degree,
                                                   rhs.pivot_numerator_degree,
                                                   rhs.path.size(),
                                                   state_dots(rhs),
                                                   state_key(rhs)};
  };

  std::vector<SearchState> frontier{{
      .selected = initial_candidate_ids,
      .path = {},
  }};
  std::set<std::vector<std::size_t>> seen{state_key(frontier.front())};
  std::optional<SearchState> solution;
  std::optional<SeparationDiagnostics> solution_diagnostics;
  bool budget_exhausted = false;
  progress.event("initial-basis", "starting factor-directed basis discovery");

  if (options.strategy == DSeparatingSearchStrategy::SequentialFactorsScored) {
    progress.set_stage("sequential-factors");
    checker.sequential_factors(
        initial_candidate_ids, pool.integrals, report,
        [&](std::span<const std::size_t> selected,
            std::span<const DSeparatingBasisSwap> path) -> std::string {
          if (!checker.rank_complete(selected, path)) return "rank_sample_failure";
          ++report.strict_validations;
          const auto validation_started = std::chrono::steady_clock::now();
          auto checked = strict_validate(selected, path);
          report.timing.strict_validation_seconds +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                            validation_started)
                  .count();
          if (checked.result == SeparationResult::Passed) {
            SearchState state;
            state.selected.assign(selected.begin(), selected.end());
            state.path.assign(path.begin(), path.end());
            solution = std::move(state);
            solution_diagnostics = std::move(checked);
            return "passed";
          }
          return checked.result == SeparationResult::MovingPole ? "moving_pole"
                                                                : "sampling_failure";
        });
    budget_exhausted = std::ranges::any_of(report.factor_sequences, [](const auto& x) {
      return x.status == "budget_exhausted";
    });
    frontier.clear();
  }
  while (!frontier.empty()) {
    std::ranges::sort(frontier, state_less);
    auto state = std::move(frontier.front());
    frontier.erase(frontier.begin());
    if (report.basis_states_tested >= options.maximum_basis_states) {
      budget_exhausted = true;
      break;
    }
    ++report.basis_states_tested;
    report.maximum_beam_depth = std::max(report.maximum_beam_depth, state.path.size());
    progress.set_basis_states(report.basis_states_tested);
    progress.set_stage("rank-filter", static_cast<unsigned>(state.path.size()));
    const auto rank_started = std::chrono::steady_clock::now();
    const bool rank_complete = checker.rank_complete(state.selected, state.path);
    report.timing.rank_filter_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - rank_started)
            .count();
    if (!rank_complete) {
      ++report.rank_deficient_rejects;
      progress.set_rejects(report.rank_deficient_rejects, report.moving_d_pole_rejects,
                           report.sampling_rejects);
      continue;
    }

    progress.set_stage("screening", static_cast<unsigned>(state.path.size()));
    const auto screening_started = std::chrono::steady_clock::now();
    auto diagnostics = checker.diagnose(state.selected, state.path, false);
    report.timing.screening_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      screening_started)
            .count();
    if (diagnostics.result == SeparationResult::Passed) {
      ++report.strict_validations;
      progress.set_stage("strict-validation", static_cast<unsigned>(state.path.size()));
      const auto strict_started = std::chrono::steady_clock::now();
      diagnostics = strict_validate(state.selected, state.path);
      report.timing.strict_validation_seconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        strict_started)
              .count();
    }
    if (diagnostics.result == SeparationResult::Passed) {
      solution = std::move(state);
      solution_diagnostics = std::move(diagnostics);
      break;
    }
    if (diagnostics.result == SeparationResult::SamplingFailure ||
        !diagnostics.witness) {
      ++report.sampling_rejects;
      progress.set_rejects(report.rank_deficient_rejects, report.moving_d_pole_rejects,
                           report.sampling_rejects);
      continue;
    }

    ++report.moving_d_pole_rejects;
    progress.set_rejects(report.rank_deficient_rejects, report.moving_d_pole_rejects,
                         report.sampling_rejects);
    if (!report.primary_witness) {
      report.primary_witness = DMovingPoleWitness{
          .target = prepared_config.targets[diagnostics.witness->target_row],
          .component = diagnostics.witness->component,
          .sector = diagnostics.witness->sector,
          .mixed_degree = diagnostics.witness->mixed_degree,
          .master_sector = masters::detail::integral_sector(
              prepared_config,
              pool.integrals[state.selected[diagnostics.witness->component]])};
    }
    progress.event(
        "pole-guided-swap",
        std::format(
            "target={} component={} sector={} mixed_degree={}",
            format_mathematica_integral(
                prepared_config.integral_header,
                prepared_config.targets[diagnostics.witness->target_row].indices),
            diagnostics.witness->component, diagnostics.witness->sector,
            diagnostics.witness->mixed_degree),
        static_cast<unsigned>(state.path.size()));
    const auto pivot_started = std::chrono::steady_clock::now();
    auto proposals = checker.propose_pivots(state.selected, state.path, diagnostics,
                                            pool.integrals, pool.dot_counts);
    report.timing.pivot_scoring_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - pivot_started)
            .count();
    report.swaps_scored += proposals.size();
    progress.set_swaps(report.swaps_scored);
    if (proposals.size() > options.swap_shortlist)
      proposals.resize(options.swap_shortlist);
    report.swaps_shortlisted += proposals.size();

    std::vector<SearchState> proposed_states;
    for (const auto& proposal : proposals) {
      SearchState child = state;
      const auto removed = child.selected[proposal.slot];
      child.selected[proposal.slot] = proposal.candidate;
      child.repairs_primary = proposal.repairs_primary;
      child.pivot_numerator_stable = proposal.pivot_numerator_stable;
      child.child_mixed_degree = proposal.child_mixed_degree;
      child.pivot_mixed_degree = proposal.pivot_mixed_degree;
      child.pivot_numerator_degree = proposal.pivot_numerator_degree;
      child.path.push_back({.slot = proposal.slot,
                            .removed = pool.integrals[removed],
                            .inserted = pool.integrals[proposal.candidate],
                            .sector = masters::detail::integral_sector(
                                prepared_config, pool.integrals[proposal.candidate]),
                            .repairs_primary = proposal.repairs_primary,
                            .pivot_numerator_stable = proposal.pivot_numerator_stable,
                            .child_mixed_degree = proposal.child_mixed_degree,
                            .pivot_mixed_degree = proposal.pivot_mixed_degree,
                            .pivot_numerator_degree = proposal.pivot_numerator_degree});
      proposed_states.push_back(std::move(child));
    }
    std::size_t accepted = 0;
    for (auto& child : proposed_states) {
      if (accepted == options.basis_beam_width) break;
      if (!seen.insert(state_key(child)).second) {
        ++report.seen_suppressed;
        continue;
      }
      frontier.push_back(std::move(child));
      ++accepted;
    }
    std::ranges::sort(frontier, state_less);
    if (frontier.size() > options.basis_beam_width) {
      report.beam_discarded += frontier.size() - options.basis_beam_width;
      frontier.resize(options.basis_beam_width);
    }
  }

  if (solution) {
    report.selected_basis =
        selected_integrals(prepared_config, pool.integrals, solution->selected);
    report.total_dots = state_dots(*solution);
    report.selected_basis_d_separating = true;
    report.swap_path = std::move(solution->path);
    report.status = DSeparatingBasisSearchStatus::Passed;
    report.message =
        "found the first strictly validated pole-guided D-separating basis";
    if (options.strategy == DSeparatingSearchStrategy::SequentialFactorsScored)
      report.message =
          "found a strictly validated basis by sequential factor elimination";
    progress.event("basis-found",
                   std::format("validated_basis_dots={} swaps={}", report.total_dots,
                               report.swap_path.size()));
    progress.event("completed", report.message);

    if (!solution_diagnostics)
      throw std::logic_error("validated basis is missing separation diagnostics");
    // Search coordinates retain swap-slot order; published masters are canonical.
    const auto count = report.selected_basis.size();
    std::vector<std::size_t> canonical_to_slot(count), slot_to_canonical(count);
    for (std::size_t index = 0; index < count; ++index) {
      const auto id = find_row(pool.integrals, report.selected_basis[index]);
      const auto found = std::ranges::find(solution->selected, id);
      if (found == solution->selected.end())
        throw std::logic_error("selected master is absent from swap slots");
      const auto slot = static_cast<std::size_t>(found - solution->selected.begin());
      canonical_to_slot[index] = slot;
      slot_to_canonical[slot] = index;
    }
    auto support = std::move(solution_diagnostics->nonzero_output_support);
    for (auto& position : support)
      position = static_cast<std::uint32_t>((position / count) * count +
                                            slot_to_canonical[position % count]);
    std::ranges::sort(support);
    finalize_statistics();
    return {.report = std::move(report),
            .oracle = nullptr,
            .final_output_support = std::move(support),
            .selected_basis_to_swap_slot = std::move(canonical_to_slot)};
  }
  if (budget_exhausted) {
    report.status = DSeparatingBasisSearchStatus::SearchBudgetExhausted;
    report.message = "search budget exhausted before finding a D-separating basis";
  } else if (report.basis_states_tested != 0 &&
             report.sampling_rejects == report.basis_states_tested) {
    report.status = DSeparatingBasisSearchStatus::SamplingFailure;
    report.message = "all rank-complete trial bases exhausted the D-sampling budget";
  } else {
    report.status = DSeparatingBasisSearchStatus::NotFound;
    report.message =
        "pole-guided search exhausted the configured same-sector candidate pool";
  }
  if (options.strategy == DSeparatingSearchStrategy::SequentialFactorsScored &&
      !budget_exhausted)
    report.message = "sequential factor chains ended without a strictly validated "
                     "basis; see per-chain status";
  progress.event("completed", report.message);
  finalize_statistics();
  return {.report = std::move(report),
          .oracle = nullptr,
          .final_output_support = {},
          .selected_basis_to_swap_slot = {}};
}

void add_search_work(DSeparatingBasisSearchReport& result,
                     const DSeparatingBasisSearchReport& previous)
{
  result.basis_states_tested += previous.basis_states_tested;
  result.raw_pairs += previous.raw_pairs;
  result.scored_pairs += previous.scored_pairs;
  result.pair_sampling_failures += previous.pair_sampling_failures;
  result.beam_discarded += previous.beam_discarded;
  result.seen_suppressed += previous.seen_suppressed;
  result.swaps_scored += previous.swaps_scored;
  result.swaps_shortlisted += previous.swaps_shortlisted;
  result.strict_validations += previous.strict_validations;
  result.rank_deficient_rejects += previous.rank_deficient_rejects;
  result.moving_d_pole_rejects += previous.moving_d_pole_rejects;
  result.sampling_rejects += previous.sampling_rejects;
  result.maximum_beam_depth =
      std::max(result.maximum_beam_depth, previous.maximum_beam_depth);
  if (result.shortlisted_by_slot.size() < previous.shortlisted_by_slot.size())
    result.shortlisted_by_slot.resize(previous.shortlisted_by_slot.size());
  for (std::size_t i = 0; i < previous.shortlisted_by_slot.size(); ++i)
    result.shortlisted_by_slot[i] += previous.shortlisted_by_slot[i];
  result.probe_statistics.logical_requests +=
      previous.probe_statistics.logical_requests;
  result.probe_statistics.full_cache_hits += previous.probe_statistics.full_cache_hits;
  result.probe_statistics.requested_target_rows +=
      previous.probe_statistics.requested_target_rows;
  result.probe_statistics.cached_target_rows +=
      previous.probe_statistics.cached_target_rows;
  result.probe_statistics.replayed_target_rows +=
      previous.probe_statistics.replayed_target_rows;
  result.probe_statistics.selected_replay_calls +=
      previous.probe_statistics.selected_replay_calls;
  result.probe_statistics.selected_compact_outputs +=
      previous.probe_statistics.selected_compact_outputs;
  result.probe_statistics.skipped_compact_outputs +=
      previous.probe_statistics.skipped_compact_outputs;
  result.probe_statistics.rational_interpolation_attempts +=
      previous.probe_statistics.rational_interpolation_attempts;
  result.probe_statistics.rational_interpolation_cache_hits +=
      previous.probe_statistics.rational_interpolation_cache_hits;
  result.probe_statistics.fixed_degree_attempts +=
      previous.probe_statistics.fixed_degree_attempts;
  result.probe_statistics.thiele_attempts += previous.probe_statistics.thiele_attempts;
  result.probe_statistics.exhaustive_fallbacks +=
      previous.probe_statistics.exhaustive_fallbacks;
  result.probe_statistics.completed_target_rows +=
      previous.probe_statistics.completed_target_rows;
  result.probe_statistics.avoided_basis_factorizations +=
      previous.probe_statistics.avoided_basis_factorizations;
  result.timing.total_seconds += previous.timing.total_seconds;
  result.timing.native_prepare_seconds += previous.timing.native_prepare_seconds;
  result.timing.rank_filter_seconds += previous.timing.rank_filter_seconds;
  result.timing.screening_seconds += previous.timing.screening_seconds;
  result.timing.pivot_scoring_seconds += previous.timing.pivot_scoring_seconds;
  result.timing.strict_validation_seconds += previous.timing.strict_validation_seconds;
  result.timing.probe_evaluation_seconds += previous.timing.probe_evaluation_seconds;
  result.timing.pointwise_rebase_seconds += previous.timing.pointwise_rebase_seconds;
  result.timing.rational_interpolation_seconds +=
      previous.timing.rational_interpolation_seconds;
  result.timing.signature_comparison_seconds +=
      previous.timing.signature_comparison_seconds;
  result.timing.rank_data_seconds += previous.timing.rank_data_seconds;
  result.timing.rank_coordinates_seconds += previous.timing.rank_coordinates_seconds;
  result.timing.rank_pivot_seconds += previous.timing.rank_pivot_seconds;
}

} // namespace

PreparedDSeparatingBasisSearch prepare_d_separating_basis_search(
    Config config, std::vector<Integral> initial_basis,
    std::span<const std::uint32_t> relation_source_sectors,
    const DSeparatingBasisSearchOptions& options,
    DSeparatingBasisProgressCallback progress_callback,
    ReductionOptions reduction_options)
{
  switch (options.strategy) {
  case DSeparatingSearchStrategy::SingleSlot:
  case DSeparatingSearchStrategy::SequentialFactorsScored:
  case DSeparatingSearchStrategy::SingleSlotThenScored:
    break;
  default:
    throw std::invalid_argument("unsupported D-separating search strategy");
  }
  const auto search_started = std::chrono::steady_clock::now();
  if (options.initial_dimension_samples <= options.dimension_holdouts ||
      options.maximum_dimension_samples < options.initial_dimension_samples ||
      options.dimension_sample_batch == 0 || options.kinematic_training_points == 0 ||
      options.kinematic_holdout_points == 0 || options.basis_beam_width == 0 ||
      options.swap_shortlist == 0 ||
      (options.strategy == DSeparatingSearchStrategy::SingleSlotThenScored &&
       options.simple_search_maximum_basis_states == 0)) {
    throw std::invalid_argument("basis-search sampling configuration is invalid");
  }

  const auto validation_targets = config.targets;
  if (initial_basis.empty())
    throw std::invalid_argument("D-separating search requires an initial basis");
  if (!config.dimension_parameter_index)
    throw std::invalid_argument(
        "D-separating basis selection requires a free dimension parameter");
  const auto pool =
      BasisIntegralPool::build(config, initial_basis, options.maximum_candidate_dots);
  const auto initial_candidate_ids = candidate_ids(pool.integrals, initial_basis);
  const auto screening = automatic_screening_targets(
      config, config.targets,
      std::min(options.screening_target_limit, config.targets.size()));

  std::vector<Integral> oracle_rows = pool.integrals;
  for (const auto& target : config.targets)
    append_unique(oracle_rows, target);
  config.targets = oracle_rows;
  const auto primes = reduction::detail::usable_firefly_primes(config, 2);
  const std::size_t kinematic_points =
      options.kinematic_training_points + options.kinematic_holdout_points;
  BasisSearchProgress progress(
      std::move(progress_callback), options.progress_interval_seconds,
      options.maximum_basis_states,
      (primes.size() * kinematic_points + options.kinematic_training_points) *
          options.maximum_dimension_samples);
  progress.event(
      "candidate-pool",
      std::format("candidates={} basis={} validation_targets={} screening_targets={} "
                  "candidate_dot_cap={}",
                  pool.integrals.size(), initial_basis.size(),
                  validation_targets.size(), screening.size(),
                  options.maximum_candidate_dots));
  progress.event("native-prepare", "starting shared fixed-basis oracle prepare");
  const auto native_progress = [&](std::string_view message,
                                   ReductionProgressEvent event) {
    std::string_view event_name = "info";
    switch (event) {
    case ReductionProgressEvent::started:
      event_name = "started";
      break;
    case ReductionProgressEvent::completed:
      event_name = "completed";
      break;
    case ReductionProgressEvent::failed:
      event_name = "failed";
      break;
    case ReductionProgressEvent::warning:
      event_name = "warning";
      break;
    case ReductionProgressEvent::info:
      break;
    }
    progress.event("native-prepare",
                   std::format("native_event={} {}", event_name, message));
  };
  const auto native_prepare_started = std::chrono::steady_clock::now();
  NativeBasisProbeDataset dataset(std::move(config), std::move(initial_basis), primes,
                                  kinematic_points, options.maximum_dimension_samples,
                                  relation_source_sectors, reduction_options,
                                  native_progress, progress);
  const auto& prepared_config = dataset.config();
  const double native_prepare_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    native_prepare_started)
          .count();
  progress.event("native-ready", "shared fixed-basis oracle prepare completed");

  std::vector<std::size_t> candidate_rows;
  candidate_rows.reserve(pool.integrals.size());
  for (const auto& integral : pool.integrals)
    candidate_rows.push_back(find_row(prepared_config.targets, integral));
  std::vector<std::size_t> screening_rows;
  for (const auto& target : screening)
    screening_rows.push_back(find_row(prepared_config.targets, target));
  std::vector<std::size_t> validation_rows;
  for (const auto& target : validation_targets) {
    validation_rows.push_back(find_row(prepared_config.targets, target));
  }

  const bool staged =
      options.strategy == DSeparatingSearchStrategy::SingleSlotThenScored;
  auto stage_options = options;
  if (staged) {
    stage_options.strategy = DSeparatingSearchStrategy::SingleSlot;
    stage_options.maximum_basis_states = std::min(
        options.maximum_basis_states, options.simple_search_maximum_basis_states);
  }
  std::vector<DSeparatingSearchStageReport> stages;
  const auto run_stage = [&](const DSeparatingBasisSearchOptions& settings) {
    progress.set_basis_states(0);
    progress.set_swaps(0);
    progress.set_rejects(0, 0, 0);
    progress.event("search-stage-start",
                   std::format("strategy={} budget={}",
                               static_cast<unsigned>(settings.strategy),
                               settings.maximum_basis_states));
    auto result = run_search_stage(dataset, pool, initial_candidate_ids, candidate_rows,
                                   screening_rows, validation_rows, settings, progress);
    auto& report = result.report;
    // Finite-point rank/sampling failures cannot establish search exhaustion.
    if (staged && report.status == DSeparatingBasisSearchStatus::NotFound &&
        report.moving_d_pole_rejects == 0 &&
        (report.sampling_rejects != 0 || report.rank_deficient_rejects != 0 ||
         report.pair_sampling_failures != 0)) {
      report.status = DSeparatingBasisSearchStatus::SamplingFailure;
      report.message = "search stage ended without a usable moving-pole witness";
    }
    stages.push_back({.strategy = settings.strategy,
                      .status = report.status,
                      .reason = report.message,
                      .budget = settings.maximum_basis_states,
                      .states_used = report.basis_states_tested,
                      .probe_statistics = report.probe_statistics,
                      .timing = report.timing});
    progress.event("search-stage-end",
                   std::format("strategy={} status={} states={} replay_calls={} "
                               "cached_rows={} elapsed_ms={:.2f}",
                               static_cast<unsigned>(settings.strategy),
                               static_cast<unsigned>(report.status),
                               report.basis_states_tested,
                               report.probe_statistics.selected_replay_calls,
                               report.probe_statistics.cached_target_rows,
                               report.timing.total_seconds * 1000));
    return result;
  };
  auto result = run_stage(stage_options);
  if (staged &&
      (result.report.status == DSeparatingBasisSearchStatus::NotFound ||
       result.report.status == DSeparatingBasisSearchStatus::SearchBudgetExhausted)) {
    if (result.report.basis_states_tested < options.maximum_basis_states) {
      const auto reason = result.report.status == DSeparatingBasisSearchStatus::NotFound
                              ? "simple_search_exhausted"
                              : "simple_stage_budget";
      stages.back().reason = reason;
      progress.event("search-stage-fallback", reason);
      stage_options.strategy = DSeparatingSearchStrategy::SequentialFactorsScored;
      stage_options.maximum_basis_states =
          options.maximum_basis_states - result.report.basis_states_tested;
      auto expanded = run_stage(stage_options);
      add_search_work(expanded.report, result.report);
      result = std::move(expanded);
    } else {
      result.report.status = DSeparatingBasisSearchStatus::SearchBudgetExhausted;
      result.report.message = "total search budget exhausted before expanded search";
    }
  }
  result.report.stages = std::move(stages);
  result.report.timing.native_prepare_seconds = native_prepare_seconds;
  result.report.timing.total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - search_started)
          .count();
  result.oracle = dataset.release_oracle();
  return result;
}

} // namespace basis
