#pragma once

#include "basis/DSeparatingBasisSearch.hpp"
#include "core/AtomicSharedPtr.hpp"
#include "core/Config.hpp"
#include "reduction/ReductionOptions.hpp"
#include "reduction/ReductionProgress.hpp"

#include <firefly/BlackBoxBase.hpp>
#include <firefly/FFInt.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace basis {

// FireFly-facing selected-basis view over the same fixed-initial-basis replay
// kernel used during D-separating discovery. Every finite-field evaluation
// performs only a small pointwise multi-RHS coordinate change.
class DSeparatingReduction : public firefly::BlackBoxBase<DSeparatingReduction> {
public:
  [[nodiscard]] static std::unique_ptr<DSeparatingReduction>
  prepare(Config& output_config, std::vector<Integral> initial_basis,
          std::span<const std::uint32_t> relation_source_sectors,
          ReductionProgressCallback progress = {}, ReductionOptions options = {});

  DSeparatingReduction(const DSeparatingReduction&) = delete;
  DSeparatingReduction& operator=(const DSeparatingReduction&) = delete;
  ~DSeparatingReduction();

  [[nodiscard]] std::span<const std::uint32_t> reconstructed_outputs() const noexcept
  {
    return final_output_support_;
  }

  [[nodiscard]] std::size_t total_output_count() const noexcept
  {
    return original_targets_.size() * prepared_.report.selected_basis.size();
  }

  void prime_changed();

  [[nodiscard]] std::vector<firefly::FFInt>
  eval_selected(const std::vector<firefly::FFInt>& values,
                const std::vector<std::uint32_t>& active_outputs);

  [[nodiscard]] std::vector<firefly::FFInt>
  eval_selected_compact(const std::vector<firefly::FFInt>& values,
                        const std::vector<std::uint32_t>& active_outputs);

  template <typename T> std::vector<T> operator()(const std::vector<T>& values)
  {
    if constexpr (!std::is_same_v<T, firefly::FFInt>) {
      throw std::runtime_error("only bunch_size = 1 (FFInt) is supported");
    } else {
      return evaluate_plan(values, *full_selection_plan_);
    }
  }

private:
  struct OutputSelectionPlan {
    std::vector<std::uint32_t> active_outputs;
    std::shared_ptr<const NativeFixedBasisOracle::PreparedRows> oracle_rows;
    std::vector<std::size_t> needed_targets;
    std::vector<std::size_t> swap_probe_positions;
    std::vector<std::size_t> target_probe_positions;
    std::vector<std::size_t> output_basis_positions;
    std::vector<std::vector<std::size_t>> output_indices_by_target;
  };

  DSeparatingReduction(PreparedDSeparatingBasisSearch prepared,
                       std::vector<Integral> original_targets);

  [[nodiscard]] std::shared_ptr<const OutputSelectionPlan>
  build_selection_plan(std::span<const std::uint32_t> active_outputs);

  [[nodiscard]] std::shared_ptr<const OutputSelectionPlan>
  selected_plan(std::span<const std::uint32_t> active_outputs);

  [[nodiscard]] std::vector<firefly::FFInt>
  evaluate_plan(const std::vector<firefly::FFInt>& values,
                const OutputSelectionPlan& plan);

  PreparedDSeparatingBasisSearch prepared_;
  std::vector<Integral> original_targets_;
  std::vector<std::size_t> swap_rows_;
  std::vector<std::size_t> swap_slots_;
  std::vector<std::size_t> target_rows_;
  std::vector<std::uint32_t> final_output_support_;
  std::shared_ptr<const OutputSelectionPlan> full_selection_plan_;
  std::mutex selection_plan_mutex_;
  core::AtomicSharedPtr<const OutputSelectionPlan> selected_selection_plan_;
  bool identity_basis_ = false;
};

} // namespace basis

namespace quotient {
using basis::DSeparatingReduction;
} // namespace quotient
