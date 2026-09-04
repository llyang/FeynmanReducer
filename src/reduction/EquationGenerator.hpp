#pragma once

#include "core/Config.hpp"
#include "reduction/Monomial.hpp"
#include "reduction/TopLpTargets.hpp"
#include "topology/SectorUtils.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

enum class AnsatzFamily : std::uint8_t { Nabla, Euler, DimensionShift };

inline constexpr std::size_t ansatz_family_count = 3;

class AnsatzClosureError : public std::runtime_error {
public:
  AnsatzClosureError()
      : std::runtime_error("projected ansatz residual expansion made no progress")
  {}

  explicit AnsatzClosureError(std::string message)
      : std::runtime_error(std::move(message))
  {}
};

struct AnsatzSeedLayer {
  unsigned g_shift = 0;
  std::vector<std::vector<int>> anchors;
};

class EquationGenerator {
  const Config& cfg;
  const SectorUtils& sector;
  std::span<const std::uint32_t> variable_slots;
  std::span<const PolynomialTerm> polynomial_terms;
  const TopLpTargetPlan* target_plan = nullptr;
  // The compiler/SectorUtils invariant caps active propagators at 31, so a
  // scalar mask is the complete support key and avoids a heap allocation per
  // ansatz point.
  std::vector<std::uint32_t> polynomial_term_support_masks;
  mutable std::unordered_map<std::uint32_t, std::vector<std::vector<unsigned>>>
      surviving_term_cache;

  [[nodiscard]] bool powers_are_allowed(const std::vector<int>& powers) const;
  [[nodiscard]] const std::vector<unsigned>&
  surviving_polynomial_terms(const std::vector<int>& powers,
                             unsigned derivative_index) const;
  [[nodiscard]] std::vector<Monomial>
  dimension_shift_generator(const std::vector<int>& powers) const;
  [[nodiscard]] std::vector<Monomial>
  nabla_k_generator(unsigned k, const std::vector<int>& powers) const;
  [[nodiscard]] std::vector<Monomial>
  euler_generator(const std::vector<int>& powers) const;
  [[nodiscard]] std::vector<std::vector<Monomial>>
  build_integral_columns(const std::vector<Integral>& integrals) const;
  void append_integral_anchors(std::vector<std::vector<int>>& anchors,
                               std::span<const Integral> integrals) const;
  [[nodiscard]] std::vector<std::vector<int>>
  enumerate_initial_dot_excess_domain(const std::vector<std::vector<int>>& anchors,
                                      unsigned seed_g_shift) const;

public:
  EquationGenerator(const Config& config, const SectorUtils& sector_utils);
  EquationGenerator(const Config& config, const SectorUtils& sector_utils,
                    const TopLpTargetPlan* target_plan);

  [[nodiscard]] std::vector<std::vector<Monomial>> build_basis_columns() const;
  [[nodiscard]] std::vector<std::vector<Monomial>> build_target_columns() const;

  [[nodiscard]] std::vector<AnsatzSeedLayer>
  build_ansatz_seed_layers(std::span<const Integral> additional_anchors = {}) const;
  [[nodiscard]] std::vector<std::vector<int>>
  build_initial_ansatz_domain(const AnsatzSeedLayer& seed_layer) const;

  [[nodiscard]] std::vector<std::vector<int>>
  build_initial_ansatz_domain(std::span<const Integral> additional_anchors = {}) const;

  [[nodiscard]] std::vector<Monomial>
  build_ansatz_column(const std::vector<int>& powers, AnsatzFamily family,
                      unsigned derivative_index = 0) const;

  template <typename Emit>
  void for_each_ansatz_column(const std::vector<std::vector<int>>& grid,
                              Emit&& emit) const
  {
    for (unsigned k = 1; k <= variable_slots.size(); ++k) {
      for (std::size_t grid_index = 0; grid_index < grid.size(); ++grid_index) {
        const auto& powers = grid[grid_index];
        auto terms = build_ansatz_column(powers, AnsatzFamily::Nabla, k);
        if (!terms.empty()) {
          emit(grid_index, AnsatzFamily::Nabla, k, std::move(terms));
        }
      }
    }

    for (std::size_t grid_index = 0; grid_index < grid.size(); ++grid_index) {
      auto terms = build_ansatz_column(grid[grid_index], AnsatzFamily::Euler);
      if (!terms.empty()) {
        emit(grid_index, AnsatzFamily::Euler, 0, std::move(terms));
      }
    }

    for (std::size_t grid_index = 0; grid_index < grid.size(); ++grid_index) {
      auto terms = build_ansatz_column(grid[grid_index], AnsatzFamily::DimensionShift);
      if (!terms.empty()) {
        emit(grid_index, AnsatzFamily::DimensionShift, 0, std::move(terms));
      }
    }
  }
};
