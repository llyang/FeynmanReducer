#pragma once

#include "core/Config.hpp"
#include "reduction/EquationGenerator.hpp"
#include "reduction/Monomial.hpp"
#include "topology/SectorUtils.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

/// Direct distributional equations for negative-index boundary jets.
/// Full powers are [G, x_1, ..., x_integral_count].  A boundary power
/// -1-r denotes delta^(r)(x).
class JetEquationGenerator {
public:
  JetEquationGenerator(const Config& config, const SectorUtils& sectors);

  [[nodiscard]] std::vector<std::vector<int>>
  get_rank_dot_ansatz_domain(std::span<const Integral> additional_anchors = {},
                             unsigned pinched_dot_halo = 1) const;

  [[nodiscard]] std::vector<int>
  seed_powers(std::span<const int> propagator_powers,
              std::span<const std::uint16_t> jet_orders) const;

  [[nodiscard]] std::vector<Monomial>
  build_column(const std::vector<int>& powers, AnsatzFamily family,
               unsigned derivative_slot = 0, bool allow_boundary_raise = false) const;

  [[nodiscard]] std::uint32_t sector_from_powers(std::span<const int> powers) const;
  [[nodiscard]] std::vector<std::uint16_t>
  jet_orders(std::span<const int> powers) const;

private:
  const Config& config_;
  const SectorUtils& sectors_;

  struct Product {
    std::vector<int> powers;
    std::int64_t coefficient = 1;
  };

  [[nodiscard]] bool row_is_allowed(std::span<const int> powers) const;
  [[nodiscard]] std::optional<Product>
  multiply(std::span<const int> powers, std::span<const std::uint8_t> monomial) const;
  [[nodiscard]] std::vector<Monomial>
  dimension_shift(const std::vector<int>& powers) const;
  [[nodiscard]] std::vector<Monomial>
  nabla(unsigned slot, const std::vector<int>& powers, bool allow_boundary_raise) const;
  [[nodiscard]] std::vector<Monomial> euler(const std::vector<int>& powers) const;
};
