#pragma once

#include "core/Config.hpp"
#include "reduction/Monomial.hpp"
#include "topology/SectorUtils.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

/// Maps integral powers to deterministic representatives of the stored LP
/// symmetry classes. The reduction matrix is built directly in this quotient
/// space; no explicit I(A)-I(pi(A)) columns are required.
class SymmetryCanonicalizer {
  struct Action {
    std::uint32_t target_sector = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> variable_map;
  };

  unsigned variable_count_ = 0;
  const SectorUtils& sectors_;
  std::unordered_map<std::uint32_t, std::uint32_t> representatives_;
  std::unordered_map<std::uint32_t, Action> to_representative_;
  std::unordered_map<std::uint32_t, std::vector<Action>> internal_actions_;
  std::vector<VariablePermutation> exact_generators_;
  mutable std::unordered_map<std::vector<int>, std::vector<int>, VectorHash> cache_;

  [[nodiscard]] std::vector<int> apply(std::span<const int> powers,
                                       const Action& action) const;

public:
  SymmetryCanonicalizer(const Config& config, const SectorUtils& sectors);
  SymmetryCanonicalizer(const Config& config, const SectorUtils& sectors,
                        std::span<const std::uint32_t> variable_slots);

  /// Inputs first transport a member sector to its stored representative. The
  /// G power is never changed.
  [[nodiscard]] std::vector<int> canonicalize(std::span<const int> powers) const;

  /// Canonicalizes every grid seed, sorts the result, and removes duplicates
  /// without expanding the grid to its complete symmetry orbit.
  void canonicalize_grid(std::vector<std::vector<int>>& grid) const;
};
