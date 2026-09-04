#pragma once

#include "core/Config.hpp"
#include "reduction/Monomial.hpp"
#include "reduction/SymmetryCanonicalizer.hpp"

#include <span>
#include <unordered_map>
#include <vector>

/// Canonicalizes full-slot jet rows.  Denominator rows retain all denominator
/// subsector symmetries; nonzero jets use only exact extended-LP permutations.
class JetSymmetryCanonicalizer {
public:
  JetSymmetryCanonicalizer(const Config& config,
                           const SymmetryCanonicalizer& denominator);

  [[nodiscard]] std::vector<int> canonicalize(std::span<const int> powers) const;

private:
  const Config& config_;
  const SymmetryCanonicalizer& denominator_;
  mutable std::unordered_map<std::vector<int>, std::vector<int>, VectorHash> cache_;
};
