#pragma once

#include "core/Config.hpp"

#include <cstdint>
#include <vector>

namespace masters {

enum class MasterCandidateMode {
  FinalBasis,
  GlobalSelection,
};

// Data-only boundary between Singular discovery, optional global selection and
// fixed-basis reduction. Keeping it independent of MasterFinder prevents the
// reduction API from importing the discovery implementation.
struct MasterCandidateSet {
  std::vector<Integral> integrals;
  // Positive-dimensional and zero-candidate sector corners that must still
  // seed the relation envelope.
  std::vector<std::uint32_t> relation_source_sectors;
  MasterCandidateMode mode = MasterCandidateMode::FinalBasis;

  [[nodiscard]] bool requires_global_selection() const noexcept
  {
    return mode == MasterCandidateMode::GlobalSelection;
  }
};

} // namespace masters
