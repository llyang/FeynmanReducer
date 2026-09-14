#pragma once

namespace basis {

enum class DSeparatingSearchStrategy {
  SingleSlot,
  MultiSlot,
  MultiSlotHistory,
  SingleFactorDiagnostic,
  SequentialFactors,
  SequentialFactorsScored,
  SingleSlotThenScored
};

} // namespace basis
