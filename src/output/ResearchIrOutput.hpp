#pragma once

#include "core/Config.hpp"
#include "core/ReductionResult.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

struct ResearchIrSectorClass {
  std::uint32_t representative = 0;
  std::vector<std::uint32_t> members;
};

struct ResearchIrMetadata {
  std::string parametric_input;
  unsigned loop_order = 0;
  unsigned propagator_count = 0;
  unsigned integral_slot_count = 0;
  std::vector<std::uint8_t> top_sector;
  // Stored as zero-based C++ slots. The Mathematica IR writes one-based slots.
  std::vector<std::uint32_t> propagator_slots;
  // Only nontrivial symmetry classes are stored. Missing sectors are singleton
  // classes whose representative is the sector itself.
  std::vector<ResearchIrSectorClass> sector_classes;
};

[[nodiscard]] ResearchIrMetadata make_research_ir_metadata(const Config& config);

void write_mathematica_research_ir(const ResearchIrMetadata& metadata,
                                   const ReductionResult& result,
                                   const std::filesystem::path& output);
