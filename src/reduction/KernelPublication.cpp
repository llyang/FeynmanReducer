#include "reduction/BlackBoxFeynman.hpp"
#include "reduction/KernelPlanning.hpp"

#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

void BlackBoxFeynman::publish_kernel_plan(
    reduction::detail::KernelPublicationInput&& input,
    const EvaluatedCoeffs<firefly::FFInt>& coeffs,
    const std::vector<firefly::FFInt>& values)
{
  square_dim = input.solution_columns.size();
  if (input.row_sectors.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("reference evaluator row count exceeds 32-bit ids");

  const auto reference_start = std::chrono::steady_clock::now();
  build_reference_evaluation_plan(input);
  const auto reference_end = std::chrono::steady_clock::now();
  build_pending_block_replay(input, coeffs, values);
  const auto recording_end = std::chrono::steady_clock::now();
  kernel_statistics_.reference_plan_seconds +=
      std::chrono::duration<double>(reference_end - reference_start).count();
  kernel_statistics_.block_recording_seconds +=
      std::chrono::duration<double>(recording_end - reference_end).count();
}
