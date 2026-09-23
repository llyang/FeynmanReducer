#pragma once

#include "core/Config.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace reduction::detail {

inline std::int64_t request_output_scale_offset(const Config& config,
                                                const Integral& integral)
{
  std::int64_t sum = 0;
  for (const int index : integral.indices) {
    if (__builtin_add_overflow(sum, static_cast<std::int64_t>(index), &sum) ||
        sum == std::numeric_limits<std::int64_t>::min())
      throw std::overflow_error("integral index sum overflow");
  }
  std::int64_t shift = 0;
  if (__builtin_mul_overflow(static_cast<std::int64_t>(config.loop_count),
                             static_cast<std::int64_t>(integral.dimension_shift / 2),
                             &shift) ||
      __builtin_sub_overflow(shift, sum, &shift))
    throw std::overflow_error("integral scale degree overflow");
  return shift;
}

inline std::vector<ReductionRequest> resolved_reduction_requests(const Config& config)
{
  if (!config.reduction_requests.empty()) return config.reduction_requests;
  std::vector<ReductionRequest> result;
  result.reserve(config.targets.size());
  for (std::size_t target = 0; target < config.targets.size(); ++target) {
    if (target > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("target index exceeds 32-bit request ids");
    ReductionRequest request;
    request.output.integral = config.targets[target];
    request.output.scale_offset =
        request_output_scale_offset(config, config.targets[target]);
    request.terms.push_back({static_cast<std::uint32_t>(target), std::string("1")});
    result.push_back(std::move(request));
  }
  return result;
}

} // namespace reduction::detail
