#pragma once

#include "reduction/ReductionProgress.hpp"
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

namespace reduction::detail {

template <typename Function>
std::invoke_result_t<Function>
run_reduction_stage(const ReductionProgressCallback& progress, std::string label,
                    Function&& function)
{
  if (progress) progress(label, ReductionProgressEvent::started);
  try {
    if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
      std::invoke(std::forward<Function>(function));
      if (progress) progress(label, ReductionProgressEvent::completed);
    } else {
      std::invoke_result_t<Function> result =
          std::invoke(std::forward<Function>(function));
      if (progress) progress(label, ReductionProgressEvent::completed);
      return result;
    }
  } catch (...) {
    if (progress) {
      try {
        progress(label, ReductionProgressEvent::failed);
      } catch (...) {
      }
    }
    throw;
  }
}

} // namespace reduction::detail
