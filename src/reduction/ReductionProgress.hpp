#pragma once

#include <functional>
#include <string_view>

enum class ReductionProgressEvent {
  started,
  completed,
  failed,
  info,
  warning,
};

using ReductionProgressCallback =
    std::function<void(std::string_view, ReductionProgressEvent)>;
