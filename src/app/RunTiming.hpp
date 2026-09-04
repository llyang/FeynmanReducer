#pragma once

#include <chrono>
#include <functional>
#include <iosfwd>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

class RunTiming {
public:
  using SystemNow = std::function<std::chrono::system_clock::time_point()>;
  using SteadyNow = std::function<std::chrono::steady_clock::time_point()>;

  RunTiming(
      std::ostream& output, std::ostream& error,
      SystemNow system_now = [] { return std::chrono::system_clock::now(); },
      SteadyNow steady_now = [] { return std::chrono::steady_clock::now(); });

  void set_info_output(std::ostream& detail) noexcept;
  void stage_started(std::string label);
  void stage_completed(std::string_view label);
  void stage_failed(std::string_view label) noexcept;
  void info(std::string_view message);
  void summary(std::string_view message);
  void error(std::string_view message);
  void complete(std::string_view label);
  [[nodiscard]] std::chrono::system_clock::time_point started_at() const noexcept;

  template <typename Function>
  std::invoke_result_t<Function> run_stage(std::string label, Function&& function)
  {
    stage_started(label);
    try {
      if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
        std::invoke(std::forward<Function>(function));
        stage_completed(label);
      } else {
        std::invoke_result_t<Function> result =
            std::invoke(std::forward<Function>(function));
        stage_completed(label);
        return result;
      }
    } catch (...) {
      stage_failed(label);
      throw;
    }
  }

private:
  struct ActiveStage {
    std::string label;
    std::chrono::steady_clock::time_point started;
  };

  void finish_stage(std::string_view label, std::string_view status);
  void emit(std::ostream& stream, std::string_view status, std::string_view label,
            std::chrono::steady_clock::time_point now,
            std::optional<std::chrono::steady_clock::duration> stage_duration);

  std::ostream& output_;
  std::ostream& error_;
  SystemNow system_now_;
  SteadyNow steady_now_;
  std::chrono::system_clock::time_point system_started_;
  std::chrono::steady_clock::time_point run_started_;
  std::ostream* info_output_;
  std::optional<ActiveStage> active_stage_;
};
