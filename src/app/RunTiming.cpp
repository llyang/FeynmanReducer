#include "app/RunTiming.hpp"

#include <ctime>
#include <format>
#include <ostream>

namespace {

std::string local_clock_text(std::chrono::system_clock::time_point time_point)
{
  const std::time_t raw = std::chrono::system_clock::to_time_t(time_point);
  std::tm local{};
#if defined(_WIN32)
  localtime_s(&local, &raw);
#else
  localtime_r(&raw, &local);
#endif
  char text[20]{};
  if (std::strftime(text, sizeof(text), "%Y-%m-%d %H:%M:%S", &local) == 0) {
    throw std::runtime_error("failed to format local clock time");
  }
  return text;
}

double seconds(std::chrono::steady_clock::duration duration)
{
  return std::chrono::duration<double>(duration).count();
}

} // namespace

RunTiming::RunTiming(std::ostream& output, std::ostream& error, SystemNow system_now,
                     SteadyNow steady_now)
    : output_(output), error_(error), system_now_(std::move(system_now)),
      steady_now_(std::move(steady_now)), system_started_(system_now_()),
      run_started_(steady_now_()), info_output_(&output_)
{}

void RunTiming::set_info_output(std::ostream& detail) noexcept
{
  info_output_ = &detail;
}

void RunTiming::stage_started(std::string label)
{
  if (active_stage_) {
    throw std::logic_error("cannot start a timing stage while another stage is active");
  }
  const auto now = steady_now_();
  active_stage_ = ActiveStage{std::move(label), now};
  emit(output_, "START", active_stage_->label, now, std::nullopt);
}

void RunTiming::stage_completed(std::string_view label)
{
  finish_stage(label, "DONE");
}

void RunTiming::stage_failed(std::string_view label) noexcept
{
  try {
    finish_stage(label, "FAILED");
  } catch (...) {
    active_stage_.reset();
  }
}

void RunTiming::info(std::string_view message)
{
  const auto now = steady_now_();
  emit(*info_output_, "INFO", message, now, std::nullopt);
}

void RunTiming::summary(std::string_view message)
{
  const auto now = steady_now_();
  emit(output_, "INFO", message, now, std::nullopt);
}

void RunTiming::error(std::string_view message)
{
  const auto now = steady_now_();
  emit(error_, "ERROR", message, now, std::nullopt);
}

void RunTiming::complete(std::string_view label)
{
  if (active_stage_) {
    throw std::logic_error("cannot complete run timing while a stage is active");
  }
  const auto now = steady_now_();
  emit(output_, "COMPLETE", label, now, std::nullopt);
}

std::chrono::system_clock::time_point RunTiming::started_at() const noexcept
{
  return system_started_;
}

void RunTiming::finish_stage(std::string_view label, std::string_view status)
{
  if (!active_stage_ || active_stage_->label != label) {
    throw std::logic_error("timing stage completion does not match its start");
  }
  const auto now = steady_now_();
  const auto duration = now - active_stage_->started;
  std::ostream& stream = status == "FAILED" ? error_ : output_;
  emit(stream, status, active_stage_->label, now, duration);
  active_stage_.reset();
}

void RunTiming::emit(std::ostream& stream, std::string_view status,
                     std::string_view label, std::chrono::steady_clock::time_point now,
                     std::optional<std::chrono::steady_clock::duration> stage_duration)
{
  stream << "[clock=" << local_clock_text(system_now_()) << ']';
  if (stage_duration) {
    stream << std::format("[stage={:.3f}s]", seconds(*stage_duration));
  }
  stream << std::format("[elapsed={:.3f}s]", seconds(now - run_started_)) << '['
         << status << "] " << label << '\n'
         << std::flush;
}
