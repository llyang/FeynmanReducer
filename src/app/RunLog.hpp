#pragma once

#include <chrono>
#include <filesystem>
#include <iosfwd>
#include <memory>

class RunLog {
public:
  RunLog(const std::filesystem::path& output_directory,
         std::chrono::system_clock::time_point started_at, std::ostream& output,
         std::ostream& error);
  ~RunLog();

  RunLog(const RunLog&) = delete;
  RunLog& operator=(const RunLog&) = delete;
  RunLog(RunLog&&) = delete;
  RunLog& operator=(RunLog&&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept;
  [[nodiscard]] std::ostream& detail() noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
