#pragma once

#include <filesystem>

class FireflyLogArchive {
public:
  FireflyLogArchive(std::filesystem::path source, std::filesystem::path destination);
  ~FireflyLogArchive();

  FireflyLogArchive(const FireflyLogArchive&) = delete;
  FireflyLogArchive& operator=(const FireflyLogArchive&) = delete;

  // Archives the log produced by this run and restores a protected pre-run
  // source log. Returns true if this run produced a log.
  bool finish();

private:
  void restore_backup();
  void finish_noexcept() noexcept;

  std::filesystem::path source_;
  std::filesystem::path destination_;
  std::filesystem::path backup_;
  bool finalized_ = false;
  bool archived_ = false;
};
