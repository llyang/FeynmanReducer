#include "app/FireflyLogArchive.hpp"

#include "core/AtomicFile.hpp"

#include <chrono>
#include <format>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace {

std::filesystem::path backup_path_for(const std::filesystem::path& source)
{
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return source.string() + ".feynman-reducer-backup-" + std::to_string(stamp);
}

void rename_or_throw(const std::filesystem::path& from, const std::filesystem::path& to,
                     const char* operation)
{
  std::error_code error;
  std::filesystem::rename(from, to, error);
  if (error) {
    throw std::runtime_error(std::format("{}: {} -> {}: {}", operation, from.string(),
                                         to.string(), error.message()));
  }
}

} // namespace

FireflyLogArchive::FireflyLogArchive(std::filesystem::path source,
                                     std::filesystem::path destination)
    : source_(std::move(source)), destination_(std::move(destination))
{
  if (std::filesystem::exists(source_)) {
    backup_ = backup_path_for(source_);
    rename_or_throw(source_, backup_, "failed to protect existing FireFly log");
  }
}

FireflyLogArchive::~FireflyLogArchive()
{
  finish_noexcept();
}

void FireflyLogArchive::restore_backup()
{
  if (backup_.empty() || !std::filesystem::exists(backup_)) {
    return;
  }
  rename_or_throw(backup_, source_, "failed to restore existing FireFly log");
}

bool FireflyLogArchive::finish()
{
  if (finalized_) {
    return archived_;
  }
  if (std::filesystem::exists(source_)) {
    core::atomic_file::move_or_copy(source_, destination_, "FireFly log");
    archived_ = true;
  }
  restore_backup();
  finalized_ = true;
  return archived_;
}

void FireflyLogArchive::finish_noexcept() noexcept
{
  if (finalized_) {
    return;
  }
  try {
    static_cast<void>(finish());
  } catch (...) { // NOLINT(bugprone-empty-catch)
    // An explicit finish() failure remains retryable here. If the retry also
    // fails, preserve the current log, protected backup and old destination
    // rather than deleting any of them from a noexcept destructor.
  }
}
