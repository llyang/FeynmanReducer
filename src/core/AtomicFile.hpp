#pragma once

#include <unistd.h>

#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace core::atomic_file {

namespace detail {

inline void prepare_parent(const std::filesystem::path& destination)
{
  if (destination.has_parent_path()) {
    std::filesystem::create_directories(destination.parent_path());
  }
}

inline std::filesystem::path temporary_sibling(const std::filesystem::path& destination)
{
  static std::atomic_size_t sequence{0};
  for (;;) {
    auto temporary = destination;
    temporary += std::format(".tmp.{}.{}", getpid(),
                             sequence.fetch_add(1, std::memory_order_relaxed));
    if (!std::filesystem::exists(temporary)) return temporary;
  }
}

inline void replace_with(const std::filesystem::path& temporary,
                         const std::filesystem::path& destination,
                         std::string_view description)
{
  std::error_code error;
  std::filesystem::rename(temporary, destination, error);
  if (error) {
    throw std::filesystem::filesystem_error(
        "failed to publish " + std::string(description), temporary, destination, error);
  }
}

inline void remove_temporary(const std::filesystem::path& temporary) noexcept
{
  std::error_code ignored;
  std::filesystem::remove(temporary, ignored);
}

} // namespace detail

template <typename Writer>
void write(const std::filesystem::path& destination, std::string_view description,
           Writer&& writer)
{
  detail::prepare_parent(destination);
  const auto temporary = detail::temporary_sibling(destination);
  try {
    {
      std::ofstream stream(temporary, std::ios::trunc);
      if (!stream) {
        throw std::runtime_error("cannot write " + std::string(description) + ": " +
                                 destination.string());
      }
      writer(stream);
      stream.close();
      if (!stream) {
        throw std::runtime_error("failed while writing " + std::string(description) +
                                 ": " + destination.string());
      }
    }
    detail::replace_with(temporary, destination, description);
  } catch (...) {
    detail::remove_temporary(temporary);
    throw;
  }
}

inline void move_or_copy(const std::filesystem::path& source,
                         const std::filesystem::path& destination,
                         std::string_view description)
{
  detail::prepare_parent(destination);

  std::error_code move_error;
  std::filesystem::rename(source, destination, move_error);
  if (!move_error) return;
  if (move_error != std::errc::cross_device_link) {
    throw std::filesystem::filesystem_error(
        "failed to move " + std::string(description), source, destination, move_error);
  }

  const auto temporary = detail::temporary_sibling(destination);
  try {
    std::error_code error;
    const bool copied = std::filesystem::copy_file(
        source, temporary, std::filesystem::copy_options::none, error);
    if (error || !copied) {
      throw std::runtime_error(std::format(
          "failed to copy {}: {} -> {}: {}", description, source.string(),
          temporary.string(), error ? error.message() : "copy did not create a file"));
    }
    detail::replace_with(temporary, destination, description);
    std::error_code remove_error;
    std::filesystem::remove(source, remove_error);
    if (remove_error) {
      throw std::filesystem::filesystem_error(
          "failed to remove copied " + std::string(description), source, remove_error);
    }
  } catch (...) {
    detail::remove_temporary(temporary);
    throw;
  }
}

} // namespace core::atomic_file
