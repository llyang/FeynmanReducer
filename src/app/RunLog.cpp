#include "app/RunLog.hpp"

#include <ctime>
#include <fstream>
#include <limits>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace {

std::string local_run_stamp(std::chrono::system_clock::time_point time_point)
{
  const std::time_t raw = std::chrono::system_clock::to_time_t(time_point);
  std::tm local{};
#if defined(_WIN32)
  localtime_s(&local, &raw);
#else
  localtime_r(&raw, &local);
#endif
  char text[13]{};
  if (std::strftime(text, sizeof(text), "%y%m%d%H%M%S", &local) == 0) {
    throw std::runtime_error("failed to format reduction log timestamp");
  }
  return text;
}

struct ReservedPath {
  std::filesystem::path log;
  std::filesystem::path reservation;
};

ReservedPath reserve_log_path(const std::filesystem::path& output_directory,
                              std::chrono::system_clock::time_point started_at)
{
  std::filesystem::create_directories(output_directory);
  const std::string prefix = "reduction_" + local_run_stamp(started_at);
  for (std::size_t index = 0; index < std::numeric_limits<std::size_t>::max();
       ++index) {
    const std::string suffix = index == 0 ? "" : "_" + std::to_string(index);
    const auto candidate = output_directory / (prefix + suffix + ".log");
    std::error_code exists_error;
    if (std::filesystem::exists(candidate, exists_error)) {
      continue;
    }
    if (exists_error) {
      throw std::runtime_error("failed to inspect reduction log path " +
                               candidate.string() + ": " + exists_error.message());
    }

    const auto reservation =
        output_directory / ("." + candidate.filename().string() + ".lock");
    std::error_code reserve_error;
    if (std::filesystem::create_directory(reservation, reserve_error)) {
      return {candidate, reservation};
    }
    if (reserve_error && reserve_error != std::errc::file_exists) {
      throw std::runtime_error("failed to reserve reduction log path " +
                               candidate.string() + ": " + reserve_error.message());
    }
  }
  throw std::runtime_error("failed to select a unique reduction log path");
}

class LogSink {
public:
  explicit LogSink(std::ofstream& stream) : stream_(stream) {}

  void write(std::string_view text)
  {
    const std::lock_guard lock(mutex_);
    stream_.write(text.data(), static_cast<std::streamsize>(text.size()));
  }

  void put(char character)
  {
    const std::lock_guard lock(mutex_);
    stream_.put(character);
  }

  void flush()
  {
    const std::lock_guard lock(mutex_);
    stream_.flush();
  }

private:
  std::ofstream& stream_;
  std::mutex mutex_;
};

class LogOnlyStreamBuffer final : public std::streambuf {
public:
  explicit LogOnlyStreamBuffer(LogSink& sink) : sink_(sink) {}

protected:
  int_type overflow(int_type character) override
  {
    if (traits_type::eq_int_type(character, traits_type::eof())) {
      return traits_type::not_eof(character);
    }
    sink_.put(traits_type::to_char_type(character));
    return character;
  }

  std::streamsize xsputn(const char* data, std::streamsize size) override
  {
    sink_.write({data, static_cast<std::size_t>(size)});
    return size;
  }

  int sync() override
  {
    sink_.flush();
    return 0;
  }

private:
  LogSink& sink_;
};

class MirroringStreamBuffer final : public std::streambuf {
public:
  MirroringStreamBuffer(std::streambuf& primary, LogSink& sink)
      : primary_(primary), sink_(sink)
  {}

protected:
  int_type overflow(int_type character) override
  {
    if (traits_type::eq_int_type(character, traits_type::eof())) {
      return traits_type::not_eof(character);
    }
    const char value = traits_type::to_char_type(character);
    if (traits_type::eq_int_type(primary_.sputc(value), traits_type::eof())) {
      return traits_type::eof();
    }
    write_filtered({&value, 1});
    return character;
  }

  std::streamsize xsputn(const char* data, std::streamsize size) override
  {
    const std::streamsize written = primary_.sputn(data, size);
    if (written > 0) {
      write_filtered({data, static_cast<std::size_t>(written)});
    }
    return written;
  }

  int sync() override
  {
    const int primary_result = primary_.pubsync();
    sink_.flush();
    return primary_result;
  }

private:
  enum class EscapeState {
    normal,
    after_escape,
    control_sequence,
  };

  void write_filtered(std::string_view text)
  {
    std::string filtered;
    filtered.reserve(text.size());
    for (const char character : text) {
      const auto byte = static_cast<unsigned char>(character);
      if (escape_state_ == EscapeState::after_escape) {
        escape_state_ =
            byte == '[' ? EscapeState::control_sequence : EscapeState::normal;
        continue;
      }
      if (escape_state_ == EscapeState::control_sequence) {
        if (byte >= 0x40 && byte <= 0x7e) {
          escape_state_ = EscapeState::normal;
        }
        continue;
      }
      if (byte == 0x1b) {
        escape_state_ = EscapeState::after_escape;
        continue;
      }
      if (byte == '\r') {
        filtered.push_back('\n');
        skip_line_feed_ = true;
        continue;
      }
      if (byte == '\n' && skip_line_feed_) {
        skip_line_feed_ = false;
        continue;
      }
      skip_line_feed_ = false;
      filtered.push_back(static_cast<char>(byte));
    }
    if (!filtered.empty()) {
      sink_.write(filtered);
    }
  }

  std::streambuf& primary_;
  LogSink& sink_;
  EscapeState escape_state_ = EscapeState::normal;
  bool skip_line_feed_ = false;
};

} // namespace

class RunLog::Impl {
public:
  Impl(const std::filesystem::path& output_directory,
       std::chrono::system_clock::time_point started_at, std::ostream& output,
       std::ostream& error)
      : output_(output), error_(error)
  {
    ReservedPath reserved = reserve_log_path(output_directory, started_at);
    path_ = std::move(reserved.log);
    stream_.open(path_, std::ios::out | std::ios::trunc);
    std::error_code remove_error;
    std::filesystem::remove(reserved.reservation, remove_error);
    if (!stream_) {
      throw std::runtime_error("failed to open reduction log " + path_.string());
    }
    if (remove_error) {
      throw std::runtime_error("failed to release reduction log reservation " +
                               reserved.reservation.string() + ": " +
                               remove_error.message());
    }

    sink_ = std::make_unique<LogSink>(stream_);
    output_buffer_ = std::make_unique<MirroringStreamBuffer>(*output_.rdbuf(), *sink_);
    error_buffer_ = std::make_unique<MirroringStreamBuffer>(*error_.rdbuf(), *sink_);
    detail_buffer_ = std::make_unique<LogOnlyStreamBuffer>(*sink_);
    detail_stream_ = std::make_unique<std::ostream>(detail_buffer_.get());
    original_output_ = output_.rdbuf(output_buffer_.get());
    original_error_ = error_.rdbuf(error_buffer_.get());
  }

  ~Impl()
  {
    output_.flush();
    error_.flush();
    detail_stream_->flush();
    output_.rdbuf(original_output_);
    error_.rdbuf(original_error_);
    sink_->flush();
  }

  const std::filesystem::path& path() const noexcept
  {
    return path_;
  }

  std::ostream& detail() noexcept
  {
    return *detail_stream_;
  }

private:
  std::ostream& output_;
  std::ostream& error_;
  std::filesystem::path path_;
  std::ofstream stream_;
  std::unique_ptr<LogSink> sink_;
  std::unique_ptr<MirroringStreamBuffer> output_buffer_;
  std::unique_ptr<MirroringStreamBuffer> error_buffer_;
  std::unique_ptr<LogOnlyStreamBuffer> detail_buffer_;
  std::unique_ptr<std::ostream> detail_stream_;
  std::streambuf* original_output_ = nullptr;
  std::streambuf* original_error_ = nullptr;
};

RunLog::RunLog(const std::filesystem::path& output_directory,
               std::chrono::system_clock::time_point started_at, std::ostream& output,
               std::ostream& error)
    : impl_(std::make_unique<Impl>(output_directory, started_at, output, error))
{}

RunLog::~RunLog() = default;

const std::filesystem::path& RunLog::path() const noexcept
{
  return impl_->path();
}

std::ostream& RunLog::detail() noexcept
{
  return impl_->detail();
}
