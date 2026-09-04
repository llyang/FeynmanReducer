#include "masters/detail/SingularProcess.hpp"

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace masters::detail {
namespace {

class UniqueFd {
public:
  UniqueFd() = default;
  explicit UniqueFd(int fd) : fd_(fd) {}
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  UniqueFd& operator=(UniqueFd&& other) noexcept
  {
    if (this != &other) {
      reset();
      fd_ = std::exchange(other.fd_, -1);
    }
    return *this;
  }
  ~UniqueFd()
  {
    reset();
  }

  [[nodiscard]] int get() const
  {
    return fd_;
  }
  [[nodiscard]] bool valid() const
  {
    return fd_ >= 0;
  }
  void reset(int fd = -1)
  {
    if (fd_ >= 0) {
      close(fd_);
    }
    fd_ = fd;
  }

private:
  int fd_ = -1;
};

struct Pipe {
  UniqueFd read_end;
  UniqueFd write_end;
};

void set_close_on_exec(int fd)
{
  const int flags = fcntl(fd, F_GETFD);
  if (flags < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0) {
    throw std::runtime_error(
        std::string("failed to configure Singular process pipe: ") +
        std::strerror(errno));
  }
}

UniqueFd move_above_standard_descriptors(UniqueFd fd)
{
  if (fd.get() > STDERR_FILENO) {
    return fd;
  }
  const int moved = fcntl(fd.get(), F_DUPFD, STDERR_FILENO + 1);
  if (moved < 0) {
    throw std::runtime_error(std::string("failed to reserve Singular process pipe: ") +
                             std::strerror(errno));
  }
  UniqueFd result(moved);
  set_close_on_exec(result.get());
  return result;
}

Pipe make_pipe()
{
  int descriptors[2];
  if (pipe(descriptors) != 0) {
    throw std::runtime_error(std::string("failed to create Singular process pipe: ") +
                             std::strerror(errno));
  }
  UniqueFd read_end(descriptors[0]);
  UniqueFd write_end(descriptors[1]);
  set_close_on_exec(read_end.get());
  set_close_on_exec(write_end.get());
  read_end = move_above_standard_descriptors(std::move(read_end));
  write_end = move_above_standard_descriptors(std::move(write_end));
  return {std::move(read_end), std::move(write_end)};
}

void set_nonblocking(int fd)
{
  const int flags = fcntl(fd, F_GETFL);
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    throw std::runtime_error(
        std::string("failed to configure Singular process pipe: ") +
        std::strerror(errno));
  }
}

class ChildProcess {
public:
  explicit ChildProcess(pid_t pid) : pid_(pid) {}
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;
  ~ChildProcess()
  {
    if (pid_ <= 0) {
      return;
    }
    if (kill(pid_, SIGKILL) != 0 && errno != ESRCH) {
      return;
    }
    while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {
    }
  }

  int wait()
  {
    int status = 0;
    while (waitpid(pid_, &status, 0) < 0) {
      if (errno != EINTR) {
        throw std::runtime_error(std::string("failed to wait for Singular: ") +
                                 std::strerror(errno));
      }
    }
    pid_ = -1;
    return status;
  }

private:
  pid_t pid_ = -1;
};

class ScopedSigpipeBlock {
public:
  ScopedSigpipeBlock()
  {
    sigemptyset(&sigpipe_set_);
    sigaddset(&sigpipe_set_, SIGPIPE);
    const int error = pthread_sigmask(SIG_BLOCK, &sigpipe_set_, &old_mask_);
    if (error != 0) {
      throw std::runtime_error(std::string("failed to block SIGPIPE: ") +
                               std::strerror(error));
    }
    active_ = true;
    sigset_t pending;
    if (sigpending(&pending) != 0) {
      const int pending_error = errno;
      pthread_sigmask(SIG_SETMASK, &old_mask_, nullptr);
      active_ = false;
      throw std::runtime_error(std::string("failed to inspect pending signals: ") +
                               std::strerror(pending_error));
    }
    sigpipe_was_pending_ = sigismember(&pending, SIGPIPE) == 1;
  }
  ScopedSigpipeBlock(const ScopedSigpipeBlock&) = delete;
  ScopedSigpipeBlock& operator=(const ScopedSigpipeBlock&) = delete;
  ~ScopedSigpipeBlock()
  {
    if (active_) {
      pthread_sigmask(SIG_SETMASK, &old_mask_, nullptr);
    }
  }

  void consume_generated_sigpipe()
  {
    if (sigpipe_was_pending_) {
      return;
    }
    sigset_t pending;
    if (sigpending(&pending) != 0 || sigismember(&pending, SIGPIPE) != 1) {
      return;
    }
    int received_signal = 0;
    int error = 0;
    do {
      error = sigwait(&sigpipe_set_, &received_signal);
    } while (error == EINTR);
  }

private:
  sigset_t sigpipe_set_{};
  sigset_t old_mask_{};
  bool active_ = false;
  bool sigpipe_was_pending_ = false;
};

[[noreturn]] void child_failure(const char* message, std::size_t size, int status)
{
  while (size > 0) {
    const ssize_t written = write(STDERR_FILENO, message, size);
    if (written > 0) {
      message += written;
      size -= static_cast<std::size_t>(written);
    } else if (written < 0 && errno == EINTR) {
      continue;
    } else {
      break;
    }
  }
  _exit(status);
}

void redirect_child_fd(int source, int destination)
{
  if (dup2(source, destination) < 0) {
    constexpr char message[] = "failed to redirect Singular process pipe\n";
    child_failure(message, sizeof(message) - 1, 126);
  }
}

void close_child_pipes(const Pipe& stdin_pipe, const Pipe& stdout_pipe,
                       const Pipe& stderr_pipe)
{
  close(stdin_pipe.read_end.get());
  close(stdin_pipe.write_end.get());
  close(stdout_pipe.read_end.get());
  close(stdout_pipe.write_end.get());
  close(stderr_pipe.read_end.get());
  close(stderr_pipe.write_end.get());
}

void drain_output(UniqueFd& fd, std::string& output)
{
  std::array<char, 16384> buffer{};
  while (fd.valid()) {
    const ssize_t count = read(fd.get(), buffer.data(), buffer.size());
    if (count > 0) {
      output.append(buffer.data(), static_cast<std::size_t>(count));
    } else if (count == 0) {
      fd.reset();
    } else if (errno == EINTR) {
      continue;
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return;
    } else {
      throw std::runtime_error(std::string("failed to read Singular process pipe: ") +
                               std::strerror(errno));
    }
  }
}

} // namespace

SingularProcessOutput run_singular_process(const std::string& singular_path,
                                           const std::string& script)
{
  Pipe stdin_pipe = make_pipe();
  Pipe stdout_pipe = make_pipe();
  Pipe stderr_pipe = make_pipe();
  set_nonblocking(stdin_pipe.write_end.get());
  set_nonblocking(stdout_pipe.read_end.get());
  set_nonblocking(stderr_pipe.read_end.get());

  const pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error(std::string("failed to fork Singular: ") +
                             std::strerror(errno));
  }
  if (pid == 0) {
    redirect_child_fd(stdin_pipe.read_end.get(), STDIN_FILENO);
    redirect_child_fd(stdout_pipe.write_end.get(), STDOUT_FILENO);
    redirect_child_fd(stderr_pipe.write_end.get(), STDERR_FILENO);
    close_child_pipes(stdin_pipe, stdout_pipe, stderr_pipe);
    execlp(singular_path.c_str(), singular_path.c_str(), "--no-tty", "--quiet",
           static_cast<char*>(nullptr));
    constexpr char message[] = "failed to execute Singular\n";
    child_failure(message, sizeof(message) - 1, 127);
  }

  ChildProcess child(pid);
  stdin_pipe.read_end.reset();
  stdout_pipe.write_end.reset();
  stderr_pipe.write_end.reset();
  ScopedSigpipeBlock sigpipe_block;
  SingularProcessOutput output;
  std::size_t script_offset = 0;
  if (script.empty()) {
    stdin_pipe.write_end.reset();
  }

  while (stdin_pipe.write_end.valid() || stdout_pipe.read_end.valid() ||
         stderr_pipe.read_end.valid()) {
    std::array<pollfd, 3> descriptors{
        pollfd{
            stdin_pipe.write_end.valid() ? stdin_pipe.write_end.get() : -1,
            POLLOUT,
            0,
        },
        pollfd{
            stdout_pipe.read_end.valid() ? stdout_pipe.read_end.get() : -1,
            POLLIN,
            0,
        },
        pollfd{
            stderr_pipe.read_end.valid() ? stderr_pipe.read_end.get() : -1,
            POLLIN,
            0,
        },
    };
    int ready = 0;
    do {
      ready = poll(descriptors.data(), descriptors.size(), -1);
    } while (ready < 0 && errno == EINTR);
    if (ready < 0) {
      throw std::runtime_error(std::string("failed to poll Singular process pipes: ") +
                               std::strerror(errno));
    }

    const short stdin_events = descriptors[0].revents;
    if (stdin_pipe.write_end.valid() && (stdin_events & POLLOUT) != 0) {
      const ssize_t written =
          write(stdin_pipe.write_end.get(), script.data() + script_offset,
                script.size() - script_offset);
      if (written > 0) {
        script_offset += static_cast<std::size_t>(written);
        if (script_offset == script.size()) {
          stdin_pipe.write_end.reset();
        }
      } else if (written < 0 && errno == EPIPE) {
        sigpipe_block.consume_generated_sigpipe();
        stdin_pipe.write_end.reset();
      } else if (written < 0 && errno != EINTR && errno != EAGAIN &&
                 errno != EWOULDBLOCK) {
        throw std::runtime_error(
            std::string("failed to write Singular process pipe: ") +
            std::strerror(errno));
      }
    }
    if (stdin_pipe.write_end.valid() &&
        (stdin_events & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      stdin_pipe.write_end.reset();
    }

    const short stdout_events = descriptors[1].revents;
    if (stdout_pipe.read_end.valid() &&
        (stdout_events & (POLLIN | POLLERR | POLLHUP)) != 0) {
      drain_output(stdout_pipe.read_end, output.stdout_text);
    }
    if (stdout_pipe.read_end.valid() && (stdout_events & POLLNVAL) != 0) {
      throw std::runtime_error("invalid Singular stdout pipe descriptor");
    }

    const short stderr_events = descriptors[2].revents;
    if (stderr_pipe.read_end.valid() &&
        (stderr_events & (POLLIN | POLLERR | POLLHUP)) != 0) {
      drain_output(stderr_pipe.read_end, output.stderr_text);
    }
    if (stderr_pipe.read_end.valid() && (stderr_events & POLLNVAL) != 0) {
      throw std::runtime_error("invalid Singular stderr pipe descriptor");
    }
  }
  output.status = child.wait();
  return output;
}

} // namespace masters::detail
