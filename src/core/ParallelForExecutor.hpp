#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace core {

// Persistent deterministic-index worker pool shared by independent batches.
// The caller owns all result slots; workers only receive their stable index and
// worker id. Calls to run() must not overlap.
class ParallelForExecutor {
public:
  explicit ParallelForExecutor(std::size_t workers)
  {
    threads_.reserve(workers > 0 ? workers - 1 : 0);
    for (std::size_t worker = 1; worker < workers; ++worker) {
      threads_.emplace_back([this, worker] { worker_loop(worker); });
    }
  }

  ParallelForExecutor(const ParallelForExecutor&) = delete;
  ParallelForExecutor& operator=(const ParallelForExecutor&) = delete;

  ~ParallelForExecutor()
  {
    {
      const std::lock_guard lock(mutex_);
      stopping_ = true;
      ++epoch_;
    }
    work_ready_.notify_all();
    for (auto& thread : threads_)
      thread.join();
  }

  template <typename Function> void run(std::size_t count, Function&& function)
  {
    if (threads_.empty() || count == 0) {
      for (std::size_t index = 0; index < count; ++index)
        function(index, 0);
      return;
    }
    {
      const std::lock_guard lock(mutex_);
      task_ = std::forward<Function>(function);
      task_count_ = count;
      next_.store(0, std::memory_order_relaxed);
      unfinished_workers_ = threads_.size();
      error_ = nullptr;
      ++epoch_;
    }
    work_ready_.notify_all();
    run_available_tasks(0);
    std::unique_lock lock(mutex_);
    finished_.wait(lock, [this] { return unfinished_workers_ == 0; });
    task_ = {};
    const auto error = error_;
    lock.unlock();
    if (error != nullptr) std::rethrow_exception(error);
  }

private:
  void run_available_tasks(std::size_t worker)
  {
    for (;;) {
      const std::size_t index = next_.fetch_add(1, std::memory_order_relaxed);
      if (index >= task_count_) return;
      try {
        task_(index, worker);
      } catch (...) {
        const std::lock_guard lock(mutex_);
        if (error_ == nullptr) error_ = std::current_exception();
        next_.store(task_count_, std::memory_order_relaxed);
        return;
      }
    }
  }

  void worker_loop(std::size_t worker)
  {
    std::size_t observed_epoch = 0;
    for (;;) {
      {
        std::unique_lock lock(mutex_);
        work_ready_.wait(lock,
                         [this, &observed_epoch] { return epoch_ != observed_epoch; });
        observed_epoch = epoch_;
        if (stopping_) return;
      }
      run_available_tasks(worker);
      {
        const std::lock_guard lock(mutex_);
        if (--unfinished_workers_ == 0) finished_.notify_one();
      }
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable finished_;
  std::function<void(std::size_t, std::size_t)> task_;
  std::atomic<std::size_t> next_{0};
  std::size_t task_count_ = 0;
  std::size_t unfinished_workers_ = 0;
  std::size_t epoch_ = 0;
  std::exception_ptr error_;
  bool stopping_ = false;
};

} // namespace core
