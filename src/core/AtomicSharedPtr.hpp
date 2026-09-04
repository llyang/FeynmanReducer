#pragma once

#include <atomic>
#include <memory>
#include <utility>
#include <version>

namespace core {

// Apple libc++ releases predating the C++20 atomic<shared_ptr> specialization
// still provide the C++11 shared_ptr atomic free functions. Keep one interface
// for both implementations so hot-path callers use the native specialization
// whenever the standard library advertises it.
template <typename T> class AtomicSharedPtr {
public:
  AtomicSharedPtr() noexcept = default;
  AtomicSharedPtr(const AtomicSharedPtr&) = delete;
  AtomicSharedPtr& operator=(const AtomicSharedPtr&) = delete;

  [[nodiscard]] std::shared_ptr<T>
  load(std::memory_order order = std::memory_order_seq_cst) const noexcept
  {
#if !defined(FR_FORCE_ATOMIC_SHARED_PTR_FALLBACK) &&                                   \
    defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
    return value_.load(order);
#else
    return std::atomic_load_explicit(&value_, order);
#endif
  }

  void store(std::shared_ptr<T> desired,
             std::memory_order order = std::memory_order_seq_cst) noexcept
  {
#if !defined(FR_FORCE_ATOMIC_SHARED_PTR_FALLBACK) &&                                   \
    defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
    value_.store(std::move(desired), order);
#else
    std::atomic_store_explicit(&value_, std::move(desired), order);
#endif
  }

private:
#if !defined(FR_FORCE_ATOMIC_SHARED_PTR_FALLBACK) &&                                   \
    defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
  std::atomic<std::shared_ptr<T>> value_;
#else
  std::shared_ptr<T> value_;
#endif
};

} // namespace core
