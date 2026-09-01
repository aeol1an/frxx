#pragma once

#include <cstddef>
#include <functional>
#include <memory>

#include <frxx/utils/integer.hpp>

#ifndef FRXX_WORKER_POOL_HARDWARE_DIVISOR
#define FRXX_WORKER_POOL_HARDWARE_DIVISOR 2
#endif

namespace frxx::utils {

/// Minimal persistent pool for independent indexed work.
class WorkerPool {
public:
    /// Use half the system-recommended hardware concurrency, with a minimum of one.
    static std::size_t recommended_thread_count();

    explicit WorkerPool(
        std::size_t thread_count = recommended_thread_count());
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;
    WorkerPool(WorkerPool&&) = delete;
    WorkerPool& operator=(WorkerPool&&) = delete;

    std::size_t thread_count() const noexcept;

    /// Invoke `function(index)` once for every index in `[begin, end)`.
    void pfor(
        i64 begin,
        i64 end,
        const std::function<void(i64)>& function);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace frxx::utils
