#include <frxx/utils/workerPool.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace frxx::utils {

namespace {

struct ParallelBatch {
    ParallelBatch(
        i64 begin,
        i64 end,
        std::size_t worker_count,
        const std::function<void(i64)>& function)
        : next(begin), end(end), remaining(worker_count), function(function) {}

    std::atomic<i64> next;
    const i64 end;
    std::atomic<std::size_t> remaining;
    std::atomic<bool> cancelled{false};
    std::function<void(i64)> function;
    std::mutex completion_mutex;
    std::condition_variable completion_condition;
    std::mutex exception_mutex;
    std::exception_ptr exception;
};

}  // namespace

class WorkerPool::Impl {
public:
    explicit Impl(std::size_t thread_count) {
        if (thread_count == 0) {
            throw std::invalid_argument("worker pool must contain at least one thread");
        }
        try {
            workers_.reserve(thread_count);
            for (std::size_t index = 0; index < thread_count; ++index) {
                workers_.emplace_back([this] { run(); });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                stopping_ = true;
            }
            queue_condition_.notify_all();
            for (auto& worker : workers_) {
                worker.join();
            }
            throw;
        }
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }
        queue_condition_.notify_all();
        for (auto& worker : workers_) {
            worker.join();
        }
    }

    std::size_t thread_count() const noexcept {
        return workers_.size();
    }

    void pfor(
        i64 begin,
        i64 end,
        const std::function<void(i64)>& function
    ) {
        if (end <= begin) {
            return;
        }
        const auto count = static_cast<std::size_t>(end - begin);
        const std::size_t participants = std::min(count, workers_.size());
        auto batch = std::make_shared<ParallelBatch>(
            begin, end, participants, function);
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            for (std::size_t index = 0; index < participants; ++index) {
                tasks_.emplace_back([batch] {
                    try {
                        while (!batch->cancelled.load(std::memory_order_relaxed)) {
                            const i64 work_index = batch->next.fetch_add(
                                1, std::memory_order_relaxed);
                            if (work_index >= batch->end) {
                                break;
                            }
                            batch->function(work_index);
                        }
                    } catch (...) {
                        {
                            std::lock_guard<std::mutex> exception_lock(
                                batch->exception_mutex);
                            if (!batch->exception) {
                                batch->exception = std::current_exception();
                            }
                        }
                        batch->cancelled.store(true, std::memory_order_relaxed);
                    }
                    if (batch->remaining.fetch_sub(
                            1, std::memory_order_acq_rel) == 1) {
                        batch->completion_condition.notify_one();
                    }
                });
            }
        }
        queue_condition_.notify_all();

        std::unique_lock<std::mutex> completion_lock(batch->completion_mutex);
        batch->completion_condition.wait(completion_lock, [&] {
            return batch->remaining.load(std::memory_order_acquire) == 0;
        });
        std::exception_ptr exception;
        {
            std::lock_guard<std::mutex> exception_lock(batch->exception_mutex);
            exception = batch->exception;
        }
        if (exception) {
            std::rethrow_exception(exception);
        }
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_condition_.wait(lock, [&] {
                    return stopping_ || !tasks_.empty();
                });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    bool stopping_ = false;
};

std::size_t WorkerPool::recommended_thread_count() {
    const std::size_t hardware_threads = std::thread::hardware_concurrency();
    return std::max<std::size_t>(
        1, hardware_threads / FRXX_WORKER_POOL_HARDWARE_DIVISOR);
}

WorkerPool::WorkerPool(std::size_t thread_count)
    : impl_(std::make_unique<Impl>(thread_count)) {}

WorkerPool::~WorkerPool() = default;

std::size_t WorkerPool::thread_count() const noexcept {
    return impl_->thread_count();
}

void WorkerPool::pfor(
    i64 begin,
    i64 end,
    const std::function<void(i64)>& function
) {
    impl_->pfor(begin, end, function);
}

}  // namespace frxx::utils
