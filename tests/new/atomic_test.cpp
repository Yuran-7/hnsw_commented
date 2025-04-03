#include <atomic>
#include <iostream>
#include <omp.h>  // OpenMP 头文件

class ThreadSafeCounter {
public:
    ThreadSafeCounter() : count(0) {}

    // 线程安全的自增操作
    void increment() {
        count.fetch_add(1, std::memory_order_relaxed);  // 原子自增
    }

    // 获取当前计数器的值
    int get() const {
        return count.load(std::memory_order_relaxed);  // 原子加载
    }

private:
    std::atomic<int> count;  // 原子计数器
};

int main() {
    ThreadSafeCounter counter;

    const int num_threads = 10;
    const int increments_per_thread = 1000;

    // 使用 OpenMP 并行区域
    #pragma omp parallel num_threads(num_threads)
    {
        // 每个线程执行 increments_per_thread 次自增操作
        for (int j = 0; j < increments_per_thread; ++j) {
            counter.increment();
        }
    }

    /*
    const int N = 10000;  // 总任务数

    // 使用 OpenMP 并行 for 循环
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        counter.increment();  // 每个线程执行自增操作
    }
    */

    // 输出最终结果
    std::cout << "Final count: " << counter.get() << std::endl;

    return 0;
}