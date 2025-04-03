#include "../../hnswlib/hnswlib.h"
#include <thread>
#include <chrono>


template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {  // 主线程负责创建新线程，22个线程，最多可以创建21个工作线程
            threads.push_back(std::thread([&, threadId] {   // 工作线程会执行一个死循环，并且被加入到线程数组中
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);   // 死循环一直调用fn函数，即addpoint函数，
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


int main() {
    std::cout << "Running multithread load test" << std::endl;
    int d = 16;
    int num_elements = 1000;
    int max_elements = 2 * num_elements;
    int num_threads = 50;   // 不是。你上来就认为我有50个线程？

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;

    hnswlib::L2Space space(d);

    // generate batch1 and batch2 data
    float* batch1 = new float[d * max_elements];
    for (int i = 0; i < d * max_elements; i++) {    // 2000个16维的float数据
        batch1[i] = distrib_real(rng);
    }
    float* batch2 = new float[d * num_elements];
    for (int i = 0; i < d * num_elements; i++) {    // 1000个16维的float数据
        batch2[i] = distrib_real(rng);
    }

    // generate random labels to delete them from index
    std::vector<int> rand_labels(max_elements);
    for (int i = 0; i < max_elements; i++) {
        rand_labels[i] = i;
    }
    std::shuffle(rand_labels.begin(), rand_labels.end(), rng);  // rand_labels[0] = 100有可能

    int iter = 0;
    while (iter < 2) {
        // allow_replace_deleted = true，允许删掉的向量被替换
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, 16, 200, 123, true);

        // add batch1 data，2000个数据
        ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
            alg_hnsw->addPoint((void*)(batch1 + d * row), row);
        });

        // delete half random elements of batch1 data
        for (int i = 0; i < num_elements; i++) {
            alg_hnsw->markDelete(rand_labels[i]);   // markDelete(labeltype label)，把2000个中的随机的1000个设置为删除
        }

        // replace deleted elements with batch2 data
        ParallelFor(0, num_elements, num_threads, [&](size_t row, size_t threadId) {    // row是id，也就是1到999
            int label = rand_labels[row] + max_elements;    // label的值范围是2000到4000
            alg_hnsw->addPoint((void*)(batch2 + d * row), label, true);
        });

        iter += 1;

        delete alg_hnsw;
    }
    
    std::cout << "Finish" << std::endl;

    delete[] batch1;
    delete[] batch2;
    return 0;
}
