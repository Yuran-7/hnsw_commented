#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

constexpr int kIterations = 100000000;  // 每个线程的修改次数
constexpr int kMaxOffset = 128;         // 最大测试间隔（字节）

// 测试函数
void testCacheLineSize() {
    for (int offset = 1; offset <= kMaxOffset; offset *= 2) {
        // 分配内存，确保变量之间的间隔为 offset 字节
        std::vector<char> data(kMaxOffset * 2);  // 分配足够大的内存
        char* var1 = data.data();
        char* var2 = var1 + offset;

        // 启动两个线程，分别修改 var1 和 var2
        auto start = std::chrono::high_resolution_clock::now();
        std::thread t1([var1]() {
            for (int i = 0; i < kIterations; ++i) {
                (*var1)++;
            }
        });
        std::thread t2([var2]() {
            for (int i = 0; i < kIterations; ++i) {
                (*var2)++;
            }
        });
        t1.join();
        t2.join();
        auto end = std::chrono::high_resolution_clock::now();

        // 计算耗时
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Offset: " << offset << " bytes, Time: " << duration << " ms" << std::endl;

        // 如果耗时显著下降，说明 offset 达到了 Cache 行大小
        if (offset >= 64 && duration < 100) {  // 假设 Cache 行大小为 64 字节
            std::cout << "Inferred Cache Line Size: " << offset << " bytes" << std::endl;
            break;
        }
    }
}

int main() {
    testCacheLineSize();
    return 0;
}