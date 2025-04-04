#include <iostream>
#include <chrono>

using namespace std;

// 计算加速比 speedup = t_slow / t_fast
double compute_speedup(double slow_time, double fast_time) {
    return slow_time / fast_time;
}

int main() {
    const size_t NUM_ITER = 10'000'000;  // 1 千万次循环

    // 1. 带 cout 的测试
    auto start1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_ITER; i++) {
        cout << "当前 i: " << i << "\r";  // 使用 \r 进行覆盖式输出，减少终端输出行数
    }
    cout << endl;  // 确保最后输出完整
    auto end1 = chrono::high_resolution_clock::now();
    double time_with_cout = chrono::duration<double>(end1 - start1).count();

    // 2. 不带 cout 的测试
    auto start2 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < NUM_ITER; i++) {
        volatile size_t x = i;  // 使用 volatile 防止编译器优化掉循环
    }
    auto end2 = chrono::high_resolution_clock::now();
    double time_without_cout = chrono::duration<double>(end2 - start2).count();

    // 计算加速比
    double speedup = compute_speedup(time_with_cout, time_without_cout);

    // 输出结果
    cout << "带 cout 时间: " << time_with_cout << " 秒" << endl;
    cout << "不带 cout 时间: " << time_without_cout << " 秒" << endl;
    cout << "加速比 (speedup): " << speedup << "x" << endl;

    return 0;
}
