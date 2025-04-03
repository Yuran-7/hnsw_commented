#include <iostream>
#include <random> // 包含随机数库

int main() {
    // 1. 创建随机数引擎
    std::random_device rd;  // 用于生成随机种子
    std::cout << rd() << std::endl;
    std::cout << "Type of rd(): " << typeid(rd()).name() << std::endl;
    std::mt19937 gen(rd()); // 使用 Mersenne Twister 引擎

    // 2. 创建均匀分布的整数分布（范围：1 到 100）
    std::uniform_int_distribution<> dis(1, 100);

    // 3. 生成随机数
    for (int i = 0; i < 10; ++i) {
        std::cout << dis(gen) << " "; // 生成并输出随机数
    }

    return 0;
}