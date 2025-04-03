#include<iostream>
#include "../../hnswlib/hnswlib.h"

using namespace std;

int main() {
    int32_t cpuInfo[4];
    cpuid(cpuInfo, 0, 0); // 获取 CPU 厂商信息

    char vendorID[13];
    memcpy(vendorID, &cpuInfo[1], 4); // EBX
    memcpy(vendorID + 4, &cpuInfo[3], 4); // EDX
    memcpy(vendorID + 8, &cpuInfo[2], 4); // ECX
    vendorID[12] = '\0';

    std::cout << "CPU Vendor ID: " << vendorID << std::endl;


    // 检查 XCR0 寄存器的值
    __int64 xcr0 = xgetbv(0);

    // 检查 AVX 支持
    bool avxSupported = (xcr0 & 0x6) == 0x6; // 检查 bit 1 (SSE) 和 bit 2 (AVX) 是否被设置
    std::cout << "AVX supported: " << (avxSupported ? "Yes" : "No") << std::endl;

    if (AVX512Capable()) {
        std::cout << "AVX512 is supported!" << std::endl;
    } else {
        std::cout << "AVX512 is NOT supported!" << std::endl;
    }
    
    return 0;
}