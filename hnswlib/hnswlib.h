#pragma once

// https://github.com/nmslib/hnswlib/pull/508
// This allows others to provide their own error stream (e.g. RcppHNSW)

// 这个的目的就是只要我们没有手动添加这个宏，就会用std::cerr替代HNSWERR
#ifndef HNSWLIB_ERR_OVERRIDE    
  #define HNSWERR std::cerr
#else
  #define HNSWERR HNSWLIB_ERR_OVERRIDE
#endif

#ifndef NO_MANUAL_VECTORIZATION  // 如果没有定义 NO_MANUAL_VECTORIZATION
  // 这些宏通常与处理器架构和编译器支持的指令集相关
  #if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))     
    // 如果支持 SSE 或者是 x86/x64 架构
    #define USE_SSE  // 启用 SSE 向量化优化
    #ifdef __AVX__  // 如果编译器定义了 AVX 支持
      #define USE_AVX  // 启用 AVX 优化
      //#ifdef __AVX512F__  // 如果编译器定义了 AVX512 支持
       // #define USE_AVX512  // 启用 AVX512 优化
      //#endif
    #endif
  #endif
#endif


#if defined(USE_AVX) || defined(USE_SSE)    // #if defined(USE_AVX) 和 #ifdef USE_AVX 没有区别，但前者更灵活
  #ifdef _MSC_VER
    #include <intrin.h> // 是 Microsoft Visual C++（MSVC）编译器提供的一个头文件，用于访问编译器内置函数（intrinsics）
    #include <stdexcept>    // 是 C++ 标准库提供的一个头文件，用于访问异常类
    // cpuid函数，获取CPU信息
    static void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
        __cpuidex(out, eax, ecx);
    }
    static __int64 xgetbv(unsigned int x) {
        return _xgetbv(x);
    }
  #else
    #include <x86intrin.h>  // 是 GCC 和 Clang 提供的一个头文件，用于访问编译器内置函数（intrinsics）
    #include <cpuid.h>
    #include <stdint.h>
    static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
        __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);    // 在 cpuid.h 中定义
    }
    static uint64_t xgetbv(unsigned int index) {
        uint32_t eax, edx;
        __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
        return ((uint64_t)edx << 32) | eax;
    }
  #endif

  #if defined(USE_AVX512)
    #include <immintrin.h>  // win和linux下都有
  #endif

  #if defined(__GNUC__)
    #define PORTABLE_ALIGN32 __attribute__((aligned(32)))   // 将变量或结构体对齐到 32 字节边界
    #define PORTABLE_ALIGN64 __attribute__((aligned(64)))   // 将变量或结构体对齐到 64 字节边界
  // MSVC 使用 __declspec(align(32))
  #else
    #define PORTABLE_ALIGN32 __declspec(align(32))
    #define PORTABLE_ALIGN64 __declspec(align(64))
  #endif

// Adapted from https://github.com/Mysticial/FeatureDetector
  #define _XCR_XFEATURE_ENABLED_MASK  0

  static bool AVXCapable() {
      int cpuInfo[4];
  
      // CPU support，检查cpu是否支持AVX
      cpuid(cpuInfo, 0, 0);
      int nIds = cpuInfo[0];
  
      bool HW_AVX = false;
      if (nIds >= 0x00000001) {
          cpuid(cpuInfo, 0x00000001, 0);
          HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
      }
  
      // OS support，检查操作系统是否支持AVX
      cpuid(cpuInfo, 1, 0);
  
      bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
      bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;
  
      bool avxSupported = false;
      if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
          uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
          avxSupported = (xcrFeatureMask & 0x6) == 0x6;
      }
      return HW_AVX && avxSupported;    // 返回cpu和os是否都支持AVX
  }
  
  static bool AVX512Capable() {
      if (!AVXCapable()) return false;
  
      int cpuInfo[4];
  
      // CPU support
      cpuid(cpuInfo, 0, 0);
      int nIds = cpuInfo[0];
  
      bool HW_AVX512F = false;
      if (nIds >= 0x00000007) {  //  AVX512 Foundation
          cpuid(cpuInfo, 0x00000007, 0);
          HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
      }
  
      // OS support
      cpuid(cpuInfo, 1, 0);
  
      bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
      bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;
  
      bool avx512Supported = false;
      if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
          uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
          avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
      }
      return HW_AVX512F && avx512Supported;
  }
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

namespace hnswlib {
typedef size_t labeltype;

// This can be extended to store state for filtering (e.g. from a std::set)
class BaseFilterFunctor {
 public:
    virtual bool operator()(hnswlib::labeltype id) { return true; } // 继承不能修改参数列表
    virtual ~BaseFilterFunctor() {};
};

template<typename dist_t>
class BaseSearchStopCondition {
 public:
    virtual void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    virtual void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_remove_extra() = 0;

    virtual void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) = 0;

    virtual ~BaseSearchStopCondition() {}
};

template <typename T>
class pairGreater {
 public:
    bool operator()(const T& p1, const T& p2) {
        return p1.first > p2.first;
    }
};

template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

// DISTFUNC<dist_t> fstdistfunc_;
// 函数指针的声明格式为：返回类型 (*指针变量名)(参数列表)
// float(*DISTFUNC)(const void *, const void *, const void *);
// DISTFUNC = SomeFunction;
template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

// class L2Space : public SpaceInterface<float>
// L2Space l2space(dim_base);
// appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, num_base, M, efConstruction);
template<typename MTYPE>
class SpaceInterface {
 public:
    // virtual void search(void *);
    virtual size_t get_data_size() = 0;

    virtual DISTFUNC<MTYPE> get_dist_func() = 0;

    virtual void *get_dist_func_param() = 0;

    virtual ~SpaceInterface() {}
};

template<typename dist_t>
class AlgorithmInterface {  // 抽象类（包含一个或多个纯虚函数），抽象类和接口是一个东西
 public:
    virtual void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) = 0;    // 纯虚函数，必须由子类实现

    virtual std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void*, size_t, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    // Return k nearest neighbor in the order of closer fist
    virtual std::vector<std::pair<dist_t, labeltype>>
        searchKnnCloserFirst(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const; // 非纯虚函数

    virtual void saveIndex(const std::string &location) = 0;
    virtual ~AlgorithmInterface(){
    }
};

template<typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k,
                                                 BaseFilterFunctor* isIdAllowed) const {
    std::vector<std::pair<dist_t, labeltype>> result;

    // here searchKnn returns the result in the order of further first
    auto ret = searchKnn(query_data, k, isIdAllowed);
    {
        size_t sz = ret.size();
        result.resize(sz);
        while (!ret.empty()) {
            result[--sz] = ret.top();
            ret.pop();
        }
    }

    return result;
}
}  // namespace hnswlib

#include "space_l2.h"
#include "space_ip.h"
#include "stop_condition.h"
#include "bruteforce.h"
#include "hnswalg.h"
