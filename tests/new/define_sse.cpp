#include <iostream>
#include "../../hnswlib/hnswlib.h"

void test_sse_macro() {
#ifdef __SSE__
    std::cout << "__SSE__ is defined" << std::endl;
#else
    std::cout << "__SSE__ is NOT defined" << std::endl;
#endif

#ifdef USE_SSE
    std::cout << "USE_SSE is defined" << std::endl;
#else
    std::cout << "USE_SSE is NOT defined" << std::endl;
#endif

#ifdef __SSE2__
    std::cout << "__SSE2__ is defined" << std::endl;
#else
    std::cout << "__SSE2__ is NOT defined" << std::endl;
#endif

#ifdef __AVX__
    std::cout << "__AVX__ is defined" << std::endl;
#else
    std::cout << "__AVX__ is NOT defined" << std::endl;
#endif

#ifdef HNSWERR
    std::cout << "HNSWLIB_ERR_OVERRIDE is defined" << std::endl;
#else
    std::cout << "HNSWLIB_ERR_OVERRIDE is NOT defined" << std::endl;
#endif

#ifdef NO_MANUAL_VECTORIZATION
    std::cout << "NO_MANUAL_VECTORIZATION is defined" << std::endl;
#else
    std::cout << "NO_MANUAL_VECTORIZATION is NOT defined" << std::endl;
#endif

#ifdef _M_IX86_FP
    std::cout << "_M_IX86_FP is defined" << std::endl;
#else
    std::cout << "_M_IX86_FP is NOT defined" << std::endl;
#endif

#ifdef _M_AMD64
    std::cout << "_M_AMD64 is defined" << std::endl;
#else
    std::cout << "_M_AMD64 is NOT defined" << std::endl;
#endif

#ifdef _M_X64
    std::cout << "_M_X64 is defined" << std::endl;
#else
    std::cout << "_M_X64 is NOT defined" << std::endl;
#endif

#ifdef __GNUC__
    std::cout << "__GNUC__ is defined" << std::endl;
#else
    std::cout << "__GNUC__ is NOT defined" << std::endl;
#endif


}

int main() {
    test_sse_macro();
    std::cout << AVXCapable() << std::endl;
    std::cout << AVX512Capable() << std::endl;
    return 0;
}
