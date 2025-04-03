#include <iostream>

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

#ifdef __AVX__
    std::cout << "__AVX__ is defined" << std::endl;
#else
    std::cout << "__AVX__ is NOT defined" << std::endl;
#endif

}

int main() {
    test_sse_macro();
    return 0;
}
