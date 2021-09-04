//
// elnormous/omath
//

#ifndef OMATH_SIMD
#define OMATH_SIMD

#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
#  include <xmmintrin.h>
#elif defined(__ARM_NEON__)
#  include <arm_neon.h>
#endif

namespace omath
{
    template <class T, std::size_t dims>
    struct CanVectorUseSimd: std::false_type {};

    template <class T, std::size_t cols, std::size_t rows>
    struct CanMatrixUseSimd: std::false_type {};

    #if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0 || defined(__ARM_NEON__)
    template <>
    struct CanVectorUseSimd<float, 4>: std::true_type {};

    template <>
    struct CanMatrixUseSimd<float, 4, 4>: std::true_type {};
    #endif

    template <class T, std::size_t dims>
    inline constexpr bool canVectorUseSimd = CanVectorUseSimd<T, dims>::value;

    template <class T, std::size_t cols, std::size_t rows>
    inline constexpr bool canMatrixUseSimd = CanMatrixUseSimd<T, cols, rows>::value;
}

#endif
