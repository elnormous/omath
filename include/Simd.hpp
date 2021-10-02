//
// elnormous/omath
//

#ifndef OMATH_SIMD
#define OMATH_SIMD

#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
#  include <xmmintrin.h>
#  define OMATH_SIMD_AVAILABLE
#  define OMATH_SIMD_SSE
#elif defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define OMATH_SIMD_AVAILABLE
#  define OMATH_SIMD_NEON
#endif

namespace omath
{
    template <typename T, std::size_t dims>
    struct CanVectorUseSimd: std::false_type {};

    template <typename T, std::size_t rows, std::size_t cols>
    struct CanMatrixUseSimd: std::false_type {};

#ifdef OMATH_SIMD_AVAILABLE
    template <>
    struct CanVectorUseSimd<float, 4>: std::true_type {};

    template <>
    struct CanMatrixUseSimd<float, 4, 4>: std::true_type {};
#endif

    template <typename T, std::size_t dims>
    inline constexpr bool canVectorUseSimd = CanVectorUseSimd<T, dims>::value;

    template <typename T, std::size_t rows, std::size_t cols>
    inline constexpr bool canMatrixUseSimd = CanMatrixUseSimd<T, rows, cols>::value;
}

#endif
