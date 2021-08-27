//
// elnormous/math
//

#ifndef OMATH_SIMD
#define OMATH_SIMD

#include <type_traits>
#ifdef __SSE__
#  include <xmmintrin.h>
#elif defined(__ARM_NEON__)
#  include <arm_neon.h>
#endif

namespace omath
{
    template <class T, std::size_t n>
    inline constexpr bool canVectorUseSimd = false;

    template <class T, std::size_t cols, std::size_t rows>
    inline constexpr bool canMatrixUseSimd = false;

#if defined(__SSE__) || defined(__ARM_NEON__)
    template <>
    inline constexpr bool canVectorUseSimd<float, 4> = true;

    template <>
    inline constexpr bool canMatrixUseSimd<float, 4, 4> = true;
#endif
}

#endif
