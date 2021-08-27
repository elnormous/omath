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
    template <class T, std::size_t cols, std::size_t rows>
    struct canUseSimd: public std::false_type {};

#if defined(__SSE__) || defined(__ARM_NEON__)
    template <>
    struct canUseSimd<float, 4, 4>: public std::true_type {};
#endif

    template <class T, std::size_t cols, std::size_t rows>
    inline constexpr bool canUseSimdValue = canUseSimd<T, cols, rows>::value;
}

#endif
