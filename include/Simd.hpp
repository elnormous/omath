//
// elnormous/omath
//

#ifndef OMATH_SIMD
#define OMATH_SIMD

#ifndef OMATH_DISABLE_SIMD
#  if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP > 0
#    include <xmmintrin.h>
#    define OMATH_SIMD_SSE
#  elif defined(__ARM_NEON__)
#    include <arm_neon.h>
#    define OMATH_SIMD_NEON
#  endif
#endif

#endif
