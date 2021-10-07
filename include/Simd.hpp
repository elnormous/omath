//
// elnormous/omath
//

#ifndef OMATH_SIMD
#define OMATH_SIMD

#if !OMATH_DISABLE_SIMD
#  if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP >= 1
#    define OMATH_SIMD_SSE
#  endif
#  if defined(__SSE2__) || defined(_M_X64) || _M_IX86_FP >= 2
#    define OMATH_SIMD_SSE2
#  endif
#  if defined(__ARM_NEON__)
#    define OMATH_SIMD_NEON
#    if defined(__x86_64__)
#      define OMATH_SIMD_NEON64
#    endif
#  endif
#endif

#endif
