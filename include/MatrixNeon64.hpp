//
// elnormous/omath
//

#ifndef OMATH_MATRIX_NEON64
#define OMATH_MATRIX_NEON64

#include "Matrix.hpp"
#include "Simd.hpp"

#ifdef OMATH_SIMD_NEON64
#  include <arm_neon.h>

namespace omath
{
    template <>
    [[nodiscard]] inline auto operator-(const Matrix<double, 4, 4>& matrix) noexcept
    {
        Matrix<double, 4, 4> result;
        vst1q_f64(&result.m[0], vnegq_f64(vld1q_f64(&matrix.m[0])));
        vst1q_f64(&result.m[2], vnegq_f64(vld1q_f64(&matrix.m[2])));
        vst1q_f64(&result.m[4], vnegq_f64(vld1q_f64(&matrix.m[4])));
        vst1q_f64(&result.m[6], vnegq_f64(vld1q_f64(&matrix.m[6])));
        vst1q_f64(&result.m[8], vnegq_f64(vld1q_f64(&matrix.m[8])));
        vst1q_f64(&result.m[10], vnegq_f64(vld1q_f64(&matrix.m[10])));
        vst1q_f64(&result.m[12], vnegq_f64(vld1q_f64(&matrix.m[12])));
        vst1q_f64(&result.m[14], vnegq_f64(vld1q_f64(&matrix.m[14])));
        return result;
    }

    template <>
    inline void negate(Matrix<double, 4, 4>& matrix) noexcept
    {
        vst1q_f64(&matrix.m[0], vnegq_f64(vld1q_f64(&matrix.m[0])));
        vst1q_f64(&matrix.m[2], vnegq_f64(vld1q_f64(&matrix.m[2])));
        vst1q_f64(&matrix.m[4], vnegq_f64(vld1q_f64(&matrix.m[4])));
        vst1q_f64(&matrix.m[6], vnegq_f64(vld1q_f64(&matrix.m[6])));
        vst1q_f64(&matrix.m[8], vnegq_f64(vld1q_f64(&matrix.m[8])));
        vst1q_f64(&matrix.m[10], vnegq_f64(vld1q_f64(&matrix.m[10])));
        vst1q_f64(&matrix.m[12], vnegq_f64(vld1q_f64(&matrix.m[12])));
        vst1q_f64(&matrix.m[14], vnegq_f64(vld1q_f64(&matrix.m[14])));
    }
}

#endif

#endif // OMATH_MATRIX_NEON64
