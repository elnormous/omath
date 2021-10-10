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

    template <>
    [[nodiscard]] inline auto operator+(const Matrix<double, 4, 4>& matrix1,
                                        const Matrix<double, 4, 4>& matrix2) noexcept
    {
        Matrix<double, 4, 4> result;
        vst1q_f64(&result.m[0], vaddq_f64(vld1q_f64(&matrix1.m[0]), vld1q_f64(&matrix2.m[0])));
        vst1q_f64(&result.m[2], vaddq_f64(vld1q_f64(&matrix1.m[2]), vld1q_f64(&matrix2.m[2])));
        vst1q_f64(&result.m[4], vaddq_f64(vld1q_f64(&matrix1.m[4]), vld1q_f64(&matrix2.m[4])));
        vst1q_f64(&result.m[6], vaddq_f64(vld1q_f64(&matrix1.m[6]), vld1q_f64(&matrix2.m[6])));
        vst1q_f64(&result.m[8], vaddq_f64(vld1q_f64(&matrix1.m[8]), vld1q_f64(&matrix2.m[8])));
        vst1q_f64(&result.m[10], vaddq_f64(vld1q_f64(&matrix1.m[10]), vld1q_f64(&matrix2.m[10])));
        vst1q_f64(&result.m[12], vaddq_f64(vld1q_f64(&matrix1.m[12]), vld1q_f64(&matrix2.m[12])));
        vst1q_f64(&result.m[14], vaddq_f64(vld1q_f64(&matrix1.m[14]), vld1q_f64(&matrix2.m[14])));
        return result;
    }

    template <>
    inline auto& operator+=(Matrix<double, 4, 4>& matrix1,
                            const Matrix<double, 4, 4>& matrix2) noexcept
    {
        vst1q_f64(&matrix1.m[0], vaddq_f64(vld1q_f64(&matrix1.m[0]), vld1q_f64(&matrix2.m[0])));
        vst1q_f64(&matrix1.m[2], vaddq_f64(vld1q_f64(&matrix1.m[2]), vld1q_f64(&matrix2.m[2])));
        vst1q_f64(&matrix1.m[4], vaddq_f64(vld1q_f64(&matrix1.m[4]), vld1q_f64(&matrix2.m[4])));
        vst1q_f64(&matrix1.m[6], vaddq_f64(vld1q_f64(&matrix1.m[6]), vld1q_f64(&matrix2.m[6])));
        vst1q_f64(&matrix1.m[8], vaddq_f64(vld1q_f64(&matrix1.m[8]), vld1q_f64(&matrix2.m[8])));
        vst1q_f64(&matrix1.m[10], vaddq_f64(vld1q_f64(&matrix1.m[10]), vld1q_f64(&matrix2.m[10])));
        vst1q_f64(&matrix1.m[12], vaddq_f64(vld1q_f64(&matrix1.m[12]), vld1q_f64(&matrix2.m[12])));
        vst1q_f64(&matrix1.m[14], vaddq_f64(vld1q_f64(&matrix1.m[14]), vld1q_f64(&matrix2.m[14])));
        return matrix1;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Matrix<double, 4, 4>& matrix1,
                                        const Matrix<double, 4, 4>& matrix2) noexcept
    {
        Matrix<double, 4, 4> result;
        vst1q_f64(&result.m[0], vsubq_f64(vld1q_f64(&matrix1.m[0]), vld1q_f64(&matrix2.m[0])));
        vst1q_f64(&result.m[2], vsubq_f64(vld1q_f64(&matrix1.m[2]), vld1q_f64(&matrix2.m[2])));
        vst1q_f64(&result.m[4], vsubq_f64(vld1q_f64(&matrix1.m[4]), vld1q_f64(&matrix2.m[4])));
        vst1q_f64(&result.m[6], vsubq_f64(vld1q_f64(&matrix1.m[6]), vld1q_f64(&matrix2.m[6])));
        vst1q_f64(&result.m[8], vsubq_f64(vld1q_f64(&matrix1.m[8]), vld1q_f64(&matrix2.m[8])));
        vst1q_f64(&result.m[10], vsubq_f64(vld1q_f64(&matrix1.m[10]), vld1q_f64(&matrix2.m[10])));
        vst1q_f64(&result.m[12], vsubq_f64(vld1q_f64(&matrix1.m[12]), vld1q_f64(&matrix2.m[12])));
        vst1q_f64(&result.m[14], vsubq_f64(vld1q_f64(&matrix1.m[14]), vld1q_f64(&matrix2.m[14])));
        return result;
    }

    template <>
    inline auto& operator-=(Matrix<double, 4, 4>& matrix1,
                            const Matrix<double, 4, 4>& matrix2) noexcept
    {
        vst1q_f64(&matrix1.m[0], vsubq_f64(vld1q_f64(&matrix1.m[0]), vld1q_f64(&matrix2.m[0])));
        vst1q_f64(&matrix1.m[2], vsubq_f64(vld1q_f64(&matrix1.m[2]), vld1q_f64(&matrix2.m[2])));
        vst1q_f64(&matrix1.m[4], vsubq_f64(vld1q_f64(&matrix1.m[4]), vld1q_f64(&matrix2.m[4])));
        vst1q_f64(&matrix1.m[6], vsubq_f64(vld1q_f64(&matrix1.m[6]), vld1q_f64(&matrix2.m[6])));
        vst1q_f64(&matrix1.m[8], vsubq_f64(vld1q_f64(&matrix1.m[8]), vld1q_f64(&matrix2.m[8])));
        vst1q_f64(&matrix1.m[10], vsubq_f64(vld1q_f64(&matrix1.m[10]), vld1q_f64(&matrix2.m[10])));
        vst1q_f64(&matrix1.m[12], vsubq_f64(vld1q_f64(&matrix1.m[12]), vld1q_f64(&matrix2.m[12])));
        vst1q_f64(&matrix1.m[14], vsubq_f64(vld1q_f64(&matrix1.m[14]), vld1q_f64(&matrix2.m[14])));
        return matrix1;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Matrix<double, 4, 4>& matrix,
                                        const double scalar) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto s = vdupq_n_f64(scalar);
        vst1q_f64(&result.m[0], vmulq_f64(vld1q_f64(&matrix.m[0]), s));
        vst1q_f64(&result.m[2], vmulq_f64(vld1q_f64(&matrix.m[2]), s));
        vst1q_f64(&result.m[4], vmulq_f64(vld1q_f64(&matrix.m[4]), s));
        vst1q_f64(&result.m[6], vmulq_f64(vld1q_f64(&matrix.m[6]), s));
        vst1q_f64(&result.m[8], vmulq_f64(vld1q_f64(&matrix.m[8]), s));
        vst1q_f64(&result.m[10], vmulq_f64(vld1q_f64(&matrix.m[10]), s));
        vst1q_f64(&result.m[12], vmulq_f64(vld1q_f64(&matrix.m[12]), s));
        vst1q_f64(&result.m[14], vmulq_f64(vld1q_f64(&matrix.m[14]), s));
        return result;
    }

    template <>
    inline auto& operator*=(Matrix<double, 4, 4>& matrix,
                            const double scalar) noexcept
    {
        const auto s = vdupq_n_f64(scalar);
        vst1q_f64(&matrix.m[0], vmulq_f64(vld1q_f64(&matrix.m[0]), s));
        vst1q_f64(&matrix.m[2], vmulq_f64(vld1q_f64(&matrix.m[2]), s));
        vst1q_f64(&matrix.m[4], vmulq_f64(vld1q_f64(&matrix.m[4]), s));
        vst1q_f64(&matrix.m[6], vmulq_f64(vld1q_f64(&matrix.m[6]), s));
        vst1q_f64(&matrix.m[8], vmulq_f64(vld1q_f64(&matrix.m[8]), s));
        vst1q_f64(&matrix.m[10], vmulq_f64(vld1q_f64(&matrix.m[10]), s));
        vst1q_f64(&matrix.m[12], vmulq_f64(vld1q_f64(&matrix.m[12]), s));
        vst1q_f64(&matrix.m[14], vmulq_f64(vld1q_f64(&matrix.m[14]), s));
        return matrix;
    }

    template <>
    [[nodiscard]] inline auto operator/(const Matrix<double, 4, 4>& matrix,
                                        double scalar) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto s = vdupq_n_f64(scalar);
        vst1q_f64(&result.m[0], vdivq_f64(vld1q_f64(&matrix.m[0]), s));
        vst1q_f64(&result.m[2], vdivq_f64(vld1q_f64(&matrix.m[2]), s));
        vst1q_f64(&result.m[4], vdivq_f64(vld1q_f64(&matrix.m[4]), s));
        vst1q_f64(&result.m[6], vdivq_f64(vld1q_f64(&matrix.m[6]), s));
        vst1q_f64(&result.m[8], vdivq_f64(vld1q_f64(&matrix.m[8]), s));
        vst1q_f64(&result.m[10], vdivq_f64(vld1q_f64(&matrix.m[10]), s));
        vst1q_f64(&result.m[12], vdivq_f64(vld1q_f64(&matrix.m[12]), s));
        vst1q_f64(&result.m[14], vdivq_f64(vld1q_f64(&matrix.m[14]), s));
        return result;
    }

    template <>
    inline auto& operator/=(Matrix<double, 4, 4>& matrix,
                            const double scalar) noexcept
    {
        const auto s = vdupq_n_f64(scalar);
        vst1q_f64(&matrix.m[0], vdivq_f64(vld1q_f64(&matrix.m[0]), s));
        vst1q_f64(&matrix.m[2], vdivq_f64(vld1q_f64(&matrix.m[2]), s));
        vst1q_f64(&matrix.m[4], vdivq_f64(vld1q_f64(&matrix.m[4]), s));
        vst1q_f64(&matrix.m[6], vdivq_f64(vld1q_f64(&matrix.m[6]), s));
        vst1q_f64(&matrix.m[8], vdivq_f64(vld1q_f64(&matrix.m[8]), s));
        vst1q_f64(&matrix.m[10], vdivq_f64(vld1q_f64(&matrix.m[10]), s));
        vst1q_f64(&matrix.m[12], vdivq_f64(vld1q_f64(&matrix.m[12]), s));
        vst1q_f64(&matrix.m[14], vdivq_f64(vld1q_f64(&matrix.m[14]), s));
        return matrix;
    }
}

#endif

#endif // OMATH_MATRIX_NEON64
