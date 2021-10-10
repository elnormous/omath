//
// elnormous/omath
//

#ifndef OMATH_MATRIX_SSE2
#define OMATH_MATRIX_SSE2

#include "Matrix.hpp"
#include "Simd.hpp"

#ifdef OMATH_SIMD_SSE
#  include <xmmintrin.h>

namespace omath
{
    template <>
    [[nodiscard]] inline auto operator-(const Matrix<double, 4, 4>& matrix) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto z = _mm_setzero_pd();
        _mm_store_pd(&result.m[0], _mm_sub_pd(z, _mm_load_pd(&matrix.m[0])));
        _mm_store_pd(&result.m[2], _mm_sub_pd(z, _mm_load_pd(&matrix.m[2])));
        _mm_store_pd(&result.m[4], _mm_sub_pd(z, _mm_load_pd(&matrix.m[4])));
        _mm_store_pd(&result.m[6], _mm_sub_pd(z, _mm_load_pd(&matrix.m[6])));
        _mm_store_pd(&result.m[8], _mm_sub_pd(z, _mm_load_pd(&matrix.m[8])));
        _mm_store_pd(&result.m[10], _mm_sub_pd(z, _mm_load_pd(&matrix.m[10])));
        _mm_store_pd(&result.m[12], _mm_sub_pd(z, _mm_load_pd(&matrix.m[12])));
        _mm_store_pd(&result.m[14], _mm_sub_pd(z, _mm_load_pd(&matrix.m[14])));
        return result;
    }

    template <>
    inline void negate(Matrix<double, 4, 4>& matrix) noexcept
    {
        const auto z = _mm_setzero_pd();
        _mm_store_pd(&matrix.m[0], _mm_sub_pd(z, _mm_load_pd(&matrix.m[0])));
        _mm_store_pd(&matrix.m[2], _mm_sub_pd(z, _mm_load_pd(&matrix.m[2])));
        _mm_store_pd(&matrix.m[4], _mm_sub_pd(z, _mm_load_pd(&matrix.m[4])));
        _mm_store_pd(&matrix.m[6], _mm_sub_pd(z, _mm_load_pd(&matrix.m[6])));
        _mm_store_pd(&matrix.m[8], _mm_sub_pd(z, _mm_load_pd(&matrix.m[8])));
        _mm_store_pd(&matrix.m[10], _mm_sub_pd(z, _mm_load_pd(&matrix.m[10])));
        _mm_store_pd(&matrix.m[12], _mm_sub_pd(z, _mm_load_pd(&matrix.m[12])));
        _mm_store_pd(&matrix.m[14], _mm_sub_pd(z, _mm_load_pd(&matrix.m[14])));
    }

    template <>
    [[nodiscard]] inline auto operator+(const Matrix<double, 4, 4>& matrix1,
                                        const Matrix<double, 4, 4>& matrix2) noexcept
    {
        Matrix<double, 4, 4> result;
        _mm_store_pd(&result.m[0], _mm_add_pd(_mm_load_pd(&matrix1.m[0]), _mm_load_pd(&matrix2.m[0])));
        _mm_store_pd(&result.m[2], _mm_add_pd(_mm_load_pd(&matrix1.m[2]), _mm_load_pd(&matrix2.m[2])));
        _mm_store_pd(&result.m[4], _mm_add_pd(_mm_load_pd(&matrix1.m[4]), _mm_load_pd(&matrix2.m[4])));
        _mm_store_pd(&result.m[6], _mm_add_pd(_mm_load_pd(&matrix1.m[6]), _mm_load_pd(&matrix2.m[6])));
        _mm_store_pd(&result.m[8], _mm_add_pd(_mm_load_pd(&matrix1.m[8]), _mm_load_pd(&matrix2.m[8])));
        _mm_store_pd(&result.m[10], _mm_add_pd(_mm_load_pd(&matrix1.m[10]), _mm_load_pd(&matrix2.m[10])));
        _mm_store_pd(&result.m[12], _mm_add_pd(_mm_load_pd(&matrix1.m[12]), _mm_load_pd(&matrix2.m[12])));
        _mm_store_pd(&result.m[14], _mm_add_pd(_mm_load_pd(&matrix1.m[14]), _mm_load_pd(&matrix2.m[14])));
        return result;
    }

    template <>
    inline auto& operator+=(Matrix<double, 4, 4>& matrix1,
                            const Matrix<double, 4, 4>& matrix2) noexcept
    {
        _mm_store_pd(&matrix1.m[0], _mm_add_pd(_mm_load_pd(&matrix1.m[0]), _mm_load_pd(&matrix2.m[0])));
        _mm_store_pd(&matrix1.m[2], _mm_add_pd(_mm_load_pd(&matrix1.m[2]), _mm_load_pd(&matrix2.m[2])));
        _mm_store_pd(&matrix1.m[4], _mm_add_pd(_mm_load_pd(&matrix1.m[4]), _mm_load_pd(&matrix2.m[4])));
        _mm_store_pd(&matrix1.m[6], _mm_add_pd(_mm_load_pd(&matrix1.m[6]), _mm_load_pd(&matrix2.m[6])));
        _mm_store_pd(&matrix1.m[8], _mm_add_pd(_mm_load_pd(&matrix1.m[8]), _mm_load_pd(&matrix2.m[8])));
        _mm_store_pd(&matrix1.m[10], _mm_add_pd(_mm_load_pd(&matrix1.m[10]), _mm_load_pd(&matrix2.m[10])));
        _mm_store_pd(&matrix1.m[12], _mm_add_pd(_mm_load_pd(&matrix1.m[12]), _mm_load_pd(&matrix2.m[12])));
        _mm_store_pd(&matrix1.m[14], _mm_add_pd(_mm_load_pd(&matrix1.m[14]), _mm_load_pd(&matrix2.m[14])));
        return matrix1;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Matrix<double, 4, 4>& matrix1,
                                        const Matrix<double, 4, 4>& matrix2) noexcept
    {
        Matrix<double, 4, 4> result;
        _mm_store_pd(&result.m[0], _mm_sub_pd(_mm_load_pd(&matrix1.m[0]), _mm_load_pd(&matrix2.m[0])));
        _mm_store_pd(&result.m[2], _mm_sub_pd(_mm_load_pd(&matrix1.m[2]), _mm_load_pd(&matrix2.m[2])));
        _mm_store_pd(&result.m[4], _mm_sub_pd(_mm_load_pd(&matrix1.m[4]), _mm_load_pd(&matrix2.m[4])));
        _mm_store_pd(&result.m[6], _mm_sub_pd(_mm_load_pd(&matrix1.m[6]), _mm_load_pd(&matrix2.m[6])));
        _mm_store_pd(&result.m[8], _mm_sub_pd(_mm_load_pd(&matrix1.m[8]), _mm_load_pd(&matrix2.m[8])));
        _mm_store_pd(&result.m[10], _mm_sub_pd(_mm_load_pd(&matrix1.m[10]), _mm_load_pd(&matrix2.m[10])));
        _mm_store_pd(&result.m[12], _mm_sub_pd(_mm_load_pd(&matrix1.m[12]), _mm_load_pd(&matrix2.m[12])));
        _mm_store_pd(&result.m[14], _mm_sub_pd(_mm_load_pd(&matrix1.m[14]), _mm_load_pd(&matrix2.m[14])));
        return result;
    }

    template <>
    inline auto& operator-=(Matrix<double, 4, 4>& matrix1,
                            const Matrix<double, 4, 4>& matrix2) noexcept
    {
        _mm_store_pd(&matrix1.m[0], _mm_sub_pd(_mm_load_pd(&matrix1.m[0]), _mm_load_pd(&matrix2.m[0])));
        _mm_store_pd(&matrix1.m[2], _mm_sub_pd(_mm_load_pd(&matrix1.m[2]), _mm_load_pd(&matrix2.m[2])));
        _mm_store_pd(&matrix1.m[4], _mm_sub_pd(_mm_load_pd(&matrix1.m[4]), _mm_load_pd(&matrix2.m[4])));
        _mm_store_pd(&matrix1.m[6], _mm_sub_pd(_mm_load_pd(&matrix1.m[6]), _mm_load_pd(&matrix2.m[6])));
        _mm_store_pd(&matrix1.m[8], _mm_sub_pd(_mm_load_pd(&matrix1.m[8]), _mm_load_pd(&matrix2.m[8])));
        _mm_store_pd(&matrix1.m[10], _mm_sub_pd(_mm_load_pd(&matrix1.m[10]), _mm_load_pd(&matrix2.m[10])));
        _mm_store_pd(&matrix1.m[12], _mm_sub_pd(_mm_load_pd(&matrix1.m[12]), _mm_load_pd(&matrix2.m[12])));
        _mm_store_pd(&matrix1.m[14], _mm_sub_pd(_mm_load_pd(&matrix1.m[14]), _mm_load_pd(&matrix2.m[14])));
        return matrix1;
    }
    
    template <>
    [[nodiscard]] inline auto operator*(const Matrix<double, 4, 4>& matrix,
                                        const double scalar) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&result.m[0], _mm_mul_pd(_mm_load_pd(&matrix.m[0]), s));
        _mm_store_pd(&result.m[2], _mm_mul_pd(_mm_load_pd(&matrix.m[2]), s));
        _mm_store_pd(&result.m[4], _mm_mul_pd(_mm_load_pd(&matrix.m[4]), s));
        _mm_store_pd(&result.m[6], _mm_mul_pd(_mm_load_pd(&matrix.m[6]), s));
        _mm_store_pd(&result.m[8], _mm_mul_pd(_mm_load_pd(&matrix.m[8]), s));
        _mm_store_pd(&result.m[10], _mm_mul_pd(_mm_load_pd(&matrix.m[10]), s));
        _mm_store_pd(&result.m[12], _mm_mul_pd(_mm_load_pd(&matrix.m[12]), s));
        _mm_store_pd(&result.m[14], _mm_mul_pd(_mm_load_pd(&matrix.m[14]), s));
        return result;
    }

    template <>
    inline auto& operator*=(Matrix<double, 4, 4>& matrix,
                            const double scalar) noexcept
    {
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&matrix.m[0], _mm_mul_pd(_mm_load_pd(&matrix.m[0]), s));
        _mm_store_pd(&matrix.m[2], _mm_mul_pd(_mm_load_pd(&matrix.m[2]), s));
        _mm_store_pd(&matrix.m[4], _mm_mul_pd(_mm_load_pd(&matrix.m[4]), s));
        _mm_store_pd(&matrix.m[6], _mm_mul_pd(_mm_load_pd(&matrix.m[6]), s));
        _mm_store_pd(&matrix.m[8], _mm_mul_pd(_mm_load_pd(&matrix.m[8]), s));
        _mm_store_pd(&matrix.m[10], _mm_mul_pd(_mm_load_pd(&matrix.m[10]), s));
        _mm_store_pd(&matrix.m[12], _mm_mul_pd(_mm_load_pd(&matrix.m[12]), s));
        _mm_store_pd(&matrix.m[14], _mm_mul_pd(_mm_load_pd(&matrix.m[14]), s));
        return matrix;
    }

    template <>
    [[nodiscard]] inline auto operator/(const Matrix<double, 4, 4>& matrix,
                                        double scalar) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&result.m[0], _mm_div_pd(_mm_load_pd(&matrix.m[0]), s));
        _mm_store_pd(&result.m[2], _mm_div_pd(_mm_load_pd(&matrix.m[2]), s));
        _mm_store_pd(&result.m[4], _mm_div_pd(_mm_load_pd(&matrix.m[4]), s));
        _mm_store_pd(&result.m[6], _mm_div_pd(_mm_load_pd(&matrix.m[6]), s));
        _mm_store_pd(&result.m[8], _mm_div_pd(_mm_load_pd(&matrix.m[8]), s));
        _mm_store_pd(&result.m[10], _mm_div_pd(_mm_load_pd(&matrix.m[10]), s));
        _mm_store_pd(&result.m[12], _mm_div_pd(_mm_load_pd(&matrix.m[12]), s));
        _mm_store_pd(&result.m[14], _mm_div_pd(_mm_load_pd(&matrix.m[14]), s));
        return result;
    }

    template <>
    inline auto& operator/=(Matrix<double, 4, 4>& matrix,
                            const double scalar) noexcept
    {
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&matrix.m[0], _mm_div_pd(_mm_load_pd(&matrix.m[0]), s));
        _mm_store_pd(&matrix.m[2], _mm_div_pd(_mm_load_pd(&matrix.m[2]), s));
        _mm_store_pd(&matrix.m[4], _mm_div_pd(_mm_load_pd(&matrix.m[4]), s));
        _mm_store_pd(&matrix.m[6], _mm_div_pd(_mm_load_pd(&matrix.m[6]), s));
        _mm_store_pd(&matrix.m[8], _mm_div_pd(_mm_load_pd(&matrix.m[8]), s));
        _mm_store_pd(&matrix.m[10], _mm_div_pd(_mm_load_pd(&matrix.m[10]), s));
        _mm_store_pd(&matrix.m[12], _mm_div_pd(_mm_load_pd(&matrix.m[12]), s));
        _mm_store_pd(&matrix.m[14], _mm_div_pd(_mm_load_pd(&matrix.m[14]), s));
        return matrix;
    }
}

#endif

#endif // OMATH_MATRIX_SSE2
