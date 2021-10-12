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

    template <>
    [[nodiscard]] inline auto operator*(const Matrix<double, 4, 4>& matrix1,
                                        const Matrix<double, 4, 4>& matrix2) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto row00 = _mm_load_pd(&matrix1.m[0]);
        const auto row01 = _mm_load_pd(&matrix1.m[2]);
        const auto row10 = _mm_load_pd(&matrix1.m[4]);
        const auto row11 = _mm_load_pd(&matrix1.m[6]);
        const auto row20 = _mm_load_pd(&matrix1.m[8]);
        const auto row21 = _mm_load_pd(&matrix1.m[10]);
        const auto row30 = _mm_load_pd(&matrix1.m[12]);
        const auto row31 = _mm_load_pd(&matrix1.m[14]);

        for (std::size_t i = 0; i < 4; ++i)
        {
            const auto e0 = _mm_set1_pd(matrix2.m[i * 4 + 0]);
            const auto e1 = _mm_set1_pd(matrix2.m[i * 4 + 1]);
            const auto e2 = _mm_set1_pd(matrix2.m[i * 4 + 2]);
            const auto e3 = _mm_set1_pd(matrix2.m[i * 4 + 3]);

            const auto v00 = _mm_mul_pd(row00, e0);
            const auto v01 = _mm_mul_pd(row01, e0);
            const auto v10 = _mm_mul_pd(row10, e1);
            const auto v11 = _mm_mul_pd(row11, e1);
            const auto v20 = _mm_mul_pd(row20, e2);
            const auto v21 = _mm_mul_pd(row21, e2);
            const auto v30 = _mm_mul_pd(row30, e3);
            const auto v31 = _mm_mul_pd(row31, e3);

            const auto a00 = _mm_add_pd(v00, v10);
            const auto a01 = _mm_add_pd(v01, v11);
            const auto a10 = _mm_add_pd(v20, v30);
            const auto a11 = _mm_add_pd(v21, v31);
            _mm_store_pd(&result.m[i * 4 + 0], _mm_add_pd(a00, a10));
            _mm_store_pd(&result.m[i * 4 + 2], _mm_add_pd(a01, a11));
        }
        return result;
    }

    template <>
    inline auto& operator*=(Matrix<double, 4, 4>& matrix1,
                           const Matrix<double, 4, 4>& matrix2) noexcept
    {
        const auto row00 = _mm_load_pd(&matrix1.m[0]);
        const auto row01 = _mm_load_pd(&matrix1.m[2]);
        const auto row10 = _mm_load_pd(&matrix1.m[4]);
        const auto row11 = _mm_load_pd(&matrix1.m[6]);
        const auto row20 = _mm_load_pd(&matrix1.m[8]);
        const auto row21 = _mm_load_pd(&matrix1.m[10]);
        const auto row30 = _mm_load_pd(&matrix1.m[12]);
        const auto row31 = _mm_load_pd(&matrix1.m[14]);

        for (std::size_t i = 0; i < 4; ++i)
        {
            const auto e0 = _mm_set1_pd(matrix2.m[i * 4 + 0]);
            const auto e1 = _mm_set1_pd(matrix2.m[i * 4 + 1]);
            const auto e2 = _mm_set1_pd(matrix2.m[i * 4 + 2]);
            const auto e3 = _mm_set1_pd(matrix2.m[i * 4 + 3]);

            const auto v00 = _mm_mul_pd(row00, e0);
            const auto v01 = _mm_mul_pd(row01, e0);
            const auto v10 = _mm_mul_pd(row10, e1);
            const auto v11 = _mm_mul_pd(row11, e1);
            const auto v20 = _mm_mul_pd(row20, e2);
            const auto v21 = _mm_mul_pd(row21, e2);
            const auto v30 = _mm_mul_pd(row30, e3);
            const auto v31 = _mm_mul_pd(row31, e3);

            const auto a00 = _mm_add_pd(v00, v10);
            const auto a01 = _mm_add_pd(v01, v11);
            const auto a10 = _mm_add_pd(v20, v30);
            const auto a11 = _mm_add_pd(v21, v31);
            _mm_store_pd(&matrix1.m[i * 4 + 0], _mm_add_pd(a00, a10));
            _mm_store_pd(&matrix1.m[i * 4 + 2], _mm_add_pd(a01, a11));
        }
        return matrix1;
    }

    template <>
    [[nodiscard]] inline auto transposed(const Matrix<double, 4, 4>& matrix) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto tmp00 = _mm_load_pd(&matrix.m[0]);
        const auto tmp01 = _mm_load_pd(&matrix.m[2]);
        const auto tmp10 = _mm_load_pd(&matrix.m[4]);
        const auto tmp11 = _mm_load_pd(&matrix.m[6]);
        const auto tmp20 = _mm_load_pd(&matrix.m[8]);
        const auto tmp21 = _mm_load_pd(&matrix.m[10]);
        const auto tmp30 = _mm_load_pd(&matrix.m[12]);
        const auto tmp31 = _mm_load_pd(&matrix.m[14]);

        _mm_store_pd(&result.m[0], _mm_shuffle_pd(tmp00, tmp10, _MM_SHUFFLE2(0, 0)));
        _mm_store_pd(&result.m[2], _mm_shuffle_pd(tmp20, tmp30, _MM_SHUFFLE2(0, 0)));
        _mm_store_pd(&result.m[4], _mm_shuffle_pd(tmp00, tmp10, _MM_SHUFFLE2(1, 1)));
        _mm_store_pd(&result.m[6], _mm_shuffle_pd(tmp20, tmp30, _MM_SHUFFLE2(1, 1)));
        _mm_store_pd(&result.m[8], _mm_shuffle_pd(tmp01, tmp11, _MM_SHUFFLE2(0, 0)));
        _mm_store_pd(&result.m[10], _mm_shuffle_pd(tmp21, tmp31, _MM_SHUFFLE2(0, 0)));
        _mm_store_pd(&result.m[12], _mm_shuffle_pd(tmp01, tmp11, _MM_SHUFFLE2(1, 1)));
        _mm_store_pd(&result.m[14], _mm_shuffle_pd(tmp21, tmp31, _MM_SHUFFLE2(1, 1)));
        return result;
    }

    template <>
    inline void transpose(Matrix<double, 4, 4>& matrix) noexcept
    {
        const auto tmp00 = _mm_load_pd(&matrix.m[0]);
        const auto tmp01 = _mm_load_pd(&matrix.m[2]);
        const auto tmp10 = _mm_load_pd(&matrix.m[4]);
        const auto tmp11 = _mm_load_pd(&matrix.m[6]);
        const auto tmp20 = _mm_load_pd(&matrix.m[8]);
        const auto tmp21 = _mm_load_pd(&matrix.m[10]);
        const auto tmp30 = _mm_load_pd(&matrix.m[12]);
        const auto tmp31 = _mm_load_pd(&matrix.m[14]);

        _mm_store_pd(&matrix.m[0], _mm_shuffle_pd(tmp00, tmp10, _MM_SHUFFLE2(0, 0)));
        _mm_store_pd(&matrix.m[2], _mm_shuffle_pd(tmp20, tmp30, _MM_SHUFFLE2(0, 0)));
        _mm_store_pd(&matrix.m[4], _mm_shuffle_pd(tmp00, tmp10, _MM_SHUFFLE2(1, 1)));
        _mm_store_pd(&matrix.m[6], _mm_shuffle_pd(tmp20, tmp30, _MM_SHUFFLE2(1, 1)));
        _mm_store_pd(&matrix.m[8], _mm_shuffle_pd(tmp01, tmp11, _MM_SHUFFLE2(0, 0)));
        _mm_store_pd(&matrix.m[10], _mm_shuffle_pd(tmp21, tmp31, _MM_SHUFFLE2(0, 0)));
        _mm_store_pd(&matrix.m[12], _mm_shuffle_pd(tmp01, tmp11, _MM_SHUFFLE2(1, 1)));
        _mm_store_pd(&matrix.m[14], _mm_shuffle_pd(tmp21, tmp31, _MM_SHUFFLE2(1, 1)));
    }
}

#endif

#endif // OMATH_MATRIX_SSE2
