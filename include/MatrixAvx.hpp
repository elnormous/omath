//
// elnormous/omath
//

#ifndef OMATH_MATRIX_AVX_HPP
#define OMATH_MATRIX_AVX_HPP

#include "Matrix.hpp"

#ifndef OMATH_DISABLE_SIMD

#ifdef __AVX__
#  include <immintrin.h>

namespace omath
{
    template <>
    [[nodiscard]] inline auto operator-(const Matrix<float, 4, 4>& matrix) noexcept
    {
        Matrix<float, 4, 4> result;
        const auto z = _mm256_setzero_ps();
        _mm256_store_ps(&result.m.v[0], _mm256_sub_ps(z, _mm256_load_ps(&matrix.m.v[0])));
        _mm256_store_ps(&result.m.v[8], _mm256_sub_ps(z, _mm256_load_ps(&matrix.m.v[8])));
        return result;
    }

    template <>
    inline void negate(Matrix<float, 4, 4>& matrix) noexcept
    {
        const auto z = _mm256_setzero_ps();
        _mm256_store_ps(&matrix.m.v[0], _mm256_sub_ps(z, _mm256_load_ps(&matrix.m.v[0])));
        _mm256_store_ps(&matrix.m.v[8], _mm256_sub_ps(z, _mm256_load_ps(&matrix.m.v[8])));
    }

    template <>
    [[nodiscard]] inline auto operator+(const Matrix<float, 4, 4>& matrix1,
                                        const Matrix<float, 4, 4>& matrix2) noexcept
    {
        Matrix<float, 4, 4> result;
        _mm256_store_ps(&result.m.v[0], _mm256_add_ps(_mm256_load_ps(&matrix1.m.v[0]),
                                                      _mm256_load_ps(&matrix2.m.v[0])));
        _mm256_store_ps(&result.m.v[8], _mm256_add_ps(_mm256_load_ps(&matrix1.m.v[8]),
                                                      _mm256_load_ps(&matrix2.m.v[8])));
        return result;
    }

    template <>
    inline auto& operator+=(Matrix<float, 4, 4>& matrix1,
                            const Matrix<float, 4, 4>& matrix2) noexcept
    {
        _mm256_store_ps(&matrix1.m.v[0], _mm256_add_ps(_mm256_load_ps(&matrix1.m.v[0]),
                                                       _mm256_load_ps(&matrix2.m.v[0])));
        _mm256_store_ps(&matrix1.m.v[8], _mm256_add_ps(_mm256_load_ps(&matrix1.m.v[8]),
                                                       _mm256_load_ps(&matrix2.m.v[8])));
        return matrix1;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Matrix<float, 4, 4>& matrix1,
                                        const Matrix<float, 4, 4>& matrix2) noexcept
    {
        Matrix<float, 4, 4> result;
        _mm256_store_ps(&result.m.v[0], _mm256_sub_ps(_mm256_load_ps(&matrix1.m.v[0]),
                                                      _mm256_load_ps(&matrix2.m.v[0])));
        _mm256_store_ps(&result.m.v[8], _mm256_sub_ps(_mm256_load_ps(&matrix1.m.v[8]),
                                                      _mm256_load_ps(&matrix2.m.v[8])));
        return result;
    }

    template <>
    inline auto& operator-=(Matrix<float, 4, 4>& matrix1,
                            const Matrix<float, 4, 4>& matrix2) noexcept
    {
        _mm256_store_ps(&matrix1.m.v[0], _mm256_sub_ps(_mm256_load_ps(&matrix1.m.v[0]),
                                                       _mm256_load_ps(&matrix2.m.v[0])));
        _mm256_store_ps(&matrix1.m.v[8], _mm256_sub_ps(_mm256_load_ps(&matrix1.m.v[8]),
                                                       _mm256_load_ps(&matrix2.m.v[8])));
        return matrix1;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Matrix<float, 4, 4>& matrix,
                                        const float scalar) noexcept
    {
        Matrix<float, 4, 4> result;
        const auto s = _mm256_set1_ps(scalar);
        _mm256_store_ps(&result.m.v[0], _mm256_mul_ps(_mm256_load_ps(&matrix.m.v[0]), s));
        _mm256_store_ps(&result.m.v[8], _mm256_mul_ps(_mm256_load_ps(&matrix.m.v[8]), s));
        return result;
    }

    template <>
    inline auto& operator*=(Matrix<float, 4, 4>& matrix,
                            const float scalar) noexcept
    {
        const auto s = _mm256_set1_ps(scalar);
        _mm256_store_ps(&matrix.m.v[0], _mm256_mul_ps(_mm256_load_ps(&matrix.m.v[0]), s));
        _mm256_store_ps(&matrix.m.v[8], _mm256_mul_ps(_mm256_load_ps(&matrix.m.v[8]), s));
        return matrix;
    }

    template <>
    [[nodiscard]] inline auto operator/(const Matrix<float, 4, 4>& matrix,
                                        float scalar) noexcept
    {
        Matrix<float, 4, 4> result;
        const auto s = _mm256_set1_ps(scalar);
        _mm256_store_ps(&result.m.v[0], _mm256_div_ps(_mm256_load_ps(&matrix.m.v[0]), s));
        _mm256_store_ps(&result.m.v[8], _mm256_div_ps(_mm256_load_ps(&matrix.m.v[8]), s));
        return result;
    }

    template <>
    inline auto& operator/=(Matrix<float, 4, 4>& matrix,
                            const float scalar) noexcept
    {
        const auto s = _mm256_set1_ps(scalar);
        _mm256_store_ps(&matrix.m.v[0], _mm256_div_ps(_mm256_load_ps(&matrix.m.v[0]), s));
        _mm256_store_ps(&matrix.m.v[8], _mm256_div_ps(_mm256_load_ps(&matrix.m.v[8]), s));
        return matrix;
    }

    template <>
    [[nodiscard]] inline auto operator*(const float scalar,
                                        const Matrix<float, 4, 4>& matrix) noexcept
    {
        Matrix<float, 4, 4> result;
        const auto s = _mm256_set1_ps(scalar);
        _mm256_store_ps(&result.m.v[0], _mm256_mul_ps(_mm256_load_ps(&matrix.m.v[0]), s));
        _mm256_store_ps(&result.m.v[8], _mm256_mul_ps(_mm256_load_ps(&matrix.m.v[8]), s));
        return result;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Matrix<double, 4, 4>& matrix) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto z = _mm256_setzero_pd();
        _mm256_store_pd(&result.m.v[0], _mm256_sub_pd(z, _mm256_load_pd(&matrix.m.v[0])));
        _mm256_store_pd(&result.m.v[4], _mm256_sub_pd(z, _mm256_load_pd(&matrix.m.v[4])));
        _mm256_store_pd(&result.m.v[8], _mm256_sub_pd(z, _mm256_load_pd(&matrix.m.v[8])));
        _mm256_store_pd(&result.m.v[12], _mm256_sub_pd(z, _mm256_load_pd(&matrix.m.v[12])));
        return result;
    }

    template <>
    inline void negate(Matrix<double, 4, 4>& matrix) noexcept
    {
        const auto z = _mm256_setzero_pd();
        _mm256_store_pd(&matrix.m.v[0], _mm256_sub_pd(z, _mm256_load_pd(&matrix.m.v[0])));
        _mm256_store_pd(&matrix.m.v[4], _mm256_sub_pd(z, _mm256_load_pd(&matrix.m.v[4])));
        _mm256_store_pd(&matrix.m.v[8], _mm256_sub_pd(z, _mm256_load_pd(&matrix.m.v[8])));
        _mm256_store_pd(&matrix.m.v[12], _mm256_sub_pd(z, _mm256_load_pd(&matrix.m.v[12])));
    }

    template <>
    [[nodiscard]] inline auto operator+(const Matrix<double, 4, 4>& matrix1,
                                        const Matrix<double, 4, 4>& matrix2) noexcept
    {
        Matrix<double, 4, 4> result;
        _mm256_store_pd(&result.m.v[0], _mm256_add_pd(_mm256_load_pd(&matrix1.m.v[0]),
                                                      _mm256_load_pd(&matrix2.m.v[0])));
        _mm256_store_pd(&result.m.v[4], _mm256_add_pd(_mm256_load_pd(&matrix1.m.v[4]),
                                                      _mm256_load_pd(&matrix2.m.v[4])));
        _mm256_store_pd(&result.m.v[8], _mm256_add_pd(_mm256_load_pd(&matrix1.m.v[8]),
                                                      _mm256_load_pd(&matrix2.m.v[8])));
        _mm256_store_pd(&result.m.v[12], _mm256_add_pd(_mm256_load_pd(&matrix1.m.v[12]),
                                                       _mm256_load_pd(&matrix2.m.v[12])));
        return result;
    }

    template <>
    inline auto& operator+=(Matrix<double, 4, 4>& matrix1,
                            const Matrix<double, 4, 4>& matrix2) noexcept
    {
        _mm256_store_pd(&matrix1.m.v[0], _mm256_add_pd(_mm256_load_pd(&matrix1.m.v[0]),
                                                       _mm256_load_pd(&matrix2.m.v[0])));
        _mm256_store_pd(&matrix1.m.v[4], _mm256_add_pd(_mm256_load_pd(&matrix1.m.v[4]),
                                                       _mm256_load_pd(&matrix2.m.v[4])));
        _mm256_store_pd(&matrix1.m.v[8], _mm256_add_pd(_mm256_load_pd(&matrix1.m.v[8]),
                                                       _mm256_load_pd(&matrix2.m.v[8])));
        _mm256_store_pd(&matrix1.m.v[12], _mm256_add_pd(_mm256_load_pd(&matrix1.m.v[12]),
                                                        _mm256_load_pd(&matrix2.m.v[12])));
        return matrix1;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Matrix<double, 4, 4>& matrix1,
                                        const Matrix<double, 4, 4>& matrix2) noexcept
    {
        Matrix<double, 4, 4> result;
        _mm256_store_pd(&result.m.v[0], _mm256_sub_pd(_mm256_load_pd(&matrix1.m.v[0]),
                                                      _mm256_load_pd(&matrix2.m.v[0])));
        _mm256_store_pd(&result.m.v[4], _mm256_sub_pd(_mm256_load_pd(&matrix1.m.v[4]),
                                                      _mm256_load_pd(&matrix2.m.v[4])));
        _mm256_store_pd(&result.m.v[8], _mm256_sub_pd(_mm256_load_pd(&matrix1.m.v[8]),
                                                      _mm256_load_pd(&matrix2.m.v[8])));
        _mm256_store_pd(&result.m.v[12], _mm256_sub_pd(_mm256_load_pd(&matrix1.m.v[12]),
                                                       _mm256_load_pd(&matrix2.m.v[12])));
        return result;
    }

    template <>
    inline auto& operator-=(Matrix<double, 4, 4>& matrix1,
                            const Matrix<double, 4, 4>& matrix2) noexcept
    {
        _mm256_store_pd(&matrix1.m.v[0], _mm256_sub_pd(_mm256_load_pd(&matrix1.m.v[0]),
                                                       _mm256_load_pd(&matrix2.m.v[0])));
        _mm256_store_pd(&matrix1.m.v[4], _mm256_sub_pd(_mm256_load_pd(&matrix1.m.v[4]),
                                                       _mm256_load_pd(&matrix2.m.v[4])));
        _mm256_store_pd(&matrix1.m.v[8], _mm256_sub_pd(_mm256_load_pd(&matrix1.m.v[8]),
                                                       _mm256_load_pd(&matrix2.m.v[8])));
        _mm256_store_pd(&matrix1.m.v[12], _mm256_sub_pd(_mm256_load_pd(&matrix1.m.v[12]),
                                                        _mm256_load_pd(&matrix2.m.v[12])));
        return matrix1;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Matrix<double, 4, 4>& matrix,
                                        const double scalar) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(&result.m.v[0], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[0]), s));
        _mm256_store_pd(&result.m.v[4], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[4]), s));
        _mm256_store_pd(&result.m.v[8], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[8]), s));
        _mm256_store_pd(&result.m.v[12], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[12]), s));
        return result;
    }

    template <>
    inline auto& operator*=(Matrix<double, 4, 4>& matrix,
                            const double scalar) noexcept
    {
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(&matrix.m.v[0], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[0]), s));
        _mm256_store_pd(&matrix.m.v[4], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[4]), s));
        _mm256_store_pd(&matrix.m.v[8], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[8]), s));
        _mm256_store_pd(&matrix.m.v[12], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[12]), s));
        return matrix;
    }

    template <>
    [[nodiscard]] inline auto operator/(const Matrix<double, 4, 4>& matrix,
                                        double scalar) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(&result.m.v[0], _mm256_div_pd(_mm256_load_pd(&matrix.m.v[0]), s));
        _mm256_store_pd(&result.m.v[4], _mm256_div_pd(_mm256_load_pd(&matrix.m.v[4]), s));
        _mm256_store_pd(&result.m.v[8], _mm256_div_pd(_mm256_load_pd(&matrix.m.v[8]), s));
        _mm256_store_pd(&result.m.v[12], _mm256_div_pd(_mm256_load_pd(&matrix.m.v[12]), s));
        return result;
    }

    template <>
    inline auto& operator/=(Matrix<double, 4, 4>& matrix,
                            const double scalar) noexcept
    {
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(&matrix.m.v[0], _mm256_div_pd(_mm256_load_pd(&matrix.m.v[0]), s));
        _mm256_store_pd(&matrix.m.v[4], _mm256_div_pd(_mm256_load_pd(&matrix.m.v[4]), s));
        _mm256_store_pd(&matrix.m.v[8], _mm256_div_pd(_mm256_load_pd(&matrix.m.v[8]), s));
        _mm256_store_pd(&matrix.m.v[12], _mm256_div_pd(_mm256_load_pd(&matrix.m.v[12]), s));
        return matrix;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Matrix<double, 4, 4>& matrix1,
                                        const Matrix<double, 4, 4>& matrix2) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto col0 = _mm256_load_pd(&matrix1.m.v[0]);
        const auto col1 = _mm256_load_pd(&matrix1.m.v[4]);
        const auto col2 = _mm256_load_pd(&matrix1.m.v[8]);
        const auto col3 = _mm256_load_pd(&matrix1.m.v[12]);

        for (std::size_t i = 0; i < 4; ++i)
        {
            const auto e0 = _mm256_set1_pd(matrix2.m.v[i * 4 + 0]);
            const auto e1 = _mm256_set1_pd(matrix2.m.v[i * 4 + 1]);
            const auto e2 = _mm256_set1_pd(matrix2.m.v[i * 4 + 2]);
            const auto e3 = _mm256_set1_pd(matrix2.m.v[i * 4 + 3]);

            const auto v0 = _mm256_mul_pd(col0, e0);
            const auto v1 = _mm256_mul_pd(col1, e1);
            const auto v2 = _mm256_mul_pd(col2, e2);
            const auto v3 = _mm256_mul_pd(col3, e3);

            const auto a0 = _mm256_add_pd(v0, v1);
            const auto a1 = _mm256_add_pd(v2, v3);
            _mm256_store_pd(&result.m.v[i * 4], _mm256_add_pd(a0, a1));
        }
        return result;
    }

    template <>
    inline auto& operator*=(Matrix<double, 4, 4>& matrix1,
                            const Matrix<double, 4, 4>& matrix2) noexcept
    {
        const auto col0 = _mm256_load_pd(&matrix1.m.v[0]);
        const auto col1 = _mm256_load_pd(&matrix1.m.v[4]);
        const auto col2 = _mm256_load_pd(&matrix1.m.v[8]);
        const auto col3 = _mm256_load_pd(&matrix1.m.v[12]);

        for (std::size_t i = 0; i < 4; ++i)
        {
            const auto e0 = _mm256_set1_pd(matrix2.m.v[i * 4 + 0]);
            const auto e1 = _mm256_set1_pd(matrix2.m.v[i * 4 + 1]);
            const auto e2 = _mm256_set1_pd(matrix2.m.v[i * 4 + 2]);
            const auto e3 = _mm256_set1_pd(matrix2.m.v[i * 4 + 3]);

            const auto v0 = _mm256_mul_pd(col0, e0);
            const auto v1 = _mm256_mul_pd(col1, e1);
            const auto v2 = _mm256_mul_pd(col2, e2);
            const auto v3 = _mm256_mul_pd(col3, e3);

            const auto a0 = _mm256_add_pd(v0, v1);
            const auto a1 = _mm256_add_pd(v2, v3);
            _mm256_store_pd(&matrix1.m.v[i * 4], _mm256_add_pd(a0, a1));
        }
        return matrix1;
    }

    template <>
    [[nodiscard]] inline auto operator*(const double scalar,
                                        const Matrix<double, 4, 4>& matrix) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(&result.m.v[0], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[0]), s));
        _mm256_store_pd(&result.m.v[4], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[4]), s));
        _mm256_store_pd(&result.m.v[8], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[8]), s));
        _mm256_store_pd(&result.m.v[12], _mm256_mul_pd(_mm256_load_pd(&matrix.m.v[12]), s));
        return result;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Vector<double, 4>& vector,
                                        const Matrix<double, 4, 4>& matrix) noexcept
    {
        Vector<double, 4> result;

        const auto row0 = _mm256_set1_pd(vector.v[0]);
        const auto row1 = _mm256_set1_pd(vector.v[1]);
        const auto row2 = _mm256_set1_pd(vector.v[2]);
        const auto row3 = _mm256_set1_pd(vector.v[3]);

        const auto col0 = _mm256_load_pd(&matrix.m.v[0]);
        const auto col1 = _mm256_load_pd(&matrix.m.v[4]);
        const auto col2 = _mm256_load_pd(&matrix.m.v[8]);
        const auto col3 = _mm256_load_pd(&matrix.m.v[12]);

        const auto s = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(row0, col0),
                                                   _mm256_mul_pd(row1, col1)),
                                     _mm256_add_pd(_mm256_mul_pd(row2, col2),
                                                   _mm256_mul_pd(row3, col3)));
        _mm256_store_pd(result.v, s);

        return result;
    }

    template <>
    inline auto& operator*=(Vector<double, 4>& vector,
                            const Matrix<double, 4, 4>& matrix) noexcept
    {
        const auto row0 = _mm256_set1_pd(vector.v[0]);
        const auto row1 = _mm256_set1_pd(vector.v[1]);
        const auto row2 = _mm256_set1_pd(vector.v[2]);
        const auto row3 = _mm256_set1_pd(vector.v[3]);

        const auto col0 = _mm256_load_pd(&matrix.m.v[0]);
        const auto col1 = _mm256_load_pd(&matrix.m.v[4]);
        const auto col2 = _mm256_load_pd(&matrix.m.v[8]);
        const auto col3 = _mm256_load_pd(&matrix.m.v[12]);

        const auto s = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(row0, col0),
                                                   _mm256_mul_pd(row1, col1)),
                                     _mm256_add_pd(_mm256_mul_pd(row2, col2),
                                                   _mm256_mul_pd(row3, col3)));
        _mm256_store_pd(vector.v, s);

        return vector;
    }
}

#endif // __AVX__

#endif // OMATH_DISABLE_SIMD

#endif // OMATH_MATRIX_AVX_HPP
