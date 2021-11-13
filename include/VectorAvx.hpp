//
// elnormous/omath
//

#ifndef OMATH_VECTOR_AVX_HPP
#define OMATH_VECTOR_AVX_HPP

#include "Vector.hpp"

#ifndef OMATH_DISABLE_SIMD

#ifdef __AVX__
#  include <immintrin.h>

namespace omath
{
    template <>
    inline auto operator-(const Vector<double, 4>& vector) noexcept
    {
        Vector<double, 4> result;
        const auto z = _mm256_setzero_pd();
        _mm256_store_pd(result.v, _mm256_sub_pd(z, _mm256_load_pd(vector.v)));
        return result;
    }

    template <>
    inline void negate(Vector<double, 4>& vector) noexcept
    {
        const auto z = _mm256_setzero_pd();
        _mm256_store_pd(vector.v, _mm256_sub_pd(z, _mm256_load_pd(vector.v)));
    }

    template <>
    [[nodiscard]] inline auto operator+(const Vector<double, 4>& vector1,
                                        const Vector<double, 4>& vector2) noexcept
    {
        Vector<double, 4> result;
        _mm256_store_pd(result.v, _mm256_add_pd(_mm256_load_pd(vector1.v), _mm256_load_pd(vector2.v)));
        return result;
    }

    template <>
    inline auto& operator+=(Vector<double, 4>& vector1,
                            const Vector<double, 4>& vector2) noexcept
    {
        _mm256_store_pd(vector1.v, _mm256_add_pd(_mm256_load_pd(vector1.v), _mm256_load_pd(vector2.v)));
        return vector1;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Vector<double, 4>& vector1,
                                        const Vector<double, 4>& vector2) noexcept
    {
        Vector<double, 4> result;
        _mm256_store_pd(result.v, _mm256_sub_pd(_mm256_load_pd(vector1.v), _mm256_load_pd(vector2.v)));
        return result;
    }

    template <>
    inline auto& operator-=(Vector<double, 4>& vector1,
                            const Vector<double, 4>& vector2) noexcept
    {
        _mm256_store_pd(vector1.v, _mm256_sub_pd(_mm256_load_pd(vector1.v), _mm256_load_pd(vector2.v)));
        return vector1;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Vector<double, 4>& vector,
                                        const double scalar) noexcept
    {
        Vector<double, 4> result;
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(result.v, _mm256_mul_pd(_mm256_load_pd(vector.v), s));
        return result;
    }

    template <>
    inline auto& operator*=(Vector<double, 4>& vector,
                            const double scalar) noexcept
    {
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(vector.v, _mm256_mul_pd(_mm256_load_pd(vector.v), s));
        return vector;
    }

    template <>
    [[nodiscard]] inline auto operator/(const Vector<double, 4>& vector,
                                        const double scalar) noexcept
    {
        Vector<double, 4> result;
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(result.v, _mm256_div_pd(_mm256_load_pd(vector.v), s));
        return result;
    }

    template <>
    inline auto& operator/=(Vector<double, 4>& vector,
                            const double scalar) noexcept
    {
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(vector.v, _mm256_div_pd(_mm256_load_pd(vector.v), s));
        return vector;
    }
}

#endif // __AVX__

#endif // OMATH_DISABLE_SIMD

#endif // OMATH_VECTOR_AVX_HPP
