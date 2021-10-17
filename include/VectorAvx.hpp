//
// elnormous/omath
//

#ifndef OMATH_VECTOR_AVX
#define OMATH_VECTOR_AVX

#include "Simd.hpp"
#include "Vector.hpp"

#ifdef OMATH_SIMD_AVX
#  include <immintrin.h>

namespace omath
{
    template <>
    inline auto operator-(const Vector<double, 4>& vector) noexcept
    {
        Vector<double, 4> result;
        const auto z = _mm256_setzero_pd();
        _mm256_store_pd(result.v.data(), _mm256_sub_pd(z, _mm256_load_pd(vector.v.data())));
        return result;
    }

    template <>
    inline void negate(Vector<double, 4>& vector) noexcept
    {
        const auto z = _mm256_setzero_pd();
        _mm256_store_pd(vector.v.data(), _mm256_sub_pd(z, _mm256_load_pd(vector.v.data())));
    }

    template <>
    [[nodiscard]] inline auto operator+(const Vector<double, 4>& vector1,
                                        const Vector<double, 4>& vector2) noexcept
    {
        Vector<double, 4> result;
        _mm256_store_pd(result.v.data(), _mm256_add_pd(_mm256_load_pd(vector1.v.data()), _mm256_load_pd(vector2.v.data())));
        return result;
    }

    template <>
    inline auto& operator+=(Vector<double, 4>& vector1,
                            const Vector<double, 4>& vector2) noexcept
    {
        _mm256_store_pd(vector1.v.data(), _mm256_add_pd(_mm256_load_pd(vector1.v.data()), _mm256_load_pd(vector2.v.data())));
        return vector1;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Vector<double, 4>& vector1,
                                        const Vector<double, 4>& vector2) noexcept
    {
        Vector<double, 4> result;
        _mm256_store_pd(result.v.data(), _mm256_sub_pd(_mm256_load_pd(vector1.v.data()), _mm256_load_pd(vector2.v.data())));
        return result;
    }

    template <>
    inline auto& operator-=(Vector<double, 4>& vector1,
                            const Vector<double, 4>& vector2) noexcept
    {
        _mm256_store_pd(vector1.v.data(), _mm256_sub_pd(_mm256_load_pd(vector1.v.data()), _mm256_load_pd(vector2.v.data())));
        return vector1;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Vector<double, 4>& vector,
                                        const double scalar) noexcept
    {
        Vector<double, 4> result;
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(result.v.data(), _mm256_mul_pd(_mm256_load_pd(vector.v.data()), s));
        return result;
    }

    template <>
    inline auto& operator*=(Vector<double, 4>& vector,
                            const double scalar) noexcept
    {
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(vector.v.data(), _mm256_mul_pd(_mm256_load_pd(vector.v.data()), s));
        return vector;
    }

    template <>
    [[nodiscard]] inline auto operator/(const Vector<double, 4>& vector,
                                        const double scalar) noexcept
    {
        Vector<double, 4> result;
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(result.v.data(), _mm256_div_pd(_mm256_load_pd(vector.v.data()), s));
        return result;
    }

    template <>
    inline auto& operator/=(Vector<double, 4>& vector,
                            const double scalar) noexcept
    {
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(vector.v.data(), _mm256_div_pd(_mm256_load_pd(vector.v.data()), s));
        return vector;
    }
}

#endif

#endif // OMATH_VECTOR_AVX
