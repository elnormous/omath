//
// elnormous/omath
//

#ifndef OMATH_VECTOR_SSE2
#define OMATH_VECTOR_SSE2

#include "Simd.hpp"
#include "Vector.hpp"

#ifdef OMATH_SIMD_SSE2
#  include <xmmintrin.h>

namespace omath
{
#  ifndef OMATH_SIMD_AVX
    template <>
    inline auto operator-(const Vector<double, 4>& vector) noexcept
    {
        Vector<double, 4> result;
        const auto z = _mm_setzero_pd();
        _mm_store_pd(&result.v[0], _mm_sub_pd(z, _mm_load_pd(&vector.v[0])));
        _mm_store_pd(&result.v[2], _mm_sub_pd(z, _mm_load_pd(&vector.v[2])));
        return result;
    }

    template <>
    inline void negate(Vector<double, 4>& vector) noexcept
    {
        const auto z = _mm_setzero_pd();
        _mm_store_pd(&vector.v[0], _mm_sub_pd(z, _mm_load_pd(&vector.v[0])));
        _mm_store_pd(&vector.v[2], _mm_sub_pd(z, _mm_load_pd(&vector.v[2])));
    }

    template <>
    [[nodiscard]] inline auto operator+(const Vector<double, 4>& vector1,
                                        const Vector<double, 4>& vector2) noexcept
    {
        Vector<double, 4> result;
        _mm_store_pd(&result.v[0], _mm_add_pd(_mm_load_pd(&vector1.v[0]), _mm_load_pd(&vector2.v[0])));
        _mm_store_pd(&result.v[2], _mm_add_pd(_mm_load_pd(&vector1.v[2]), _mm_load_pd(&vector2.v[2])));
        return result;
    }

    template <>
    inline auto& operator+=(Vector<double, 4>& vector1,
                            const Vector<double, 4>& vector2) noexcept
    {
        _mm_store_pd(&vector1.v[0], _mm_add_pd(_mm_load_pd(&vector1.v[0]), _mm_load_pd(&vector2.v[0])));
        _mm_store_pd(&vector1.v[2], _mm_add_pd(_mm_load_pd(&vector1.v[2]), _mm_load_pd(&vector2.v[2])));
        return vector1;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Vector<double, 4>& vector1,
                                        const Vector<double, 4>& vector2) noexcept
    {
        Vector<double, 4> result;
        _mm_store_pd(&result.v[0], _mm_sub_pd(_mm_load_pd(&vector1.v[0]), _mm_load_pd(&vector2.v[0])));
        _mm_store_pd(&result.v[2], _mm_sub_pd(_mm_load_pd(&vector1.v[2]), _mm_load_pd(&vector2.v[2])));
        return result;
    }

    template <>
    inline auto& operator-=(Vector<double, 4>& vector1,
                            const Vector<double, 4>& vector2) noexcept
    {
        _mm_store_pd(&vector1.v[0], _mm_sub_pd(_mm_load_pd(&vector1.v[0]), _mm_load_pd(&vector2.v[0])));
        _mm_store_pd(&vector1.v[2], _mm_sub_pd(_mm_load_pd(&vector1.v[2]), _mm_load_pd(&vector2.v[2])));
        return vector1;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Vector<double, 4>& vector,
                                        const double scalar) noexcept
    {
        Vector<double, 4> result;
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&result.v[0], _mm_mul_pd(_mm_load_pd(&vector.v[0]), s));
        _mm_store_pd(&result.v[2], _mm_mul_pd(_mm_load_pd(&vector.v[2]), s));
        return result;
    }

    template <>
    inline auto& operator*=(Vector<double, 4>& vector,
                            const double scalar) noexcept
    {
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&vector.v[0], _mm_mul_pd(_mm_load_pd(&vector.v[0]), s));
        _mm_store_pd(&vector.v[2], _mm_mul_pd(_mm_load_pd(&vector.v[2]), s));
        return vector;
    }

    template <>
    [[nodiscard]] inline auto operator/(const Vector<double, 4>& vector,
                                        const double scalar) noexcept
    {
        Vector<double, 4> result;
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&result.v[0], _mm_div_pd(_mm_load_pd(&vector.v[0]), s));
        _mm_store_pd(&result.v[2], _mm_div_pd(_mm_load_pd(&vector.v[2]), s));
        return result;
    }

    template <>
    inline auto& operator/=(Vector<double, 4>& vector,
                            const double scalar) noexcept
    {
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&vector.v[0], _mm_div_pd(_mm_load_pd(&vector.v[0]), s));
        _mm_store_pd(&vector.v[2], _mm_div_pd(_mm_load_pd(&vector.v[2]), s));
        return vector;
    }
#  endif
}

#endif

#endif // OMATH_VECTOR_SSE2
