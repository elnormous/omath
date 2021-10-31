//
// elnormous/omath
//

#ifndef OMATH_VECTOR_SSE
#define OMATH_VECTOR_SSE

#include "Vector.hpp"

#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP >= 1
#  include <xmmintrin.h>
#endif

#if defined(__SSE2__) || defined(_M_X64) || _M_IX86_FP >= 2
#  include <emmintrin.h>
#endif

#ifdef __SSE3__
#  include <pmmintrin.h>
#endif // __SSE3__

namespace omath
{
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP >= 1
    template <>
    inline auto operator-(const Vector<float, 4>& vector) noexcept
    {
        Vector<float, 4> result;
        const auto z = _mm_setzero_ps();
        _mm_store_ps(result.v.data(), _mm_sub_ps(z, _mm_load_ps(vector.v.data())));
        return result;
    }

    template <>
    inline void negate(Vector<float, 4>& vector) noexcept
    {
        const auto z = _mm_setzero_ps();
        _mm_store_ps(vector.v.data(), _mm_sub_ps(z, _mm_load_ps(vector.v.data())));
    }

    template <>
    [[nodiscard]] inline auto operator+(const Vector<float, 4>& vector1,
                                        const Vector<float, 4>& vector2) noexcept
    {
        Vector<float, 4> result;
        _mm_store_ps(result.v.data(), _mm_add_ps(_mm_load_ps(vector1.v.data()), _mm_load_ps(vector2.v.data())));
        return result;
    }

    template <>
    inline auto& operator+=(Vector<float, 4>& vector1,
                            const Vector<float, 4>& vector2) noexcept
    {
        _mm_store_ps(vector1.v.data(), _mm_add_ps(_mm_load_ps(vector1.v.data()), _mm_load_ps(vector2.v.data())));
        return vector1;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Vector<float, 4>& vector1,
                                        const Vector<float, 4>& vector2) noexcept
    {
        Vector<float, 4> result;
        _mm_store_ps(result.v.data(), _mm_sub_ps(_mm_load_ps(vector1.v.data()), _mm_load_ps(vector2.v.data())));
        return result;
    }

    template <>
    inline auto& operator-=(Vector<float, 4>& vector1,
                            const Vector<float, 4>& vector2) noexcept
    {
        _mm_store_ps(vector1.v.data(), _mm_sub_ps(_mm_load_ps(vector1.v.data()), _mm_load_ps(vector2.v.data())));
        return vector1;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Vector<float, 4>& vector,
                                        const float scalar) noexcept
    {
        Vector<float, 4> result;
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(result.v.data(), _mm_mul_ps(_mm_load_ps(vector.v.data()), s));
        return result;
    }

    template <>
    inline auto& operator*=(Vector<float, 4>& vector,
                            const float scalar) noexcept
    {
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(vector.v.data(), _mm_mul_ps(_mm_load_ps(vector.v.data()), s));
        return vector;
    }

    template <>
    [[nodiscard]] inline auto operator/(const Vector<float, 4>& vector,
                                        const float scalar) noexcept
    {
        Vector<float, 4> result;
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(result.v.data(), _mm_div_ps(_mm_load_ps(vector.v.data()), s));
        return result;
    }

    template <>
    inline auto& operator/=(Vector<float, 4>& vector,
                            const float scalar) noexcept
    {
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(vector.v.data(), _mm_div_ps(_mm_load_ps(vector.v.data()), s));
        return vector;
    }

#  ifndef __SSE3__
    template <>
    [[nodiscard]] inline auto length(const Vector<float, 4>& vector) noexcept
    {
        const auto v = _mm_load_ps(vector.v.data());
        const auto t1 = _mm_mul_ps(v, v);
        const auto t2 = _mm_add_ps(t1, _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2, 1, 0, 3)));
        const auto t3 = _mm_add_ps(t2, _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1, 0, 3, 2)));
        const auto s = _mm_sqrt_ps(t3);
        return _mm_cvtss_f32(s);
    }

    template <>
    [[nodiscard]] inline auto lengthSquared(const Vector<float, 4>& vector) noexcept
    {
        const auto v = _mm_load_ps(vector.v.data());
        const auto t1 = _mm_mul_ps(v, v);
        const auto t2 = _mm_add_ps(t1, _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2, 1, 0, 3)));
        const auto t3 = _mm_add_ps(t2, _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtss_f32(t3);
    }

    template <>
    [[nodiscard]] inline auto dot(const Vector<float, 4>& vector1,
                                  const Vector<float, 4>& vector2) noexcept
    {
        const auto t1 = _mm_mul_ps(_mm_load_ps(vector1.v.data()), _mm_load_ps(vector2.v.data()));
        const auto t2 = _mm_add_ps(t1, _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2, 1, 0, 3)));
        const auto t3 = _mm_add_ps(t2, _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtss_f32(t3);
    }
#  endif // __SSE3__
#endif // SSE

#if defined(__SSE2__) || defined(_M_X64) || _M_IX86_FP >= 2
#  ifndef __AVX__
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
#  endif // __AVX__
#endif // SSE2

#ifdef __SSE3__
    template <>
    [[nodiscard]] inline auto length(const Vector<float, 4>& vector) noexcept
    {
        const auto v = _mm_load_ps(vector.v.data());
        const auto t1 = _mm_mul_ps(v, v);
        const auto t2 = _mm_hadd_ps(t1, t1);
        const auto t3 = _mm_hadd_ps(t2, t2);
        const auto s = _mm_sqrt_ps(t3);
        return _mm_cvtss_f32(s);
    }

    template <>
    [[nodiscard]] inline auto lengthSquared(const Vector<float, 4>& vector) noexcept
    {
        const auto v = _mm_load_ps(vector.v.data());
        const auto t1 = _mm_mul_ps(v, v);
        const auto t2 = _mm_hadd_ps(t1, t1);
        const auto t3 = _mm_hadd_ps(t2, t2);
        return _mm_cvtss_f32(t3);
    }

    template <>
    [[nodiscard]] inline auto dot(const Vector<float, 4>& vector1,
                                  const Vector<float, 4>& vector2) noexcept
    {
        const auto t1 = _mm_mul_ps(_mm_load_ps(vector1.v.data()), _mm_load_ps(vector2.v.data()));
        const auto t2 = _mm_hadd_ps(t1, t1);
        const auto t3 = _mm_hadd_ps(t2, t2);
        return _mm_cvtss_f32(t3);
    }
#endif // __SSE3__
}

#endif // OMATH_VECTOR_SSE
