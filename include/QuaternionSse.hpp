//
// elnormous/omath
//

#ifndef OMATH_QUATERNION_SSE
#define OMATH_QUATERNION_SSE

#include "Quaternion.hpp"
#include "Simd.hpp"

#ifdef OMATH_SIMD_SSE
#  include <xmmintrin.h>
#endif

#ifdef OMATH_SIMD_SSE2
#  include <emmintrin.h>
#endif

namespace omath
{
#ifdef OMATH_SIMD_SSE
    template <>
    inline auto operator-(const Quaternion<float>& quat) noexcept
    {
        Quaternion<float> result;
        const auto z = _mm_setzero_ps();
        _mm_store_ps(result.v.data(), _mm_sub_ps(z, _mm_load_ps(quat.v.data())));
        return result;
    }

    template <>
    inline void negate(Quaternion<float>& quat) noexcept
    {
        const auto z = _mm_setzero_ps();
        _mm_store_ps(quat.v.data(), _mm_sub_ps(z, _mm_load_ps(quat.v.data())));
    }

    template <>
    [[nodiscard]] inline auto operator+(const Quaternion<float>& quat1,
                                        const Quaternion<float>& quat2) noexcept
    {
        Quaternion<float> result;
        _mm_store_ps(result.v.data(), _mm_add_ps(_mm_load_ps(quat1.v.data()), _mm_load_ps(quat2.v.data())));
        return result;
    }

    template <>
    inline auto& operator+=(Quaternion<float>& quat1,
                            const Quaternion<float>& quat2) noexcept
    {
        _mm_store_ps(quat1.v.data(), _mm_add_ps(_mm_load_ps(quat1.v.data()), _mm_load_ps(quat2.v.data())));
        return quat1;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Quaternion<float>& quat1,
                                        const Quaternion<float>& quat2) noexcept
    {
        Quaternion<float> result;
        _mm_store_ps(result.v.data(), _mm_sub_ps(_mm_load_ps(quat1.v.data()), _mm_load_ps(quat2.v.data())));
        return result;
    }

    template <>
    inline auto& operator-=(Quaternion<float>& quat1,
                            const Quaternion<float>& quat2) noexcept
    {
        _mm_store_ps(quat1.v.data(), _mm_sub_ps(_mm_load_ps(quat1.v.data()), _mm_load_ps(quat2.v.data())));
        return quat1;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Quaternion<float>& quat,
                                        const float scalar) noexcept
    {
        Quaternion<float> result;
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(result.v.data(), _mm_mul_ps(_mm_load_ps(quat.v.data()), s));
        return result;
    }

    template <>
    inline auto& operator*=(Quaternion<float>& quat,
                            const float scalar) noexcept
    {
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(quat.v.data(), _mm_mul_ps(_mm_load_ps(quat.v.data()), s));
        return quat;
    }

    template <>
    [[nodiscard]] inline auto operator/(const Quaternion<float>& quat,
                                        const float scalar) noexcept
    {
        Quaternion<float> result;
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(result.v.data(), _mm_div_ps(_mm_load_ps(quat.v.data()), s));
        return result;
    }

    template <>
    inline auto& operator/=(Quaternion<float>& quat,
                            const float scalar) noexcept
    {
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(quat.v.data(), _mm_div_ps(_mm_load_ps(quat.v.data()), s));
        return quat;
    }
#endif // OMATH_SIMD_SSE

#ifdef OMATH_SIMD_SSE2
#  ifndef OMATH_SIMD_AVX
    template <>
    inline auto operator-(const Quaternion<double>& quat) noexcept
    {
        Quaternion<double> result;
        const auto z = _mm_setzero_pd();
        _mm_store_pd(&result.v[0], _mm_sub_pd(z, _mm_load_pd(&quat.v[0])));
        _mm_store_pd(&result.v[2], _mm_sub_pd(z, _mm_load_pd(&quat.v[2])));
        return result;
    }

    template <>
    inline void negate(Quaternion<double>& quat) noexcept
    {
        const auto z = _mm_setzero_pd();
        _mm_store_pd(&quat.v[0], _mm_sub_pd(z, _mm_load_pd(&quat.v[0])));
        _mm_store_pd(&quat.v[2], _mm_sub_pd(z, _mm_load_pd(&quat.v[2])));
    }

    template <>
    [[nodiscard]] inline auto operator+(const Quaternion<double>& quat1,
                                        const Quaternion<double>& quat2) noexcept
    {
        Quaternion<double> result;
        _mm_store_pd(&result.v[0], _mm_add_pd(_mm_load_pd(&quat1.v[0]), _mm_load_pd(&quat2.v[0])));
        _mm_store_pd(&result.v[2], _mm_add_pd(_mm_load_pd(&quat1.v[2]), _mm_load_pd(&quat2.v[2])));
        return result;
    }

    template <>
    inline auto& operator+=(Quaternion<double>& quat1,
                            const Quaternion<double>& quat2) noexcept
    {
        _mm_store_pd(&quat1.v[0], _mm_add_pd(_mm_load_pd(&quat1.v[0]), _mm_load_pd(&quat2.v[0])));
        _mm_store_pd(&quat1.v[2], _mm_add_pd(_mm_load_pd(&quat1.v[2]), _mm_load_pd(&quat2.v[2])));
        return quat1;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Quaternion<double>& quat1,
                                        const Quaternion<double>& quat2) noexcept
    {
        Quaternion<double> result;
        _mm_store_pd(&result.v[0], _mm_sub_pd(_mm_load_pd(&quat1.v[0]), _mm_load_pd(&quat2.v[0])));
        _mm_store_pd(&result.v[2], _mm_sub_pd(_mm_load_pd(&quat1.v[2]), _mm_load_pd(&quat2.v[2])));
        return result;
    }

    template <>
    inline auto& operator-=(Quaternion<double>& quat1,
                            const Quaternion<double>& quat2) noexcept
    {
        _mm_store_pd(&quat1.v[0], _mm_sub_pd(_mm_load_pd(&quat1.v[0]), _mm_load_pd(&quat2.v[0])));
        _mm_store_pd(&quat1.v[2], _mm_sub_pd(_mm_load_pd(&quat1.v[2]), _mm_load_pd(&quat2.v[2])));
        return quat1;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Quaternion<double>& quat,
                                        const double scalar) noexcept
    {
        Quaternion<double> result;
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&result.v[0], _mm_mul_pd(_mm_load_pd(&quat.v[0]), s));
        _mm_store_pd(&result.v[2], _mm_mul_pd(_mm_load_pd(&quat.v[2]), s));
        return result;
    }

    template <>
    inline auto& operator*=(Quaternion<double>& quat,
                            const double scalar) noexcept
    {
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&quat.v[0], _mm_mul_pd(_mm_load_pd(&quat.v[0]), s));
        _mm_store_pd(&quat.v[2], _mm_mul_pd(_mm_load_pd(&quat.v[2]), s));
        return quat;
    }

    template <>
    [[nodiscard]] inline auto operator/(const Quaternion<double>& quat,
                                        const double scalar) noexcept
    {
        Quaternion<double> result;
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&result.v[0], _mm_div_pd(_mm_load_pd(&quat.v[0]), s));
        _mm_store_pd(&result.v[2], _mm_div_pd(_mm_load_pd(&quat.v[2]), s));
        return result;
    }

    template <>
    inline auto& operator/=(Quaternion<double>& quat,
                            const double scalar) noexcept
    {
        const auto s = _mm_set1_pd(scalar);
        _mm_store_pd(&quat.v[0], _mm_div_pd(_mm_load_pd(&quat.v[0]), s));
        _mm_store_pd(&quat.v[2], _mm_div_pd(_mm_load_pd(&quat.v[2]), s));
        return quat;
    }
#  endif
#endif // OMATH_SIMD_SSE2
}

#endif // OMATH_QUATERNION_SSE
