//
// elnormous/omath
//

#ifndef OMATH_QUATERNION_SSE
#define OMATH_QUATERNION_SSE

#include "Quaternion.hpp"
#include "Simd.hpp"

#ifdef OMATH_SIMD_SSE
#  include <xmmintrin.h>

namespace omath
{
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
}

#endif

#endif // OMATH_QUATERNION_SSE
