//
// elnormous/omath
//

#ifndef OMATH_QUATERNION_SSE2
#define OMATH_QUATERNION_SSE2

#include "Quaternion.hpp"
#include "Simd.hpp"

#ifdef OMATH_SIMD_SSE2
#  include <xmmintrin.h>

namespace omath
{
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
}

#endif

#endif // OMATH_QUATERNION_SSE2
