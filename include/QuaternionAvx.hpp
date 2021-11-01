//
// elnormous/omath
//

#ifndef OMATH_QUATERNION_AVX
#define OMATH_QUATERNION_AVX

#include "Quaternion.hpp"

#ifdef __AVX__
#  include <immintrin.h>

namespace omath
{
    template <>
    inline auto operator-(const Quaternion<double>& quat) noexcept
    {
        Quaternion<double> result;
        const auto z = _mm256_setzero_pd();
        _mm256_store_pd(result.v.data(), _mm256_sub_pd(z, _mm256_load_pd(quat.v.data())));
        return result;
    }

    template <>
    inline void negate(Quaternion<double>& quat) noexcept
    {
        const auto z = _mm256_setzero_pd();
        _mm256_store_pd(quat.v.data(), _mm256_sub_pd(z, _mm256_load_pd(quat.v.data())));
    }

    template <>
    [[nodiscard]] inline auto operator+(const Quaternion<double>& quat1,
                                        const Quaternion<double>& quat2) noexcept
    {
        Quaternion<double> result;
        _mm256_store_pd(result.v.data(), _mm256_add_pd(_mm256_load_pd(quat1.v.data()), _mm256_load_pd(quat2.v.data())));
        return result;
    }

    template <>
    inline auto& operator+=(Quaternion<double>& quat1,
                            const Quaternion<double>& quat2) noexcept
    {
        _mm256_store_pd(quat1.v.data(), _mm256_add_pd(_mm256_load_pd(quat1.v.data()), _mm256_load_pd(quat2.v.data())));
        return quat1;
    }

    template <>
    [[nodiscard]] inline auto operator-(const Quaternion<double>& quat1,
                                        const Quaternion<double>& quat2) noexcept
    {
        Quaternion<double> result;
        _mm256_store_pd(result.v.data(), _mm256_sub_pd(_mm256_load_pd(quat1.v.data()), _mm256_load_pd(quat2.v.data())));
        return result;
    }

    template <>
    inline auto& operator-=(Quaternion<double>& quat1,
                            const Quaternion<double>& quat2) noexcept
    {
        _mm256_store_pd(quat1.v.data(), _mm256_sub_pd(_mm256_load_pd(quat1.v.data()), _mm256_load_pd(quat2.v.data())));
        return quat1;
    }

    template <>
    [[nodiscard]] inline auto operator*(const Quaternion<double>& quat,
                                        const double scalar) noexcept
    {
        Quaternion<double> result;
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(result.v.data(), _mm256_mul_pd(_mm256_load_pd(quat.v.data()), s));
        return result;
    }

    template <>
    inline auto& operator*=(Quaternion<double>& quat,
                            const double scalar) noexcept
    {
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(quat.v.data(), _mm256_mul_pd(_mm256_load_pd(quat.v.data()), s));
        return quat;
    }

    template <>
    [[nodiscard]] inline auto operator/(const Quaternion<double>& quat,
                                        const double scalar) noexcept
    {
        Quaternion<double> result;
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(result.v.data(), _mm256_div_pd(_mm256_load_pd(quat.v.data()), s));
        return result;
    }

    template <>
    inline auto& operator/=(Quaternion<double>& quat,
                            const double scalar) noexcept
    {
        const auto s = _mm256_set1_pd(scalar);
        _mm256_store_pd(quat.v.data(), _mm256_div_pd(_mm256_load_pd(quat.v.data()), s));
        return quat;
    }
}

#endif // __AVX__

#endif // OMATH_QUATERNION_AVX
