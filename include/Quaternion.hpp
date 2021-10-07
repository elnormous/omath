//
// elnormous/omath
//

#ifndef OMATH_QUATERNION
#define OMATH_QUATERNION

#include <array>
#include <type_traits>
#include "Simd.hpp"

namespace omath
{
    template <typename T>
    class Quaternion final
    {
    public:
#if defined(OMATH_SIMD_SSE) || defined(OMATH_SIMD_NEON)
        alignas(std::is_same_v<T, float> ? 4 * sizeof(T) : sizeof(T))
#endif
        std::array<T, 4> v;

        [[nodiscard]] auto& operator[](const std::size_t index) noexcept { return v[index]; }
        [[nodiscard]] constexpr auto operator[](const std::size_t index) const noexcept { return v[index]; }

        [[nodiscard]] auto& x() noexcept { return v[0]; }
        [[nodiscard]] constexpr auto x() const noexcept { return v[0]; }

        [[nodiscard]] auto& y() noexcept { return v[1]; }
        [[nodiscard]] constexpr auto y() const noexcept { return v[1]; }

        [[nodiscard]] auto& z() noexcept { return v[2]; }
        [[nodiscard]] constexpr auto z() const noexcept { return v[2]; }

        [[nodiscard]] auto& w() noexcept { return v[3]; }
        [[nodiscard]] constexpr auto w() const noexcept { return v[3]; }
    };

    template <typename T>
    [[nodiscard]] constexpr auto identityQuaternion() noexcept
    {
        return Quaternion<T>{T(0), T(0), T(0), T(1)};
    }

    template <typename T>
    constexpr void setIdentity(Quaternion<T>& quat) noexcept
    {
        quat.v = {T(0), T(0), T(0), T(1)};
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator==(const Quaternion<T>& quat1,
                                            const Quaternion<T>& quat2) noexcept
    {
        return quat1.v[0] == quat2.v[0] &&
            quat1.v[1] == quat2.v[1] &&
            quat1.v[2] == quat2.v[2] &&
            quat1.v[3] == quat2.v[3];
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator!=(const Quaternion<T>& quat1,
                                            const Quaternion<T>& quat2) noexcept
    {
        return quat1.v[0] != quat2.v[0] ||
            quat1.v[1] != quat2.v[1] ||
            quat1.v[2] != quat2.v[2] ||
            quat1.v[3] != quat2.v[3];
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator+(const Quaternion<T>& quat) noexcept
    {
        return quat;
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator-(const Quaternion<T>& quat) noexcept
    {
        return Quaternion<T>{-quat.v[0], -quat.v[1], -quat.v[2], -quat.v[3]};
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline auto operator-(const Quaternion<float>& quat) noexcept
    {
        Quaternion<float> result;
        const auto z = _mm_setzero_ps();
        _mm_store_ps(result.v.data(), _mm_sub_ps(z, _mm_load_ps(quat.v.data())));
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline auto operator-(const Quaternion<float>& quat) noexcept
    {
        Quaternion<float> result;
        vst1q_f32(result.v.data(), vnegq_f32(vld1q_f32(quat.v.data())));
        return result;
    }
#endif

    template <typename T>
    [[nodiscard]] constexpr auto operator+(const Quaternion<T>& quat1,
                                           const Quaternion<T>& quat2) noexcept
    {
        return Quaternion<T>{
            quat1.v[0] + quat2.v[0],
            quat1.v[1] + quat2.v[1],
            quat1.v[2] + quat2.v[2],
            quat1.v[3] + quat2.v[3]
        };
    }

#ifdef OMATH_SIMD_SSE
    template <>
    [[nodiscard]] inline auto operator+(const Quaternion<float>& quat1,
                                        const Quaternion<float>& quat2) noexcept
    {
        Quaternion<float> result;
        _mm_store_ps(result.v.data(), _mm_add_ps(_mm_load_ps(quat1.v.data()), _mm_load_ps(quat2.v.data())));
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    [[nodiscard]] inline auto operator+(const Quaternion<float>& quat1,
                                        const Quaternion<float>& quat2) noexcept
    {
        Quaternion<float> result;
        vst1q_f32(result.v.data(), vaddq_f32(vld1q_f32(quat1.v.data()), vld1q_f32(quat2.v.data())));
        return result;
    }
#endif

    template <typename T>
    constexpr void negate(Quaternion<T>& quat) noexcept
    {
        quat.v[0] = -quat.v[0];
        quat.v[1] = -quat.v[1];
        quat.v[2] = -quat.v[2];
        quat.v[3] = -quat.v[3];
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline void negate(Quaternion<float>& quat) noexcept
    {
        const auto z = _mm_setzero_ps();
        _mm_store_ps(quat.v.data(), _mm_sub_ps(z, _mm_load_ps(quat.v.data())));
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline void negate(Quaternion<float>& quat) noexcept
    {
        vst1q_f32(quat.v.data(), vnegq_f32(vld1q_f32(quat.v.data())));
    }
#endif

    template <typename T>
    constexpr auto& operator+=(Quaternion<T>& quat1,
                               const Quaternion<T>& quat2) noexcept
    {
        quat1.v[0] += quat2.v[0];
        quat1.v[1] += quat2.v[1];
        quat1.v[2] += quat2.v[2];
        quat1.v[3] += quat2.v[3];

        return quat1;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline auto& operator+=(Quaternion<float>& quat1,
                            const Quaternion<float>& quat2) noexcept
    {
        _mm_store_ps(quat1.v.data(), _mm_add_ps(_mm_load_ps(quat1.v.data()), _mm_load_ps(quat2.v.data())));
        return quat1;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline auto& operator+=(Quaternion<float>& quat1,
                            const Quaternion<float>& quat2) noexcept
    {
        vst1q_f32(quat1.v.data(), vaddq_f32(vld1q_f32(quat1.v.data()), vld1q_f32(quat2.v.data())));
        return quat1;
    }
#endif

    template <typename T>
    [[nodiscard]] constexpr auto operator-(const Quaternion<T>& quat1,
                                           const Quaternion<T>& quat2) noexcept
    {
        return Quaternion<T>{
            quat1.v[0] - quat2.v[0],
            quat1.v[1] - quat2.v[1],
            quat1.v[2] - quat2.v[2],
            quat1.v[3] - quat2.v[3]
        };
    }

#ifdef OMATH_SIMD_SSE
    template <>
    [[nodiscard]] inline auto operator-(const Quaternion<float>& quat1,
                                        const Quaternion<float>& quat2) noexcept
    {
        Quaternion<float> result;
        _mm_store_ps(result.v.data(), _mm_sub_ps(_mm_load_ps(quat1.v.data()), _mm_load_ps(quat2.v.data())));
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    [[nodiscard]] inline auto operator-(const Quaternion<float>& quat1,
                                        const Quaternion<float>& quat2) noexcept
    {
        Quaternion<float> result;
        vst1q_f32(result.v.data(), vsubq_f32(vld1q_f32(quat1.v.data()), vld1q_f32(quat2.v.data())));
        return result;
    }
#endif

    template <typename T>
    constexpr auto& operator-=(Quaternion<T>& quat1,
                               const Quaternion<T>& quat2) noexcept
    {
        quat1.v[0] -= quat2.v[0];
        quat1.v[1] -= quat2.v[1];
        quat1.v[2] -= quat2.v[2];
        quat1.v[3] -= quat2.v[3];

        return quat1;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline auto& operator-=(Quaternion<float>& quat1,
                            const Quaternion<float>& quat2) noexcept
    {
        _mm_store_ps(quat1.v.data(), _mm_sub_ps(_mm_load_ps(quat1.v.data()), _mm_load_ps(quat2.v.data())));
        return quat1;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline auto& operator-=(Quaternion<float>& quat1,
                            const Quaternion<float>& quat2) noexcept
    {
        vst1q_f32(quat1.v.data(), vsubq_f32(vld1q_f32(quat1.v.data()), vld1q_f32(quat2.v.data())));
        return quat1;
    }
#endif

    template <typename T>
    [[nodiscard]] constexpr auto operator*(const Quaternion<T>& quat1,
                                           const Quaternion<T>& quat2) noexcept
    {
        return Quaternion<T>{
            quat1.v[0] * quat2.v[3] + quat1.v[1] * quat2.v[2] - quat1.v[2] * quat2.v[1] + quat1.v[3] * quat2.v[0],
            -quat1.v[0] * quat2.v[2] + quat1.v[1] * quat2.v[3] + quat1.v[2] * quat2.v[0] + quat1.v[3] * quat2.v[1],
            quat1.v[0] * quat2.v[1] - quat1.v[1] * quat2.v[0] + quat1.v[2] * quat2.v[3] + quat1.v[3] * quat2.v[2],
            -quat1.v[0] * quat2.v[0] - quat1.v[1] * quat2.v[1] - quat1.v[2] * quat2.v[2] + quat1.v[3] * quat2.v[3]
        };
    }

    template <typename T>
    constexpr auto& operator*=(Quaternion<T>& quat1,
                               const Quaternion<T>& quat2) noexcept
    {
        quat1.v = {
            quat1.v[0] * quat2.v[3] + quat1.v[1] * quat2.v[2] - quat1.v[2] * quat2.v[1] + quat1.v[3] * quat2.v[0],
            -quat1.v[0] * quat2.v[2] + quat1.v[1] * quat2.v[3] + quat1.v[2] * quat2.v[0] + quat1.v[3] * quat2.v[1],
            quat1.v[0] * quat2.v[1] - quat1.v[1] * quat2.v[0] + quat1.v[2] * quat2.v[3] + quat1.v[3] * quat2.v[2],
            -quat1.v[0] * quat2.v[0] - quat1.v[1] * quat2.v[1] - quat1.v[2] * quat2.v[2] + quat1.v[3] * quat2.v[3]
        };

        return quat1;
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator*(const Quaternion<T>& quat,
                                           const T scalar) noexcept
    {
        return Quaternion<T>{
            quat.v[0] * scalar,
            quat.v[1] * scalar,
            quat.v[2] * scalar,
            quat.v[3] * scalar
        };
    }

#ifdef OMATH_SIMD_SSE
    template <>
    [[nodiscard]] inline auto operator*(const Quaternion<float>& quat,
                                        const float scalar) noexcept
    {
        Quaternion<float> result;
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(result.v.data(), _mm_mul_ps(_mm_load_ps(quat.v.data()), s));
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    [[nodiscard]] inline auto operator*(const Quaternion<float>& quat,
                                        const float scalar) noexcept
    {
        Quaternion<float> result;
        const auto s = vdupq_n_f32(scalar);
        vst1q_f32(result.v.data(), vmulq_f32(vld1q_f32(quat.v.data()), s));
        return result;
    }
#endif

    template <typename T>
    constexpr auto& operator*=(Quaternion<T>& quat,
                               const T scalar) noexcept
    {
        quat.v[0] *= scalar;
        quat.v[1] *= scalar;
        quat.v[2] *= scalar;
        quat.v[3] *= scalar;

        return quat;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline auto& operator*=(Quaternion<float>& quat,
                            const float scalar) noexcept
    {
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(quat.v.data(), _mm_mul_ps(_mm_load_ps(quat.v.data()), s));
        return quat;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline auto& operator*=(Quaternion<float>& quat,
                            const float scalar) noexcept
    {
        const auto s = vdupq_n_f32(scalar);
        vst1q_f32(quat.v.data(), vmulq_f32(vld1q_f32(quat.v.data()), s));
        return quat;
    }
#endif

    template <typename T>
    [[nodiscard]] constexpr auto operator/(const Quaternion<T>& quat,
                                           const T scalar) noexcept
    {
        return Quaternion<T>{
            quat.v[0] / scalar,
            quat.v[1] / scalar,
            quat.v[2] / scalar,
            quat.v[3] / scalar
        };
    }

#ifdef OMATH_SIMD_SSE
    template <>
    [[nodiscard]] inline auto operator/(const Quaternion<float>& quat,
                                        const float scalar) noexcept
    {
        Quaternion<float> result;
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(result.v.data(), _mm_div_ps(_mm_load_ps(quat.v.data()), s));
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    [[nodiscard]] inline auto operator/(const Quaternion<float>& quat,
                                        const float scalar) noexcept
    {
        Quaternion<float> result;
        const auto s = vdupq_n_f32(scalar);
        vst1q_f32(result.v.data(), vdivq_f32(vld1q_f32(quat.v.data()), s));
        return result;
    }
#endif

    template <typename T>
    constexpr auto& operator/=(Quaternion<T>& quat,
                               const T scalar) noexcept
    {
        quat.v[0] /= scalar;
        quat.v[1] /= scalar;
        quat.v[2] /= scalar;
        quat.v[3] /= scalar;

        return quat;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline auto& operator/=(Quaternion<float>& quat,
                            const float scalar) noexcept
    {
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(quat.v.data(), _mm_div_ps(_mm_load_ps(quat.v.data()), s));
        return quat;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline auto& operator/=(Quaternion<float>& quat,
                            const float scalar) noexcept
    {
        const auto s = vdupq_n_f32(scalar);
        vst1q_f32(quat.v.data(), vdivq_f32(vld1q_f32(quat.v.data()), s));
        return quat;
    }
#endif
}

#endif
