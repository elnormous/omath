//
// elnormous/omath
//

#ifndef OMATH_VECTOR
#define OMATH_VECTOR

#include <array>
#include <cmath>
#include <type_traits>
#include "Simd.hpp"

namespace omath
{
    template <typename T, std::size_t dims>
    class Vector final
    {
    public:
#if defined(OMATH_SIMD_SSE) || defined(OMATH_SIMD_NEON)
        alignas(std::is_same_v<T, float> && dims == 4 ? dims * sizeof(T) : sizeof(T))
#endif
        std::array<T, dims> v;

        [[nodiscard]] auto& operator[](const std::size_t index) noexcept { return v[index]; }
        [[nodiscard]] constexpr auto operator[](const std::size_t index) const noexcept { return v[index]; }

        [[nodiscard]] auto& x() noexcept
        {
            static_assert(dims >= 1);
            return v[0];
        }

        [[nodiscard]] constexpr auto x() const noexcept
        {
            static_assert(dims >= 1);
            return v[0];
        }

        [[nodiscard]] auto& y() noexcept
        {
            static_assert(dims >= 2);
            return v[1];
        }

        [[nodiscard]] constexpr auto y() const noexcept
        {
            static_assert(dims >= 2);
            return v[1];
        }

        [[nodiscard]] auto& z() noexcept
        {
            static_assert(dims >= 3);
            return v[2];
        }

        [[nodiscard]] constexpr auto z() const noexcept
        {
            static_assert(dims >= 3);
            return v[2];
        }

        [[nodiscard]] auto& w() noexcept
        {
            static_assert(dims >= 4);
            return v[3];
        }

        [[nodiscard]] constexpr auto w() const noexcept
        {
            static_assert(dims >= 4);
            return v[3];
        }
    };

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator==(const Vector<T, dims>& vector1,
                                            const Vector<T, dims>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            if (vector1.v[i] != vector2.v[i]) return false;
        return true;
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator!=(const Vector<T, dims>& vector1,
                                            const Vector<T, dims>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            if (vector1.v[i] != vector2.v[i]) return true;
        return false;
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator<(const Vector<T, dims>& vector1,
                                           const Vector<T, dims>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            if (vector1.v[i] < vector2.v[i]) return true;
        return false;
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator>(const Vector<T, dims>& vector1,
                                           const Vector<T, dims>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            if (vector1.v[i] > vector2.v[i]) return true;
        return false;
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator+(Vector<T, dims>& vector) noexcept
    {
        return vector;
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator-(const Vector<T, dims>& vector) noexcept
    {
        Vector<T, dims> result;
        for (std::size_t i = 0; i < dims; ++i) result.v[i] = -vector.v[i];
        return result;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline auto operator-(const Vector<float, 4>& vector) noexcept
    {
        Vector<float, 4> result;
        const auto z = _mm_setzero_ps();
        _mm_store_ps(result.v.data(), _mm_sub_ps(z, _mm_load_ps(vector.v.data())));
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline auto operator-(const Vector<float, 4>& vector) noexcept
    {
        Vector<float, 4> result;
        vst1q_f32(result.v.data(), vnegq_f32(vld1q_f32(vector.v.data())));
        return result;
    }
#endif

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator+(const Vector<T, dims>& vector1,
                                           const Vector<T, dims>& vector2) noexcept
    {
        Vector<T, dims> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector1.v[i] + vector2.v[i];
        return result;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    [[nodiscard]] inline auto operator+(const Vector<float, 4>& vector1,
                                        const Vector<float, 4>& vector2) noexcept
    {
        Vector<float, 4> result;
        _mm_store_ps(result.v.data(), _mm_add_ps(_mm_load_ps(vector1.v.data()), _mm_load_ps(vector2.v.data())));
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    [[nodiscard]] inline auto operator+(const Vector<float, 4>& vector1,
                                        const Vector<float, 4>& vector2) noexcept
    {
        Vector<float, 4> result;
        vst1q_f32(result.v.data(), vaddq_f32(vld1q_f32(vector1.v.data()), vld1q_f32(vector2.v.data())));
        return result;
    }
#endif

    template <typename T, std::size_t dims>
    constexpr void negate(Vector<T, dims>& vector) noexcept
    {
        for (auto& c : vector.v) c = -c;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline void negate(Vector<float, 4>& vector) noexcept
    {
        const auto z = _mm_setzero_ps();
        _mm_store_ps(vector.v.data(), _mm_sub_ps(z, _mm_load_ps(vector.v.data())));
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline void negate(Vector<float, 4>& vector) noexcept
    {
        vst1q_f32(vector.v.data(), vnegq_f32(vld1q_f32(vector.v.data())));
    }
#endif

    template <typename T, std::size_t dims>
    auto& operator+=(Vector<T, dims>& vector1,
                     const Vector<T, dims>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            vector1.v[i] += vector2.v[i];
        return vector1;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline auto& operator+=(Vector<float, 4>& vector1,
                            const Vector<float, 4>& vector2) noexcept
    {
        _mm_store_ps(vector1.v.data(), _mm_add_ps(_mm_load_ps(vector1.v.data()), _mm_load_ps(vector2.v.data())));
        return vector1;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline auto& operator+=(Vector<float, 4>& vector1,
                            const Vector<float, 4>& vector2) noexcept
    {
        vst1q_f32(vector1.v.data(), vaddq_f32(vld1q_f32(vector1.v.data()), vld1q_f32(vector2.v.data())));
        return vector1;
    }
#endif

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator-(const Vector<T, dims>& vector1,
                                           const Vector<T, dims>& vector2) noexcept
    {
        Vector<T, dims> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector1.v[i] - vector2.v[i];
        return result;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    [[nodiscard]] inline auto operator-(const Vector<float, 4>& vector1,
                                        const Vector<float, 4>& vector2) noexcept
    {
        Vector<float, 4> result;
        _mm_store_ps(result.v.data(), _mm_sub_ps(_mm_load_ps(vector1.v.data()), _mm_load_ps(vector2.v.data())));
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    [[nodiscard]] inline auto operator-(const Vector<float, 4>& vector1,
                                        const Vector<float, 4>& vector2) noexcept
    {
        Vector<float, 4> result;
        vst1q_f32(result.v.data(), vsubq_f32(vld1q_f32(vector1.v.data()), vld1q_f32(vector2.v.data())));
        return result;
    }
#endif

    template <typename T, std::size_t dims>
    auto& operator-=(Vector<T, dims>& vector1,
                     const Vector<T, dims>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            vector1.v[i] -= vector2.v[i];
        return vector1;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline auto& operator-=(Vector<float, 4>& vector1,
                            const Vector<float, 4>& vector2) noexcept
    {
        _mm_store_ps(vector1.v.data(), _mm_sub_ps(_mm_load_ps(vector1.v.data()), _mm_load_ps(vector2.v.data())));
        return vector1;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline auto& operator-=(Vector<float, 4>& vector1,
                            const Vector<float, 4>& vector2) noexcept
    {
        vst1q_f32(vector1.v.data(), vsubq_f32(vld1q_f32(vector1.v.data()), vld1q_f32(vector2.v.data())));
        return vector1;
    }
#endif

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator*(const Vector<T, dims>& vector,
                                           const T scalar) noexcept
    {
        Vector<T, dims> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector.v[i] * scalar;
        return result;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    [[nodiscard]] inline auto operator*(const Vector<float, 4>& vector,
                                        const float scalar) noexcept
    {
        Vector<float, 4> result;
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(result.v.data(), _mm_mul_ps(_mm_load_ps(vector.v.data()), s));
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    [[nodiscard]] inline auto operator*(const Vector<float, 4>& vector,
                                        const float scalar) noexcept
    {
        Vector<float, 4> result;
        const auto s = vdupq_n_f32(scalar);
        vst1q_f32(result.v.data(), vmulq_f32(vld1q_f32(vector.v.data()), s));
        return result;
    }
#endif

    template <typename T, std::size_t dims>
    auto& operator*=(Vector<T, dims>& vector, const T scalar) noexcept
    {
        for (auto& c : vector.v) c *= scalar;
        return vector;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline auto& operator*=(Vector<float, 4>& vector,
                            const float scalar) noexcept
    {
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(vector.v.data(), _mm_mul_ps(_mm_load_ps(vector.v.data()), s));
        return vector;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline auto& operator*=(Vector<float, 4>& vector,
                            const float scalar) noexcept
    {
        const auto s = vdupq_n_f32(scalar);
        vst1q_f32(vector.v.data(), vmulq_f32(vld1q_f32(vector.v.data()), s));
        return vector;
    }
#endif

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator/(const Vector<T, dims>& vector,
                                           const T scalar) noexcept
    {
        Vector<T, dims> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector.v[i] / scalar;
        return result;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    [[nodiscard]] inline auto operator/(const Vector<float, 4>& vector,
                                        const float scalar) noexcept
    {
        Vector<float, 4> result;
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(result.v.data(), _mm_div_ps(_mm_load_ps(vector.v.data()), s));
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    [[nodiscard]] inline auto operator/(const Vector<float, 4>& vector,
                                        const float scalar) noexcept
    {
        Vector<float, 4> result;
        const auto s = vdupq_n_f32(scalar);
        vst1q_f32(result.v.data(), vdivq_f32(vld1q_f32(vector.v.data()), s));
        return result;
    }
#endif

    template <typename T, std::size_t dims>
    auto& operator/=(Vector<T, dims>& vector, const T scalar) noexcept
    {
        for (auto& c : vector.v) c /= scalar;
        return vector;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    inline auto& operator/=(Vector<float, 4>& vector,
                            const float scalar) noexcept
    {
        const auto s = _mm_set1_ps(scalar);
        _mm_store_ps(vector.v.data(), _mm_div_ps(_mm_load_ps(vector.v.data()), s));
        return vector;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    inline auto& operator/=(Vector<float, 4>& vector,
                            const float scalar) noexcept
    {
        const auto s = vdupq_n_f32(scalar);
        vst1q_f32(vector.v.data(), vdivq_f32(vld1q_f32(vector.v.data()), s));
        return vector;
    }
#endif

    template <typename T>
    [[nodiscard]] constexpr auto cross(const Vector<T, 3>& vector1,
                                       const Vector<T, 3>& vector2) noexcept
    {
        return Vector<T, 3>{
            (vector1.v[1] * vector2.v[2]) - (vector1.v[2] * vector2.v[1]),
            (vector1.v[2] * vector2.v[0]) - (vector1.v[0] * vector2.v[2]),
            (vector1.v[0] * vector2.v[1]) - (vector1.v[1] * vector2.v[0])
        };
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] auto length(const Vector<T, dims>& vector) noexcept
    {
        T lengthSquared{};
        for (const auto& c : vector.v) lengthSquared += c * c;
        return std::sqrt(lengthSquared);
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto lengthSquared(const Vector<T, dims>& vector) noexcept
    {
        T lengthSquared{};
        for (const auto& c : vector.v) lengthSquared += c * c;
        return lengthSquared;
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto dot(const Vector<T, dims>& vector1,
                                     const Vector<T, dims>& vector2) noexcept
    {
        T result{};
        for (std::size_t i = 0; i < dims; ++i)
            result += vector1.v[i] * vector2[i];
        return result;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    [[nodiscard]] inline auto dot(const Vector<float, 4>& vector1,
                                  const Vector<float, 4>& vector2) noexcept
    {
        float result;
        const auto t1 = _mm_mul_ps(_mm_load_ps(vector1.v.data()), _mm_load_ps(vector2.v.data()));
        const auto t2 = _mm_add_ps(t1, _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2, 1, 0, 3)));
        const auto t3 = _mm_add_ps(t2, _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1, 0, 3, 2)));
        result = _mm_cvtss_f32(t3);
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON
    template <>
    [[nodiscard]] inline auto dot(const Vector<float, 4>& vector1,
                                  const Vector<float, 4>& vector2) noexcept
    {
        float result;
        const auto t1 = vmulq_f32(vld1q_f32(vector1.v.data()), vld1q_f32(vector2.v.data()));
        const auto t2 = vaddq_f32(t1, vrev64q_f32(t1));
        const auto t3 = vaddq_f32(t2, vcombine_f32(vget_high_f32(t2), vget_low_f32(t2)));
        result = vgetq_lane_f32(t3, 0);
        return result;
    }
#endif

    template <typename T, std::size_t dims>
    [[nodiscard]] auto distance(const Vector<T, dims>& vector1,
                                const Vector<T, dims>& vector2) noexcept
    {
        T distanceSquared{};
        for (std::size_t i = 0; i < dims; ++i)
            distanceSquared += (vector1.v[i] - vector2.v[i]) * (vector1.v[i] - vector2.v[i]);
        return std::sqrt(distanceSquared);
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto distanceSquared(const Vector<T, dims>& vector1,
                                                 const Vector<T, dims>& vector2) noexcept
    {
        T distanceSquared{};
        for (std::size_t i = 0; i < dims; ++i)
            distanceSquared += (vector1.v[i] - vector2.v[i]) * (vector1.v[i] - vector2.v[i]);
        return distanceSquared;
    }

    template <typename T, std::size_t dims>
    void normalize(Vector<T, dims>& vector) noexcept
    {
        if (const auto l = length(vector); l > T(0))
            for (auto& c : vector.v) c /= l;
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] auto normalized(const Vector<T, dims>& vector) noexcept
    {
        Vector<T, dims> result;
        if (const auto l = length(vector); l > T(0))
            for (std::size_t i = 0; i < dims; ++i)
                result.v[i] = vector.v[i] / l;
        return result;
    }
}

#endif
