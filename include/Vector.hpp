//
// elnormous/omath
//

#ifndef OMATH_VECTOR
#define OMATH_VECTOR

#include <array>
#include <cmath>
#include <type_traits>

namespace omath
{
    template <typename T, std::size_t dims>
    class Vector final
    {
    public:
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP >= 1 || defined(__ARM_NEON__)
        alignas(std::is_same_v<T, float> && dims == 4 ? dims * sizeof(T) : sizeof(T))
#endif
#if (defined(__SSE2__) || defined(_M_X64) || _M_IX86_FP >= 2) || (defined(__ARM_NEON__) && defined(__aarch64__))
        alignas(std::is_same_v<T, double> && dims == 4 ? dims * sizeof(T) : sizeof(T))
#endif
        std::array<T, dims> v;

        [[nodiscard]] auto& operator[](const std::size_t index) noexcept { return v[index]; }
        [[nodiscard]] constexpr auto operator[](const std::size_t index) const noexcept { return v[index]; }
        [[nodiscard]] auto& operator()(const std::size_t index) noexcept { return v[index]; }
        [[nodiscard]] constexpr auto operator()(const std::size_t index) const noexcept { return v[index]; }

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

    template <typename T, std::size_t dims>
    constexpr void negate(Vector<T, dims>& vector) noexcept
    {
        for (auto& c : vector.v) c = -c;
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator+(const Vector<T, dims>& vector1,
                                           const Vector<T, dims>& vector2) noexcept
    {
        Vector<T, dims> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector1.v[i] + vector2.v[i];
        return result;
    }

    template <typename T, std::size_t dims>
    auto& operator+=(Vector<T, dims>& vector1,
                     const Vector<T, dims>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            vector1.v[i] += vector2.v[i];
        return vector1;
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator-(const Vector<T, dims>& vector1,
                                           const Vector<T, dims>& vector2) noexcept
    {
        Vector<T, dims> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector1.v[i] - vector2.v[i];
        return result;
    }

    template <typename T, std::size_t dims>
    auto& operator-=(Vector<T, dims>& vector1,
                     const Vector<T, dims>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            vector1.v[i] -= vector2.v[i];
        return vector1;
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator*(const Vector<T, dims>& vector,
                                           const T scalar) noexcept
    {
        Vector<T, dims> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector.v[i] * scalar;
        return result;
    }

    template <typename T, std::size_t dims>
    auto& operator*=(Vector<T, dims>& vector, const T scalar) noexcept
    {
        for (auto& c : vector.v) c *= scalar;
        return vector;
    }

    template <typename T, std::size_t dims>
    [[nodiscard]] constexpr auto operator/(const Vector<T, dims>& vector,
                                           const T scalar) noexcept
    {
        Vector<T, dims> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector.v[i] / scalar;
        return result;
    }

    template <typename T, std::size_t dims>
    auto& operator/=(Vector<T, dims>& vector, const T scalar) noexcept
    {
        for (auto& c : vector.v) c /= scalar;
        return vector;
    }

    template <typename T, std::size_t N>
    [[nodiscard]] auto operator*(const T scalar,
                                 const Vector<T, N>& vec) noexcept
    {
        return vec * scalar;
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
    [[nodiscard]] constexpr auto dot(const Vector<T, dims>& vector1,
                                     const Vector<T, dims>& vector2) noexcept
    {
        T result{};
        for (std::size_t i = 0; i < dims; ++i)
            result += vector1.v[i] * vector2[i];
        return result;
    }

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

#include "VectorAvx.hpp"
#include "VectorNeon.hpp"
#include "VectorSse.hpp"

#endif
