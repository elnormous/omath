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
    template <typename T, std::size_t dims, bool simd = canVectorUseSimd<T, dims>>
    class Vector final
    {
        static_assert(!simd || canVectorUseSimd<T, dims>);
    public:
        alignas(simd ? dims * sizeof(T) : alignof(T)) std::array<T, dims> v;

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

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] constexpr auto operator==(const Vector<T, dims, simd1>& vector1,
                                            const Vector<T, dims, simd2>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            if (vector1.v[i] != vector2.v[i]) return false;
        return true;
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] constexpr auto operator!=(const Vector<T, dims, simd1>& vector1,
                                            const Vector<T, dims, simd2>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            if (vector1.v[i] != vector2.v[i]) return true;
        return false;
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] constexpr auto operator<(const Vector<T, dims, simd1>& vector1,
                                           const Vector<T, dims, simd2>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            if (vector1.v[i] < vector2.v[i]) return true;
        return false;
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] constexpr auto operator>(const Vector<T, dims, simd1>& vector1,
                                           const Vector<T, dims, simd2>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            if (vector1.v[i] > vector2.v[i]) return true;
        return false;
    }

    template <typename T, std::size_t dims, bool simd>
    [[nodiscard]] constexpr auto operator+(Vector<T, dims, simd>& vector) noexcept
    {
        return vector;
    }

    template <typename T, std::size_t dims, bool simd>
    [[nodiscard]] constexpr auto operator-(const Vector<T, dims, simd>& vector) noexcept
    {
        Vector<T, dims, simd> result;
        for (std::size_t i = 0; i < dims; ++i) result.v[i] = -vector.v[i];
        return result;
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] constexpr auto operator+(const Vector<T, dims, simd1>& vector1,
                                           const Vector<T, dims, simd2>& vector2) noexcept
    {
        Vector<T, dims, simd1> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector1.v[i] + vector2.v[i];
        return result;
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    auto& operator+=(Vector<T, dims, simd1>& vector1,
                     const Vector<T, dims, simd2>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            vector1.v[i] += vector2.v[i];
        return vector1;
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] constexpr auto operator-(const Vector<T, dims, simd1>& vector1,
                                           const Vector<T, dims, simd2>& vector2) noexcept
    {
        Vector<T, dims, simd1> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector1.v[i] - vector2.v[i];
        return result;
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    auto& operator-=(Vector<T, dims, simd1>& vector1,
                     const Vector<T, dims, simd2>& vector2) noexcept
    {
        for (std::size_t i = 0; i < dims; ++i)
            vector1.v[i] -= vector2.v[i];
        return vector1;
    }

    template <typename T, std::size_t dims, bool simd>
    [[nodiscard]] constexpr auto operator*(const Vector<T, dims, simd>& vector,
                                           const T scalar) noexcept
    {
        Vector<T, dims, simd> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector.v[i] * scalar;
        return result;
    }

    template <typename T, std::size_t dims, bool simd>
    auto& operator*=(Vector<T, dims, simd>& vector, const T scalar) noexcept
    {
        for (auto& c : vector.v) c *= scalar;
        return vector;
    }

    template <typename T, std::size_t dims, bool simd>
    [[nodiscard]] constexpr auto operator/(const Vector<T, dims, simd>& vector,
                                           const T scalar) noexcept
    {
        Vector<T, dims, simd> result;
        for (std::size_t i = 0; i < dims; ++i)
            result.v[i] = vector.v[i] / scalar;
        return result;
    }

    template <typename T, std::size_t dims, bool simd>
    auto& operator/=(Vector<T, dims, simd>& vector, const T scalar) noexcept
    {
        for (auto& c : vector.v) c /= scalar;
        return vector;
    }

    template <typename T, bool simd1, bool simd2>
    [[nodiscard]] constexpr auto cross(const Vector<T, 3, simd1>& vector1,
                                       const Vector<T, 3, simd2>& vector2) noexcept
    {
        return Vector<T, 3, simd1>{
            (vector1.v[1] * vector2.v[2]) - (vector1.v[2] * vector2.v[1]),
            (vector1.v[2] * vector2.v[0]) - (vector1.v[0] * vector2.v[2]),
            (vector1.v[0] * vector2.v[1]) - (vector1.v[1] * vector2.v[0])
        };
    }

    template <typename T, std::size_t dims, bool simd>
    [[nodiscard]] auto length(const Vector<T, dims, simd>& vector) noexcept
    {
        T lengthSquared{};
        for (const auto& c : vector.v) lengthSquared += c * c;
        return std::sqrt(lengthSquared);
    }

    template <typename T, std::size_t dims, bool simd>
    [[nodiscard]] constexpr auto lengthSquared(const Vector<T, dims, simd>& vector) noexcept
    {
        T lengthSquared{};
        for (const auto& c : vector.v) lengthSquared += c * c;
        return lengthSquared;
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] constexpr auto dot(const Vector<T, dims, simd1>& vector1,
                                     const Vector<T, dims, simd2>& vector2) noexcept
    {
        T result{};
        for (std::size_t i = 0; i < dims; ++i)
            result += vector1.v[i] * vector2[i];
        return result;
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] auto distance(const Vector<T, dims, simd1>& vector1,
                                const Vector<T, dims, simd2>& vector2) noexcept
    {
        T distanceSquared{};
        for (std::size_t i = 0; i < dims; ++i)
            distanceSquared += (vector1.v[i] - vector2.v[i]) * (vector1.v[i] - vector2.v[i]);
        return std::sqrt(distanceSquared);
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] constexpr auto distanceSquared(const Vector<T, dims, simd1>& vector1,
                                                 const Vector<T, dims, simd2>& vector2) noexcept
    {
        T distanceSquared{};
        for (std::size_t i = 0; i < dims; ++i)
            distanceSquared += (vector1.v[i] - vector2.v[i]) * (vector1.v[i] - vector2.v[i]);
        return distanceSquared;
    }

    template <typename T, std::size_t dims, bool simd>
    void normalize(Vector<T, dims, simd>& vector) noexcept
    {
        if (const auto l = length(vector); l > T(0))
            for (auto& c : vector.v) c /= l;
    }

    template <typename T, std::size_t dims, bool simd>
    [[nodiscard]] auto normalized(const Vector<T, dims, simd>& vector) noexcept
    {
        Vector<T, dims, simd> result;
        if (const auto l = length(vector); l > T(0))
            for (std::size_t i = 0; i < dims; ++i)
                result.v[i] = vector.v[i] / l;
        return result;
    }
}

#endif
