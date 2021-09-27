//
// elnormous/omath
//

#ifndef OMATH_VECTOR
#define OMATH_VECTOR

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>
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

        [[nodiscard]] constexpr auto operator<(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < dims; ++i)
                if (v[i] < vec.v[i]) return true;
                else if (vec.v[i] < v[i]) return false;

            return false;
        }

        [[nodiscard]] constexpr auto operator>(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < dims; ++i)
                if (v[i] > vec.v[i]) return true;
                else if (vec.v[i] > v[i]) return false;

            return false;
        }

        [[nodiscard]] constexpr auto operator==(const Vector& vec) const noexcept
        {
            return std::equal(std::begin(v), std::end(v), std::begin(vec.v));
        }

        [[nodiscard]] constexpr auto operator!=(const Vector& vec) const noexcept
        {
            return !std::equal(std::begin(v), std::end(v), std::begin(vec.v));
        }

        [[nodiscard]] constexpr auto operator+() const noexcept
        {
            return *this;
        }

        [[nodiscard]] constexpr auto operator-() const noexcept
        {
            return generateInverse(std::make_index_sequence<dims>{});
        }

        [[nodiscard]] constexpr auto operator+(const Vector& vec) const noexcept
        {
            return generateSum(std::make_index_sequence<dims>{}, vec);
        }

        auto& operator+=(const Vector& vec) noexcept
        {
            for (std::size_t i = 0; i < dims; ++i)
                v[i] += vec.v[i];
            return *this;
        }

        [[nodiscard]] constexpr auto operator-(const Vector& vec) const noexcept
        {
            return generateDiff(std::make_index_sequence<dims>{}, vec);
        }

        auto& operator-=(const Vector& vec) noexcept
        {
            for (std::size_t i = 0; i < dims; ++i)
                v[i] -= vec.v[i];
            return *this;
        }

        [[nodiscard]] constexpr auto operator*(const T scalar) const noexcept
        {
            return generateMul(std::make_index_sequence<dims>{}, scalar);
        }

        auto& operator*=(const T scalar) noexcept
        {
            for (auto& c : v) c *= scalar;
            return *this;
        }

        [[nodiscard]] constexpr auto operator/(const T scalar) const noexcept
        {
            return generateDiv(std::make_index_sequence<dims>{}, scalar);
        }

        auto& operator/=(const T scalar) noexcept
        {
            for (auto& c : v) c /= scalar;
            return *this;
        }

    private:
        template <std::size_t ...I>
        constexpr auto generateInverse(const std::index_sequence<I...>) const noexcept
        {
            return Vector{-v[I]...};
        }

        template <std::size_t ...I>
        constexpr auto generateSum(const std::index_sequence<I...>, const Vector& vec) const noexcept
        {
            return Vector{(v[I] + vec.v[I])...};
        }

        template <std::size_t ...I>
        constexpr auto generateDiff(const std::index_sequence<I...>, const Vector& vec) const noexcept
        {
            return Vector{(v[I] - vec.v[I])...};
        }

        template <std::size_t ...I>
        constexpr auto generateMul(const std::index_sequence<I...>, const T scalar) const noexcept
        {
            return Vector{(v[I] * scalar)...};
        }

        template <std::size_t ...I>
        constexpr auto generateDiv(const std::index_sequence<I...>, const T scalar) const noexcept
        {
            return Vector{(v[I] / scalar)...};
        }
    };

    namespace detail
    {
        template <typename T, std::size_t dims, bool simd, std::size_t ...I>
        constexpr auto generateLengthSquared(const Vector<T, dims, simd>& vector,
                                             const std::index_sequence<I...>) noexcept
        {
            return ((vector.v[I] * vector.v[I]) + ...);
        }

        template <typename T, std::size_t dims, bool simd1, bool simd2, std::size_t ...I>
        constexpr auto generateDot(const Vector<T, dims, simd1>& vector1,
                                   const Vector<T, dims, simd2>& vector2,
                                   const std::index_sequence<I...>) noexcept
        {
            return ((vector1.v[I] * vector2.v[I]) + ...);
        }

        template <typename T, std::size_t dims, bool simd1, bool simd2, std::size_t ...I>
        constexpr auto generateDistanceSquared(const Vector<T, dims, simd1>& vector1,
                                               const Vector<T, dims, simd2>& vector2,
                                               const std::index_sequence<I...>) noexcept
        {
            return (((vector1.v[I] - vector2.v[I]) * (vector1.v[I] - vector2.v[I])) + ...);
        }
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
        return std::sqrt(detail::generateLengthSquared(vector, std::make_index_sequence<dims>{}));
    }

    template <typename T, std::size_t dims, bool simd>
    [[nodiscard]] constexpr auto lengthSquared(const Vector<T, dims, simd>& vector) noexcept
    {
        return detail::generateLengthSquared(vector, std::make_index_sequence<dims>{});
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] constexpr auto dot(const Vector<T, dims, simd1>& vector1,
                                     const Vector<T, dims, simd2>& vector2) noexcept
    {
        return detail::generateDot(vector1, vector2, std::make_index_sequence<dims>{});
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] auto distance(const Vector<T, dims, simd1>& vector1,
                                const Vector<T, dims, simd2>& vector2) noexcept
    {
        return std::sqrt(detail::generateDistanceSquared(vector1, vector2, std::make_index_sequence<dims>{}));
    }

    template <typename T, std::size_t dims, bool simd1, bool simd2>
    [[nodiscard]] constexpr auto distanceSquared(const Vector<T, dims, simd1>& vector1,
                                                 const Vector<T, dims, simd2>& vector2) noexcept
    {
        return detail::generateDistanceSquared(vector1, vector2, std::make_index_sequence<dims>{});
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
