//
// elnormous/math
//

#ifndef MATH_VECTOR
#define MATH_VECTOR

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

namespace math
{
    template <typename T, std::size_t N> class Vector final
    {
    public:
#if defined(__SSE__)
        alignas(N == 4 ? 4 * sizeof(T) : alignof(T))
#endif
        std::array<T, N> v{};

        constexpr Vector() noexcept {}

        template <typename ...A>
        explicit constexpr Vector(const A... args) noexcept:
            v{args...}
        {
        }

        auto& operator[](const std::size_t index) noexcept { return v[index]; }
        constexpr auto operator[](const std::size_t index) const noexcept { return v[index]; }

        template <auto X = N, std::enable_if_t<(X >= 1)>* = nullptr>
        auto& x() noexcept { return v[0]; }

        template <auto X = N, std::enable_if_t<(X >= 1)>* = nullptr>
        constexpr auto x() const noexcept { return v[0]; }

        template <auto X = N, std::enable_if_t<(X >= 2)>* = nullptr>
        auto& y() noexcept { return v[1]; }

        template <auto X = N, std::enable_if_t<(X >= 2)>* = nullptr>
        constexpr auto y() const noexcept { return v[1]; }

        template <auto X = N, std::enable_if_t<(X >= 3)>* = nullptr>
        auto& z() noexcept { return v[2]; }

        template <auto X = N, std::enable_if_t<(X >= 3)>* = nullptr>
        constexpr auto z() const noexcept { return v[2]; }

        template <auto X = N, std::enable_if_t<(X >= 4)>* = nullptr>
        auto& w() noexcept { return v[3]; }

        template <auto X = N, std::enable_if_t<(X >= 4)>* = nullptr>
        constexpr auto w() const noexcept { return v[3]; }

        constexpr auto operator<(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                if (v[i] < vec.v[i]) return true;
                else if (vec.v[i] < v[i]) return false;

            return false;
        }

        constexpr auto operator>(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                if (v[i] > vec.v[i]) return true;
                else if (vec.v[i] > v[i]) return false;

            return false;
        }

        constexpr auto operator==(const Vector& vec) const noexcept
        {
            return std::equal(std::begin(v), std::end(v), std::begin(vec.v));
        }

        constexpr auto operator!=(const Vector& vec) const noexcept
        {
            return !std::equal(std::begin(v), std::end(v), std::begin(vec.v));
        }

        constexpr auto operator-() const noexcept
        {
            return generateInverse(std::make_index_sequence<N>{});
        }

        constexpr const auto operator+(const Vector& vec) const noexcept
        {
            return generateSum(std::make_index_sequence<N>{}, vec);
        }

        auto& operator+=(const Vector& vec) noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                v[i] += vec.v[i];
            return *this;
        }

        constexpr const auto operator-(const Vector& vec) const noexcept
        {
            return generateDiff(std::make_index_sequence<N>{}, vec);
        }

        auto& operator-=(const Vector& vec) noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                v[i] -= vec.v[i];
            return *this;
        }

        constexpr const auto operator*(const T scalar) const noexcept
        {
            return generateMul(std::make_index_sequence<N>{}, scalar);
        }

        auto& operator*=(const T scalar) noexcept
        {
            for (T& c : v) c *= scalar;
            return *this;
        }

        constexpr const auto operator/(const T scalar) const noexcept
        {
            return generateDiv(std::make_index_sequence<N>{}, scalar);
        }

        auto& operator/=(const T scalar) noexcept
        {
            for (T& c : v) c /= scalar;
            return *this;
        }

        auto length() const noexcept
        {
            return std::sqrt(generateLengthSquared(std::make_index_sequence<N>{}));
        }

        constexpr auto lengthSquared() const noexcept
        {
            return generateLengthSquared(std::make_index_sequence<N>{});
        }

        template <auto X = N, std::enable_if_t<(X == 3)>* = nullptr>
        constexpr auto cross(const Vector& vec) const noexcept
        {
            return Vector{
                (v[1] * vec.v[2]) - (v[2] * vec.v[1]),
                (v[2] * vec.v[0]) - (v[0] * vec.v[2]),
                (v[0] * vec.v[1]) - (v[1] * vec.v[0])
            };
        }

        constexpr auto dot(const Vector& vec) const noexcept
        {
            return generateDot(std::make_index_sequence<N>{}, vec);
        }

        auto distance(const Vector& vec) const noexcept
        {
            return std::sqrt(generateDistanceSquared(std::make_index_sequence<N>{}, vec));
        }

        constexpr auto distanceSquared(const Vector& vec) const noexcept
        {
            return generateDistanceSquared(std::make_index_sequence<N>{}, vec);
        }

    private:
        template <std::size_t ...I>
        constexpr auto generateInverse(const std::index_sequence<I...>) const
        {
            return Vector{-v[I]...};
        }

        template <std::size_t ...I>
        constexpr auto generateSum(const std::index_sequence<I...>, const Vector& vec) const
        {
            return Vector{(v[I] + vec.v[I])...};
        }

        template <std::size_t ...I>
        constexpr auto generateDiff(const std::index_sequence<I...>, const Vector& vec) const
        {
            return Vector{(v[I] - vec.v[I])...};
        }

        template <std::size_t ...I>
        constexpr auto generateMul(const std::index_sequence<I...>, const T scalar) const
        {
            return Vector{(v[I] * scalar)...};
        }

        template <std::size_t ...I>
        constexpr auto generateDiv(const std::index_sequence<I...>, const T scalar) const
        {
            return Vector{(v[I] / scalar)...};
        }

        template<typename ...A>
        static constexpr auto sum(const A... args) noexcept
        {
            return (args + ...);
        }

        template <std::size_t ...I>
        constexpr auto generateLengthSquared(const std::index_sequence<I...>) const
        {
            return sum((v[I] * v[I])...);
        }

        template <std::size_t ...I>
        constexpr auto generateDot(const std::index_sequence<I...>, const Vector& vec) const
        {
            return sum((v[I] * vec.v[I])...);
        }

        template <std::size_t ...I>
        constexpr auto generateDistanceSquared(const std::index_sequence<I...>, const Vector& vec) const
        {
            return sum(((v[I] - vec.v[I]) * (v[I] - vec.v[I]))...);
        }
    };
}

#endif
