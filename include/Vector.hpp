//
// elnormous/math
//

#ifndef MATH_VECTOR
#define MATH_VECTOR

#include <array>
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

        auto operator<(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                if (v[i] < vec.v[i]) return true;
                else if (vec.v[i] < v[i]) return false;

            return false;
        }

        auto operator>(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                if (v[i] > vec.v[i]) return true;
                else if (vec.v[i] > v[i]) return false;

            return false;
        }

        auto operator==(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                if (v[i] != vec.v[i]) return false;
            return true;
        }

        auto operator!=(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                if (v[i] != vec.v[i]) return true;
            return false;
        }

        constexpr const auto operator-() const noexcept
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

        const auto operator*(const T scalar) const noexcept
        {
            return generateMul(std::make_index_sequence<N>{}, scalar);
        }

        auto& operator*=(const T scalar) noexcept
        {
            for (T& c : v) c *= scalar;
            return *this;
        }

        const auto operator/(const T scalar) const noexcept
        {
            return generateDiv(std::make_index_sequence<N>{}, scalar);
        }

        auto& operator/=(const T scalar) noexcept
        {
            for (T& c : v) c /= scalar;
            return *this;
        }

    private:
        template <std::size_t...I>
        constexpr auto generateInverse(const std::index_sequence<I...>) const
        {
            return Vector{-v[I]...};
        }

        template <std::size_t...I>
        constexpr auto generateSum(const std::index_sequence<I...>, const Vector& vec) const
        {
            return Vector{(v[I] + vec.v[I])...};
        }

        template <std::size_t...I>
        constexpr auto generateDiff(const std::index_sequence<I...>, const Vector& vec) const
        {
            return Vector{(v[I] - vec.v[I])...};
        }

        template <std::size_t...I>
        constexpr auto generateMul(const std::index_sequence<I...>, T scalar) const
        {
            return Vector{(v[I] * scalar)...};
        }

        template <std::size_t...I>
        constexpr auto generateDiv(const std::index_sequence<I...>, T scalar) const
        {
            return Vector{(v[I] / scalar)...};
        }
    };
}

#endif
