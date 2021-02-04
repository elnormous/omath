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
        explicit constexpr Vector(A... args) noexcept:
            v{static_cast<T>(args)...}
        {
        }

        auto& operator[](std::size_t index) noexcept { return v[index]; }
        constexpr auto operator[](std::size_t index) const noexcept { return v[index]; }

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

        constexpr const Vector operator-() const noexcept
        {
            return generateInverse(std::make_index_sequence<N>{});
        }

        const Vector operator+(const Vector& vec) const noexcept
        {
            auto result{*this};
            for (std::size_t i = 0; i < N; ++i)
                result.v[i] += vec.v[i];
            return result;
        }

        Vector& operator+=(const Vector& vec) noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                v[i] += vec.v[i];
            return *this;
        }

        const Vector operator-(const Vector& vec) const noexcept
        {
            auto result{*this};
            for (std::size_t i = 0; i < N; ++i)
                result.v[i] -= vec.v[i];
            return result;
        }

        Vector& operator-=(const Vector& vec) noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                v[i] -= vec.v[i];
            return *this;
        }

        const Vector operator*(const T scalar) const noexcept
        {
            auto result{*this};
            for (T& c : result.v)
                c *= scalar;
            return result;
        }

        Vector& operator*=(const T scalar) noexcept
        {
            for (T& c : v)
                c *= scalar;
            return *this;
        }

        const Vector operator/(const T scalar) const noexcept
        {
            auto result{*this};
            for (T& c : result.v)
                c /= scalar;
            return result;
        }

        Vector& operator/=(const T scalar) noexcept
        {
            for (T& c : v)
                c /= scalar;
            return *this;
        }

        bool operator<(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                if (v[i] < vec.v[i]) return true;
                else if (vec.v[i] < v[i]) return false;

            return false;
        }

        bool operator>(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                if (v[i] > vec.v[i]) return true;
                else if (vec.v[i] > v[i]) return false;

            return false;
        }

        bool operator==(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                if (v[i] != vec.v[i]) return false;
            return true;
        }

        bool operator!=(const Vector& vec) const noexcept
        {
            for (std::size_t i = 0; i < N; ++i)
                if (v[i] != vec.v[i]) return true;
            return false;
        }

    private:
        template <std::size_t...I>
        constexpr Vector generateInverse(std::index_sequence<I...>) const
        {
            return Vector{-v[I]...};
        }
    };
}

#endif
