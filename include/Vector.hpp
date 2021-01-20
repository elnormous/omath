//
// elnormous/math
//

#ifndef MATH_VECTOR
#define MATH_VECTOR

#include <array>
#include <type_traits>

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

        T& operator[](std::size_t index) noexcept { return v[index]; }
        constexpr T operator[](std::size_t index) const noexcept { return v[index]; }

        template <std::size_t X = N, std::enable_if_t<(X >= 1)>* = nullptr>
        T& x() noexcept { return v[0]; }

        template <std::size_t X = N, std::enable_if_t<(X >= 1)>* = nullptr>
        constexpr T x() const noexcept { return v[0]; }

        template <std::size_t X = N, std::enable_if_t<(X >= 2)>* = nullptr>
        T& y() noexcept { return v[1]; }

        template <std::size_t X = N, std::enable_if_t<(X >= 2)>* = nullptr>
        constexpr T y() const noexcept { return v[1]; }

        template <std::size_t X = N, std::enable_if_t<(X >= 3)>* = nullptr>
        T& z() noexcept { return v[2]; }

        template <std::size_t X = N, std::enable_if_t<(X >= 3)>* = nullptr>
        constexpr T z() const noexcept { return v[2]; }

        template <std::size_t X = N, std::enable_if_t<(X >= 4)>* = nullptr>
        T& w() noexcept { return v[3]; }

        template <std::size_t X = N, std::enable_if_t<(X >= 4)>* = nullptr>
        constexpr T w() const noexcept { return v[3]; }
    };
}

#endif
