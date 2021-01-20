//
// elnormous/math
//

#ifndef MATH_QUATERNION
#define MATH_QUATERNION

#include <array>
#include <type_traits>

namespace math
{
    template <typename T> class Quaternion final
    {
    public:
#if defined(__SSE__)
        alignas(4 * sizeof(T))
#endif
        std::array<T, 4> v{};

        constexpr Quaternion() noexcept {}

        constexpr Quaternion(const T x, const T y, const T z, const T w) noexcept:
            v{x, y, z, w}
        {
        }

        T& operator[](std::size_t index) noexcept { return v[index]; }
        constexpr T operator[](std::size_t index) const noexcept { return v[index]; }

        T& x() noexcept { return v[0]; }
        constexpr T x() const noexcept { return v[0]; }

        T& y() noexcept { return v[1]; }
        constexpr T y() const noexcept { return v[1]; }

        T& z() noexcept { return v[2]; }
        constexpr T z() const noexcept { return v[2]; }

        T& w() noexcept { return v[3]; }
        constexpr T w() const noexcept { return v[3]; }

        static constexpr Quaternion identity() noexcept
        {
            return Quaternion{0, 0, 0, 1};
        }
    };
}

#endif
