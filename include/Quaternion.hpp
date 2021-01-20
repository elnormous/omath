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

        auto& operator[](std::size_t index) noexcept { return v[index]; }
        constexpr auto operator[](std::size_t index) const noexcept { return v[index]; }

        auto& x() noexcept { return v[0]; }
        constexpr auto x() const noexcept { return v[0]; }

        auto& y() noexcept { return v[1]; }
        constexpr auto y() const noexcept { return v[1]; }

        auto& z() noexcept { return v[2]; }
        constexpr auto z() const noexcept { return v[2]; }

        auto& w() noexcept { return v[3]; }
        constexpr auto w() const noexcept { return v[3]; }

        static constexpr Quaternion identity() noexcept
        {
            return Quaternion{0, 0, 0, 1};
        }
    };
}

#endif
