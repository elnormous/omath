//
// elnormous/omath
//

#ifndef OMATH_PLANE
#define OMATH_PLANE

#include <array>
#include "Vector.hpp"

namespace omath
{
    template <typename T>
    class Plane final
    {
    public:
#if defined(__SSE__)
        alignas(4 * sizeof(T))
#endif
        std::array<T, 4> v;

        [[nodiscard]] auto& operator[](const std::size_t index) noexcept { return v[index]; }
        [[nodiscard]] constexpr auto operator[](const std::size_t index) const noexcept { return v[index]; }

        [[nodiscard]] auto& a() noexcept { return v[0]; }
        [[nodiscard]] constexpr auto a() const noexcept { return v[0]; }

        [[nodiscard]] auto& b() noexcept { return v[1]; }
        [[nodiscard]] constexpr auto b() const noexcept { return v[1]; }

        [[nodiscard]] auto& c() noexcept { return v[2]; }
        [[nodiscard]] constexpr auto c() const noexcept { return v[2]; }

        [[nodiscard]] auto& d() noexcept { return v[3]; }
        [[nodiscard]] constexpr auto d() const noexcept { return v[3]; }

        [[nodiscard]] auto operator==(const Plane& p) const noexcept
        {
            return v[0] == p.v[0] && v[1] == p.v[1] && v[2] == p.v[2] && v[3] == p.v[3];
        }

        [[nodiscard]] auto operator!=(const Plane& p) const noexcept
        {
            return v[0] != p.v[0] || v[1] != p.v[1] || v[2] != p.v[2] || v[3] != p.v[3];
        }

        [[nodiscard]] constexpr auto operator+() const noexcept
        {
            return *this;
        }

        [[nodiscard]] constexpr auto operator-() const noexcept
        {
            return Plane{-v[0], -v[1], -v[2], -v[3]};
        }

        [[nodiscard]] constexpr auto dot(const Vector<T, 3>& vec) const noexcept
        {
            return v[0] * vec.v[0] + v[1] * vec.v[1] + v[2] * vec.v[2] + v[3];
        }

        [[nodiscard]] auto distance(const Vector<T, 3>& vec) const
        {
            return std::abs(dot(vec)) /
                std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        }
    };
}

#endif
