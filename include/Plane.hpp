//
// elnormous/omath
//

#ifndef OMATH_PLANE
#define OMATH_PLANE

#include <array>
#include <type_traits>
#include "Simd.hpp"
#include "Vector.hpp"

namespace omath
{
    template <typename T>
    class Plane final
    {
    public:
#ifdef OMATH_SIMD_AVAILABLE
        alignas(std::is_same_v<T, float> ? 4 * sizeof(T) : sizeof(T))
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
    };

    template <typename T>
    [[nodiscard]] constexpr auto dot(const Plane<T>& plane, const Vector<T, 3>& vec) noexcept
    {
        return plane.v[0] * vec.v[0] + plane.v[1] * vec.v[1] + plane.v[2] * vec.v[2] + plane.v[3];
    }

    template <typename T>
    [[nodiscard]] auto distance(const Plane<T>& plane, const Vector<T, 3>& vec)
    {
        return std::abs(dot(plane, vec)) /
            std::sqrt(plane.v[0] * plane.v[0] + plane.v[1] * plane.v[1] + plane.v[2] * plane.v[2]);
    }
}

#endif
