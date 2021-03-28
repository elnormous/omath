//
// elnormous/math
//

#ifndef MATH_PLANE
#define MATH_PLANE

#include <array>

namespace math
{
    template <typename T> class Plane final
    {
    public:
#if defined(__SSE__)
        alignas(4 * sizeof(T))
#endif
        std::array<T, 4> v{};

        constexpr Plane() noexcept {}

        constexpr Plane(const T a, const T b, const T c, const T d) noexcept:
            v{{a, b, c, d}}
        {
        }

        auto& operator[](const std::size_t index) noexcept { return v[index]; }
        constexpr auto operator[](const std::size_t index) const noexcept { return v[index]; }

        auto& a() noexcept { return v[0]; }
        constexpr auto a() const noexcept { return v[0]; }

        auto& b() noexcept { return v[1]; }
        constexpr auto b() const noexcept { return v[1]; }

        auto& c() noexcept { return v[2]; }
        constexpr auto c() const noexcept { return v[2]; }

        auto& d() noexcept { return v[3]; }
        constexpr auto d() const noexcept { return v[3]; }

        auto operator==(const Plane& p) const noexcept
        {
            return v[0] == p.v[0] && v[1] == p.v[1] && v[2] == p.v[2] && v[3] == p.v[3];
        }

        auto operator!=(const Plane& p) const noexcept
        {
            return v[0] != p.v[0] || v[1] != p.v[1] || v[2] != p.v[2] || v[3] != p.v[3];
        }

        constexpr auto operator-() const noexcept
        {
            return Plane{-v[0], -v[1], -v[2], -v[3]};
        }
    };
}

#endif
