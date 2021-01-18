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
            v{a, b, c, d}
        {
        }

        T& operator[](std::size_t index) noexcept { return v[index]; }
        constexpr T operator[](std::size_t index) const noexcept { return v[index]; }
    };
}

#endif
