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
    };
}

#endif
