//
// elnormous/math
//

#include <array>

namespace math
{
    template <typename T, std::size_t C, std::size_t R> class Matrix final
    {
    public:
#if defined(__SSE__)
        alignas((C == 4 && R == 4) ? 4 * sizeof(T) : alignof(T))
#endif
        std::array<T, C * R> m{}; // row-major matrix (transformation is pre-multiplying)

        T& operator[](std::size_t index) noexcept { return m[index]; }
        constexpr T operator[](std::size_t index) const noexcept { return m[index]; }
    };
}
