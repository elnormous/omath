//
// elnormous/math
//

#ifndef MATH_MATRIX
#define MATH_MATRIX

#include <array>
#include <type_traits>
#include <utility>

namespace math
{
    template <typename T, std::size_t C, std::size_t R = C> class Matrix final
    {
    public:
#if defined(__SSE__)
        alignas((C == 4 && R == 4) ? 4 * sizeof(T) : alignof(T))
#endif
        std::array<T, C * R> m{}; // row-major matrix (transformation is pre-multiplying)

        constexpr Matrix() noexcept {}

        template <typename ...A>
        explicit constexpr Matrix(A... args) noexcept:
            m{static_cast<T>(args)...}
        {
        }

        auto& operator[](std::size_t index) noexcept { return m[index]; }
        constexpr auto operator[](std::size_t index) const noexcept { return m[index]; }

        template <auto X = C, auto Y = R, std::enable_if_t<(X == Y)>* = nullptr>
        static constexpr Matrix identity() noexcept
        {
            return generateIdentity(std::make_index_sequence<C * R>{});
        }

    private:
        template <std::size_t...I>
        static constexpr Matrix generateIdentity(std::index_sequence<I...>)
        {
            return Matrix{
                ((I % C == I / R) ? T(1) : T(0))...
            };
        }
    };
}

#endif
