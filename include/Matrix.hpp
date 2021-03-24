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
        explicit constexpr Matrix(const A... args) noexcept:
            m{args...}
        {
        }

        auto operator[](const std::size_t row) noexcept { return &m[row * C]; }
        constexpr auto operator[](const std::size_t row) const noexcept { return &m[row * C]; }

        template <auto X = C, auto Y = R, std::enable_if_t<(X == Y)>* = nullptr>
        static constexpr auto identity() noexcept
        {
            return generateIdentity(std::make_index_sequence<C * R>{});
        }

        auto operator==(const Matrix& mat) const noexcept
        {
            for (std::size_t i = 0; i < C * R; ++i)
                if (m[i] != mat.m[i]) return false;
            return true;
        }

        auto operator!=(const Matrix& mat) const noexcept
        {
            for (std::size_t i = 0; i < C * R; ++i)
                if (m[i] != mat.m[i]) return true;
            return false;
        }

        template <auto X = C, auto Y = R, std::enable_if_t<(X == Y)>* = nullptr>
        Matrix operator*(const Matrix& mat) const noexcept
        {
            Matrix result;

            for (std::size_t r = 0; r < R; ++r)
                for (std::size_t c = 0; c < C; ++c)
                    for (std::size_t i = 0; i < C; ++i)
                        result[r][c] += m[i * C + c] * mat.m[r * C + i];

            return result;
        }

    private:
        template <std::size_t...I>
        static constexpr auto generateIdentity(const std::index_sequence<I...>)
        {
            return Matrix{(I % C == I / R) ? T(1) : T(0)...};
        }
    };
}

#endif
