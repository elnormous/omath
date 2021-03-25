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

        template <auto X = C, auto Y = R, std::enable_if_t<(X == Y)>* = nullptr>
        void transpose() noexcept
        {
            for (std::size_t r = 0; r < R; ++r)
                for (std::size_t c = r + 1; c < C; ++c)
                    std::swap(m[r * C + c], m[c * R + r]);
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

        constexpr const auto operator*(const T scalar) const noexcept
        {
            return generateMul(std::make_index_sequence<C * R>{}, scalar);
        }

        auto& operator*=(const T scalar) noexcept
        {
            for (std::size_t i = 0; i < C * R; ++i)
                m[i] *= scalar;

            return *this;
        }

        template <auto C1 = C, auto R1 = R, std::size_t C2, std::size_t R2, std::enable_if_t<(R1 == C2)>* = nullptr>
        auto operator*(const Matrix<T, C2, R2>& mat) const noexcept
        {
            Matrix<T, C1, R2> result;

            for (std::size_t r = 0; r < R2; ++r)
                for (std::size_t c = 0; c < C1; ++c)
                    for (std::size_t i = 0; i < R1; ++i)
                        result.m[r * C1 + c] += m[i * C1 + c] * mat.m[r * C2 + i];

            return result;
        }

    private:
        template <std::size_t...I>
        static constexpr auto generateIdentity(const std::index_sequence<I...>)
        {
            return Matrix{(I % C == I / R) ? T(1) : T(0)...};
        }

        template <std::size_t...I>
        constexpr auto generateMul(const std::index_sequence<I...>, T scalar) const
        {
            return Matrix{(m[I] * scalar)...};
        }
    };
}

#endif
