//
// elnormous/math
//

#ifndef MATH_MATRIX
#define MATH_MATRIX

#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>
#ifdef __SSE__
#  include <xmmintrin.h>
#endif

namespace math
{
    template <typename T, std::size_t C, std::size_t R = C, bool simdEnabled = true> class Matrix final
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

        [[nodiscard]] auto operator[](const std::size_t row) noexcept { return &m[row * C]; }
        [[nodiscard]] constexpr auto operator[](const std::size_t row) const noexcept { return &m[row * C]; }

        template <auto X = C, auto Y = R, std::enable_if_t<(X == Y)>* = nullptr>
        [[nodiscard]] static constexpr auto identity() noexcept
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

        [[nodiscard]] constexpr auto operator==(const Matrix& mat) const noexcept
        {
            return std::equal(std::begin(m), std::end(m), std::begin(mat.m));
        }

        [[nodiscard]] constexpr auto operator!=(const Matrix& mat) const noexcept
        {
            return !std::equal(std::begin(m), std::end(m), std::begin(mat.m));
        }

        template <auto X = C, auto Y = R, auto s = simdEnabled, std::enable_if_t<(X != 4 || Y != 4 || !s)>* = nullptr>
        [[nodiscard]] constexpr auto operator-() const noexcept
        {
            return generateNegative(std::make_index_sequence<C * R>{});
        }

        template <auto X = C, auto Y = R, auto s = simdEnabled, std::enable_if_t<(X == 4 && Y == 4 && s)>* = nullptr>
        [[nodiscard]] constexpr auto operator-() const noexcept
        {
#if defined(__SSE__)
            Matrix result;
            __m128 z = _mm_setzero_ps();
            _mm_store_ps(&result.m[0], _mm_sub_ps(z, _mm_load_ps(&m[0])));
            _mm_store_ps(&result.m[4], _mm_sub_ps(z, _mm_load_ps(&m[4])));
            _mm_store_ps(&result.m[8], _mm_sub_ps(z, _mm_load_ps(&m[8])));
            _mm_store_ps(&result.m[12], _mm_sub_ps(z, _mm_load_ps(&m[12])));
            return result;
#else
#  error "SIMD not supported"
#endif
        }

        [[nodiscard]] constexpr const auto operator+(const Matrix& mat) const noexcept
        {
            return generateSum(std::make_index_sequence<C * R>{}, mat);
        }

        auto& operator+=(const Matrix& mat) noexcept
        {
            for (std::size_t i = 0; i < C * R; ++i)
                m[i] += mat.m[i];
            return *this;
        }

        [[nodiscard]] constexpr const auto operator-(const Matrix& mat) const noexcept
        {
            return generateDiff(std::make_index_sequence<C * R>{}, mat);
        }

        auto& operator-=(const Matrix& mat) noexcept
        {
            for (std::size_t i = 0; i < C * R; ++i)
                m[i] -= mat.m[i];
            return *this;
        }

        [[nodiscard]] constexpr const auto operator*(const T scalar) const noexcept
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
        [[nodiscard]] auto operator*(const Matrix<T, C2, R2>& mat) const noexcept
        {
            Matrix<T, C1, R2> result;

            for (std::size_t r = 0; r < R2; ++r)
                for (std::size_t c = 0; c < C1; ++c)
                    for (std::size_t i = 0; i < R1; ++i)
                        result.m[r * C1 + c] += m[i * C1 + c] * mat.m[r * C2 + i];

            return result;
        }

    private:
        template <std::size_t ...I>
        static constexpr auto generateIdentity(const std::index_sequence<I...>)
        {
            return Matrix{(I % C == I / R) ? T(1) : T(0)...};
        }

        template <std::size_t ...I>
        constexpr auto generateNegative(const std::index_sequence<I...>) const
        {
            return Matrix{(-m[I])...};
        }

        template <std::size_t ...I>
        constexpr auto generateSum(const std::index_sequence<I...>, const Matrix& mat) const
        {
            return Matrix{(m[I] + mat.m[I])...};
        }

        template <std::size_t ...I>
        constexpr auto generateDiff(const std::index_sequence<I...>, const Matrix& mat) const
        {
            return Matrix{(m[I] - mat.m[I])...};
        }

        template <std::size_t ...I>
        constexpr auto generateMul(const std::index_sequence<I...>, const T scalar) const
        {
            return Matrix{(m[I] * scalar)...};
        }
    };
}

#endif
