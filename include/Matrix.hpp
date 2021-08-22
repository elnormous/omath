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
#elif defined(__ARM_NEON__)
#  include <arm_neon.h>
#endif

namespace math
{
    template <typename T, std::size_t cols, std::size_t rows = cols, bool simd = std::is_same_v<T, float> && rows == 4 && cols == 4>
    class Matrix final
    {
    public:
#if defined(__SSE__)
        alignas(simd ? cols * sizeof(T) : alignof(T))
#endif
        std::array<T, cols * rows> m{}; // row-major matrix (transformation is pre-multiplying)

        constexpr Matrix() noexcept {}

        template <typename ...A>
        explicit constexpr Matrix(const A... args) noexcept:
            m{args...}
        {
        }

        [[nodiscard]] auto operator[](const std::size_t row) noexcept { return &m[row * cols]; }
        [[nodiscard]] constexpr auto operator[](const std::size_t row) const noexcept { return &m[row * cols]; }

        template <auto c = cols, auto r = rows, std::enable_if_t<(c == r)>* = nullptr>
        [[nodiscard]] static constexpr auto identity() noexcept
        {
            return generateIdentity(std::make_index_sequence<cols * rows>{});
        }

        template <auto c = cols, auto r = rows, std::enable_if_t<(c == r)>* = nullptr>
        void transpose() noexcept
        {
            for (std::size_t row = 0; row < rows; ++row)
                for (std::size_t col = row + 1; col < cols; ++col)
                    std::swap(m[row * cols + col], m[col * rows + row]);
        }

        [[nodiscard]] constexpr auto operator==(const Matrix& mat) const noexcept
        {
            return std::equal(std::begin(m), std::end(m), std::begin(mat.m));
        }

        [[nodiscard]] constexpr auto operator!=(const Matrix& mat) const noexcept
        {
            return !std::equal(std::begin(m), std::end(m), std::begin(mat.m));
        }

        template <auto c = cols, auto r = rows, auto s = simd, std::enable_if_t<(!std::is_same_v<T, float> || c != 4 || r != 4 || !s)>* = nullptr>
        [[nodiscard]] constexpr auto operator-() const noexcept
        {
            return generateNegative(std::make_index_sequence<cols * rows>{});
        }

        template <auto c = cols, auto r = rows, auto s = simd, std::enable_if_t<(std::is_same_v<T, float> && c == 4 && r == 4 && s)>* = nullptr>
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
#elif defined(__ARM_NEON__)
            Matrix result;
            vst1q_f32(&result.m[0], vnegq_f32(vld1q_f32(&m[0])));
            vst1q_f32(&result.m[4], vnegq_f32(vld1q_f32(&m[4])));
            vst1q_f32(&result.m[8], vnegq_f32(vld1q_f32(&m[8])));
            vst1q_f32(&result.m[12], vnegq_f32(vld1q_f32(&m[12])));
            return result;
#else
#  error "SIMD not supported"
#endif
        }

        [[nodiscard]] constexpr const auto operator+(const Matrix& mat) const noexcept
        {
            return generateSum(std::make_index_sequence<cols * rows>{}, mat);
        }

        auto& operator+=(const Matrix& mat) noexcept
        {
            for (std::size_t i = 0; i < cols * rows; ++i)
                m[i] += mat.m[i];
            return *this;
        }

        [[nodiscard]] constexpr const auto operator-(const Matrix& mat) const noexcept
        {
            return generateDiff(std::make_index_sequence<cols * rows>{}, mat);
        }

        auto& operator-=(const Matrix& mat) noexcept
        {
            for (std::size_t i = 0; i < cols * rows; ++i)
                m[i] -= mat.m[i];
            return *this;
        }

        [[nodiscard]] constexpr const auto operator*(const T scalar) const noexcept
        {
            return generateMul(std::make_index_sequence<cols * rows>{}, scalar);
        }

        auto& operator*=(const T scalar) noexcept
        {
            for (std::size_t i = 0; i < cols * rows; ++i)
                m[i] *= scalar;

            return *this;
        }

        template <auto c1 = cols, auto r1 = rows, std::size_t c2, std::size_t r2, std::enable_if_t<(r1 == c2)>* = nullptr>
        [[nodiscard]] auto operator*(const Matrix<T, c2, r2>& mat) const noexcept
        {
            Matrix<T, c1, r2> result;

            for (std::size_t row = 0; row < r2; ++row)
                for (std::size_t col = 0; col < c1; ++col)
                    for (std::size_t i = 0; i < r1; ++i)
                        result.m[row * c1 + col] += m[i * c1 + col] * mat.m[row * c2 + i];

            return result;
        }

    private:
        template <std::size_t ...i>
        static constexpr auto generateIdentity(const std::index_sequence<i...>)
        {
            return Matrix{(i % cols == i / rows) ? T(1) : T(0)...};
        }

        template <std::size_t ...i>
        constexpr auto generateNegative(const std::index_sequence<i...>) const
        {
            return Matrix{(-m[i])...};
        }

        template <std::size_t ...i>
        constexpr auto generateSum(const std::index_sequence<i...>, const Matrix& mat) const
        {
            return Matrix{(m[i] + mat.m[i])...};
        }

        template <std::size_t ...i>
        constexpr auto generateDiff(const std::index_sequence<i...>, const Matrix& mat) const
        {
            return Matrix{(m[i] - mat.m[i])...};
        }

        template <std::size_t ...i>
        constexpr auto generateMul(const std::index_sequence<i...>, const T scalar) const
        {
            return Matrix{(m[i] * scalar)...};
        }
    };
}

#endif
