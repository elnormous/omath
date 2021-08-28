//
// elnormous/math
//

#ifndef OMATH_MATRIX
#define OMATH_MATRIX

#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>
#include "Simd.hpp"

namespace omath
{
    template <typename T, std::size_t cols, std::size_t rows = cols, bool simd = canMatrixUseSimd<T, cols, rows>>
    class Matrix final
    {
        static_assert(!simd || canMatrixUseSimd<T, cols, rows>);
    public:
        alignas(simd ? cols * sizeof(T) : alignof(T)) std::array<T, cols * rows> m; // row-major matrix (transformation is pre-multiplying)

        [[nodiscard]] auto operator[](const std::size_t row) noexcept { return &m[row * cols]; }
        [[nodiscard]] constexpr auto operator[](const std::size_t row) const noexcept { return &m[row * cols]; }

        [[nodiscard]] static constexpr auto identity() noexcept
        {
            static_assert(cols == rows);
            return generateIdentity(std::make_index_sequence<cols * rows>{});
        }

        void transpose() noexcept
        {
            static_assert(cols == rows);
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

        [[nodiscard]] auto operator-() const noexcept
        {
            if constexpr (simd)
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
#endif
            }
            else
                return generateNegative(std::make_index_sequence<cols * rows>{});
        }

        [[nodiscard]] constexpr const auto operator+(const Matrix& mat) const noexcept
        {
            if constexpr (simd)
            {
#if defined(__SSE__)
                Matrix result;
                _mm_store_ps(&result.m[0], _mm_add_ps(_mm_load_ps(&m[0]), _mm_load_ps(&mat.m[0])));
                _mm_store_ps(&result.m[4], _mm_add_ps(_mm_load_ps(&m[4]), _mm_load_ps(&mat.m[4])));
                _mm_store_ps(&result.m[8], _mm_add_ps(_mm_load_ps(&m[8]), _mm_load_ps(&mat.m[8])));
                _mm_store_ps(&result.m[12], _mm_add_ps(_mm_load_ps(&m[12]), _mm_load_ps(&mat.m[12])));
                return result;
#elif defined(__ARM_NEON__)
                Matrix result;
                vst1q_f32(&result.m[0], vaddq_f32(vld1q_f32(&m[0]), vld1q_f32(&mat.m[0])));
                vst1q_f32(&result.m[4], vaddq_f32(vld1q_f32(&m[4]), vld1q_f32(&mat.m[4])));
                vst1q_f32(&result.m[8], vaddq_f32(vld1q_f32(&m[8]), vld1q_f32(&mat.m[8])));
                vst1q_f32(&result.m[12], vaddq_f32(vld1q_f32(&m[12]), vld1q_f32(&mat.m[12])));
                return result;
#endif
            }
            else
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
            if constexpr (simd)
            {
#if defined(__SSE__)
                Matrix result;
                _mm_store_ps(&result.m[0], _mm_sub_ps(_mm_load_ps(&m[0]), _mm_load_ps(&mat.m[0])));
                _mm_store_ps(&result.m[4], _mm_sub_ps(_mm_load_ps(&m[4]), _mm_load_ps(&mat.m[4])));
                _mm_store_ps(&result.m[8], _mm_sub_ps(_mm_load_ps(&m[8]), _mm_load_ps(&mat.m[8])));
                _mm_store_ps(&result.m[12], _mm_sub_ps(_mm_load_ps(&m[12]), _mm_load_ps(&mat.m[12])));
                return result;
#elif defined(__ARM_NEON__)
                Matrix result;
                vst1q_f32(&result.m[0], vsubq_f32(vld1q_f32(&m[0]), vld1q_f32(&mat.m[0])));
                vst1q_f32(&result.m[4], vsubq_f32(vld1q_f32(&m[4]), vld1q_f32(&mat.m[4])));
                vst1q_f32(&result.m[8], vsubq_f32(vld1q_f32(&m[8]), vld1q_f32(&mat.m[8])));
                vst1q_f32(&result.m[12], vsubq_f32(vld1q_f32(&m[12]), vld1q_f32(&mat.m[12])));
                return result;
#endif
            }
            else
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

        template <std::size_t c2, std::size_t r2>
        [[nodiscard]] auto operator*(const Matrix<T, c2, r2>& mat) const noexcept
        {
            static_assert(rows == c2);

            Matrix<T, cols, r2> result{};

            for (std::size_t row = 0; row < r2; ++row)
                for (std::size_t col = 0; col < cols; ++col)
                    for (std::size_t i = 0; i < rows; ++i)
                        result.m[row * cols + col] += m[i * cols + col] * mat.m[row * c2 + i];

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
