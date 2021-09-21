//
// elnormous/omath
//

#ifndef OMATH_MATRIX
#define OMATH_MATRIX

#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>
#include "Simd.hpp"
#include "Vector.hpp"

namespace omath
{
    template <typename T, std::size_t rows, std::size_t cols = rows, bool simd = canMatrixUseSimd<T, rows, cols>>
    class Matrix final
    {
        static_assert(!simd);
    public:
        std::array<T, cols * rows> m; // row-major matrix (transformation is pre-multiplying)

        [[nodiscard]] auto operator[](const std::size_t row) noexcept { return &m[row * cols]; }
        [[nodiscard]] constexpr auto operator[](const std::size_t row) const noexcept { return &m[row * cols]; }

        [[nodiscard]] static constexpr auto identity() noexcept
        {
            static_assert(cols == rows);
            return generateIdentity(std::make_index_sequence<cols * rows>{});
        }

        [[nodiscard]] constexpr auto operator==(const Matrix& mat) const noexcept
        {
            return std::equal(std::begin(m), std::end(m), std::begin(mat.m));
        }

        [[nodiscard]] constexpr auto operator!=(const Matrix& mat) const noexcept
        {
            return !std::equal(std::begin(m), std::end(m), std::begin(mat.m));
        }

        [[nodiscard]] constexpr auto operator+() const noexcept
        {
            return *this;
        }
        
        [[nodiscard]] constexpr auto operator-() const noexcept
        {
            return generateNegative(std::make_index_sequence<cols * rows>{});
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

        [[nodiscard]] constexpr const auto operator/(const T scalar) const noexcept
        {
            return generateDiv(std::make_index_sequence<cols * rows>{}, scalar);
        }

        auto& operator/=(const T scalar) noexcept
        {
            for (std::size_t i = 0; i < cols * rows; ++i)
                m[i] /= scalar;
            return *this;
        }

        template <std::size_t rows2, std::size_t cols2, bool simd2>
        [[nodiscard]] const auto operator*(const Matrix<T, rows2, cols2, simd2>& mat) const noexcept
        {
            static_assert(cols == rows2);

            Matrix<T, rows, cols2, simd && simd2> result{};

            // TODO: make constexpr
            for (std::size_t i = 0; i < rows; ++i)
                for (std::size_t j = 0; j < cols2; ++j)
                    for (std::size_t k = 0; k < rows2; ++k)
                        result.m[i * cols2 + j] += m[i * cols + k] * mat.m[k * cols2 + j];

            return result;
        }

        auto& operator*=(const Matrix& mat) noexcept
        {
            static_assert(rows == cols);

            const auto temp = m;
            m = {};

            for (std::size_t i = 0; i < rows; ++i)
                for (std::size_t j = 0; j < cols; ++j)
                    for (std::size_t k = 0; k < rows; ++k)
                        m[i * cols + j] += temp[i * cols + k] * mat.m[k * cols + j];

            return *this;
        }

        void transpose() noexcept
        {
            static_assert(cols == rows);

            using std::swap;

            for (std::size_t i = 1; i < rows; ++i)
                for (std::size_t j = 0; j < i; ++j)
                    swap(m[i * cols + j], m[j * rows + i]);
        }

        [[nodiscard]] constexpr auto transposed() const noexcept
        {
            return generateTransposed(std::make_index_sequence<cols * rows>{});
        }

        [[nodiscard]] constexpr auto determinant() const noexcept
        {
            static_assert(rows > 0 && cols > 0 && rows == cols);
            static_assert(rows <= 2 && cols <= 2);

            if constexpr (rows == 1 && cols == 1)
                return m[0];
            else if constexpr (rows == 2 && cols == 2)
                return m[0] * m[3] - m[1] * m[2];
        }

    private:
        template <std::size_t ...i>
        static constexpr auto generateIdentity(const std::index_sequence<i...>) noexcept
        {
            return Matrix{(i % cols == i / rows) ? T(1) : T(0)...};
        }

        template <std::size_t ...i>
        constexpr auto generateNegative(const std::index_sequence<i...>) const noexcept
        {
            return Matrix{(-m[i])...};
        }

        template <std::size_t ...i>
        constexpr auto generateSum(const std::index_sequence<i...>, const Matrix& mat) const noexcept
        {
            return Matrix{(m[i] + mat.m[i])...};
        }

        template <std::size_t ...i>
        constexpr auto generateDiff(const std::index_sequence<i...>, const Matrix& mat) const noexcept
        {
            return Matrix{(m[i] - mat.m[i])...};
        }

        template <std::size_t ...i>
        constexpr auto generateMul(const std::index_sequence<i...>, const T scalar) const noexcept
        {
            return Matrix{(m[i] * scalar)...};
        }

        template <std::size_t ...i>
        constexpr auto generateDiv(const std::index_sequence<i...>, const T scalar) const noexcept
        {
            return Matrix{(m[i] / scalar)...};
        }

        template <std::size_t ...i>
        constexpr auto generateTransposed(const std::index_sequence<i...>) const noexcept
        {
            return Matrix<T, cols, rows, simd>{(m[i % rows * cols  + i / rows])...};
        }
    };

    template <>
    class Matrix<float, 4, 4, true> final
    {
        static_assert(canMatrixUseSimd<float, 4, 4>);
    public:
        alignas(4 * alignof(float)) std::array<float, 4 * 4> m; // row-major matrix (transformation is pre-multiplying)

        [[nodiscard]] auto operator[](const std::size_t row) noexcept { return &m[row * 4]; }
        [[nodiscard]] constexpr auto operator[](const std::size_t row) const noexcept { return &m[row * 4]; }

        [[nodiscard]] static constexpr auto identity() noexcept
        {
            return Matrix{
                1.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 1.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 1.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 1.0F
            };
        }

        [[nodiscard]] auto operator==(const Matrix& mat) const noexcept
        {
            return std::equal(std::begin(m), std::end(m), std::begin(mat.m));
        }

        [[nodiscard]] auto operator!=(const Matrix& mat) const noexcept
        {
            return !std::equal(std::begin(m), std::end(m), std::begin(mat.m));
        }

        [[nodiscard]] constexpr auto operator+() const noexcept
        {
            return *this;
        }

        [[nodiscard]] auto operator-() const noexcept
        {
            Matrix result;
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            const auto z = _mm_setzero_ps();
            _mm_store_ps(&result.m[0], _mm_sub_ps(z, _mm_load_ps(&m[0])));
            _mm_store_ps(&result.m[4], _mm_sub_ps(z, _mm_load_ps(&m[4])));
            _mm_store_ps(&result.m[8], _mm_sub_ps(z, _mm_load_ps(&m[8])));
            _mm_store_ps(&result.m[12], _mm_sub_ps(z, _mm_load_ps(&m[12])));
#elif defined(__ARM_NEON__)
            vst1q_f32(&result.m[0], vnegq_f32(vld1q_f32(&m[0])));
            vst1q_f32(&result.m[4], vnegq_f32(vld1q_f32(&m[4])));
            vst1q_f32(&result.m[8], vnegq_f32(vld1q_f32(&m[8])));
            vst1q_f32(&result.m[12], vnegq_f32(vld1q_f32(&m[12])));
#endif
            return result;
        }

        [[nodiscard]] const auto operator+(const Matrix& mat) const noexcept
        {
            Matrix result;
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            _mm_store_ps(&result.m[0], _mm_add_ps(_mm_load_ps(&m[0]), _mm_load_ps(&mat.m[0])));
            _mm_store_ps(&result.m[4], _mm_add_ps(_mm_load_ps(&m[4]), _mm_load_ps(&mat.m[4])));
            _mm_store_ps(&result.m[8], _mm_add_ps(_mm_load_ps(&m[8]), _mm_load_ps(&mat.m[8])));
            _mm_store_ps(&result.m[12], _mm_add_ps(_mm_load_ps(&m[12]), _mm_load_ps(&mat.m[12])));
#elif defined(__ARM_NEON__)
            vst1q_f32(&result.m[0], vaddq_f32(vld1q_f32(&m[0]), vld1q_f32(&mat.m[0])));
            vst1q_f32(&result.m[4], vaddq_f32(vld1q_f32(&m[4]), vld1q_f32(&mat.m[4])));
            vst1q_f32(&result.m[8], vaddq_f32(vld1q_f32(&m[8]), vld1q_f32(&mat.m[8])));
            vst1q_f32(&result.m[12], vaddq_f32(vld1q_f32(&m[12]), vld1q_f32(&mat.m[12])));
#endif
            return result;
        }

        auto& operator+=(const Matrix& mat) noexcept
        {
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            _mm_store_ps(&m[0], _mm_add_ps(_mm_load_ps(&m[0]), _mm_load_ps(&mat.m[0])));
            _mm_store_ps(&m[4], _mm_add_ps(_mm_load_ps(&m[4]), _mm_load_ps(&mat.m[4])));
            _mm_store_ps(&m[8], _mm_add_ps(_mm_load_ps(&m[8]), _mm_load_ps(&mat.m[8])));
            _mm_store_ps(&m[12], _mm_add_ps(_mm_load_ps(&m[12]), _mm_load_ps(&mat.m[12])));
#elif defined(__ARM_NEON__)
            vst1q_f32(&m[0], vaddq_f32(vld1q_f32(&m[0]), vld1q_f32(&mat.m[0])));
            vst1q_f32(&m[4], vaddq_f32(vld1q_f32(&m[4]), vld1q_f32(&mat.m[4])));
            vst1q_f32(&m[8], vaddq_f32(vld1q_f32(&m[8]), vld1q_f32(&mat.m[8])));
            vst1q_f32(&m[12], vaddq_f32(vld1q_f32(&m[12]), vld1q_f32(&mat.m[12])));
#endif
            return *this;
        }

        [[nodiscard]] const auto operator-(const Matrix& mat) const noexcept
        {
            Matrix result;
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            _mm_store_ps(&result.m[0], _mm_sub_ps(_mm_load_ps(&m[0]), _mm_load_ps(&mat.m[0])));
            _mm_store_ps(&result.m[4], _mm_sub_ps(_mm_load_ps(&m[4]), _mm_load_ps(&mat.m[4])));
            _mm_store_ps(&result.m[8], _mm_sub_ps(_mm_load_ps(&m[8]), _mm_load_ps(&mat.m[8])));
            _mm_store_ps(&result.m[12], _mm_sub_ps(_mm_load_ps(&m[12]), _mm_load_ps(&mat.m[12])));
#elif defined(__ARM_NEON__)
            vst1q_f32(&result.m[0], vsubq_f32(vld1q_f32(&m[0]), vld1q_f32(&mat.m[0])));
            vst1q_f32(&result.m[4], vsubq_f32(vld1q_f32(&m[4]), vld1q_f32(&mat.m[4])));
            vst1q_f32(&result.m[8], vsubq_f32(vld1q_f32(&m[8]), vld1q_f32(&mat.m[8])));
            vst1q_f32(&result.m[12], vsubq_f32(vld1q_f32(&m[12]), vld1q_f32(&mat.m[12])));
#endif
            return result;
        }

        auto& operator-=(const Matrix& mat) noexcept
        {
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            _mm_store_ps(&m[0], _mm_sub_ps(_mm_load_ps(&m[0]), _mm_load_ps(&mat.m[0])));
            _mm_store_ps(&m[4], _mm_sub_ps(_mm_load_ps(&m[4]), _mm_load_ps(&mat.m[4])));
            _mm_store_ps(&m[8], _mm_sub_ps(_mm_load_ps(&m[8]), _mm_load_ps(&mat.m[8])));
            _mm_store_ps(&m[12], _mm_sub_ps(_mm_load_ps(&m[12]), _mm_load_ps(&mat.m[12])));
#elif defined(__ARM_NEON__)
            vst1q_f32(&m[0], vsubq_f32(vld1q_f32(&m[0]), vld1q_f32(&mat.m[0])));
            vst1q_f32(&m[4], vsubq_f32(vld1q_f32(&m[4]), vld1q_f32(&mat.m[4])));
            vst1q_f32(&m[8], vsubq_f32(vld1q_f32(&m[8]), vld1q_f32(&mat.m[8])));
            vst1q_f32(&m[12], vsubq_f32(vld1q_f32(&m[12]), vld1q_f32(&mat.m[12])));
#endif
            return *this;
        }

        [[nodiscard]] const auto operator*(const float scalar) const noexcept
        {
            Matrix result;
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            const auto s = _mm_set1_ps(scalar);
            _mm_store_ps(&result.m[0], _mm_mul_ps(_mm_load_ps(&m[0]), s));
            _mm_store_ps(&result.m[4], _mm_mul_ps(_mm_load_ps(&m[4]), s));
            _mm_store_ps(&result.m[8], _mm_mul_ps(_mm_load_ps(&m[8]), s));
            _mm_store_ps(&result.m[12], _mm_mul_ps(_mm_load_ps(&m[12]), s));
#elif defined(__ARM_NEON__)
            const auto s = vdupq_n_f32(scalar);
            vst1q_f32(&result.m[0], vmulq_f32(vld1q_f32(&m[0]), s));
            vst1q_f32(&result.m[4], vmulq_f32(vld1q_f32(&m[4]), s));
            vst1q_f32(&result.m[8], vmulq_f32(vld1q_f32(&m[8]), s));
            vst1q_f32(&result.m[12], vmulq_f32(vld1q_f32(&m[12]), s));
#endif
            return result;
        }

        auto& operator*=(const float scalar) noexcept
        {
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            const auto s = _mm_set1_ps(scalar);
            _mm_store_ps(&m[0], _mm_mul_ps(_mm_load_ps(&m[0]), s));
            _mm_store_ps(&m[4], _mm_mul_ps(_mm_load_ps(&m[4]), s));
            _mm_store_ps(&m[8], _mm_mul_ps(_mm_load_ps(&m[8]), s));
            _mm_store_ps(&m[12], _mm_mul_ps(_mm_load_ps(&m[12]), s));
#elif defined(__ARM_NEON__)
            const auto s = vdupq_n_f32(scalar);
            vst1q_f32(&m[0], vmulq_f32(vld1q_f32(&m[0]), s));
            vst1q_f32(&m[4], vmulq_f32(vld1q_f32(&m[4]), s));
            vst1q_f32(&m[8], vmulq_f32(vld1q_f32(&m[8]), s));
            vst1q_f32(&m[12], vmulq_f32(vld1q_f32(&m[12]), s));
#endif
            return *this;
        }

        [[nodiscard]] const auto operator/(const float scalar) const noexcept
        {
            Matrix result;
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            const auto s = _mm_set1_ps(scalar);
            _mm_store_ps(&result.m[0], _mm_div_ps(_mm_load_ps(&m[0]), s));
            _mm_store_ps(&result.m[4], _mm_div_ps(_mm_load_ps(&m[4]), s));
            _mm_store_ps(&result.m[8], _mm_div_ps(_mm_load_ps(&m[8]), s));
            _mm_store_ps(&result.m[12], _mm_div_ps(_mm_load_ps(&m[12]), s));
#elif defined(__ARM_NEON__)
            const auto s = vdupq_n_f32(scalar);
            vst1q_f32(&result.m[0], vdivq_f32(vld1q_f32(&m[0]), s));
            vst1q_f32(&result.m[4], vdivq_f32(vld1q_f32(&m[4]), s));
            vst1q_f32(&result.m[8], vdivq_f32(vld1q_f32(&m[8]), s));
            vst1q_f32(&result.m[12], vdivq_f32(vld1q_f32(&m[12]), s));
#endif
            return result;
        }

        auto& operator/=(const float scalar) noexcept
        {
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            const auto s = _mm_set1_ps(scalar);
            _mm_store_ps(&m[0], _mm_div_ps(_mm_load_ps(&m[0]), s));
            _mm_store_ps(&m[4], _mm_div_ps(_mm_load_ps(&m[4]), s));
            _mm_store_ps(&m[8], _mm_div_ps(_mm_load_ps(&m[8]), s));
            _mm_store_ps(&m[12], _mm_div_ps(_mm_load_ps(&m[12]), s));
#elif defined(__ARM_NEON__)
            const auto s = vdupq_n_f32(scalar);
            vst1q_f32(&m[0], vdivq_f32(vld1q_f32(&m[0]), s));
            vst1q_f32(&m[4], vdivq_f32(vld1q_f32(&m[4]), s));
            vst1q_f32(&m[8], vdivq_f32(vld1q_f32(&m[8]), s));
            vst1q_f32(&m[12], vdivq_f32(vld1q_f32(&m[12]), s));
#endif
            return *this;
        }

        [[nodiscard]] const auto operator*(const Matrix& mat) const noexcept
        {
            Matrix result;
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            const auto row0 = _mm_load_ps(&m[0]);
            const auto row1 = _mm_load_ps(&m[4]);
            const auto row2 = _mm_load_ps(&m[8]);
            const auto row3 = _mm_load_ps(&m[12]);

            for (std::size_t i = 0; i < 4; ++i)
            {
                const auto e0 = _mm_set1_ps(mat.m[i * 4 + 0]);
                const auto e1 = _mm_set1_ps(mat.m[i * 4 + 1]);
                const auto e2 = _mm_set1_ps(mat.m[i * 4 + 2]);
                const auto e3 = _mm_set1_ps(mat.m[i * 4 + 3]);

                const auto v0 = _mm_mul_ps(row0, e0);
                const auto v1 = _mm_mul_ps(row1, e1);
                const auto v2 = _mm_mul_ps(row2, e2);
                const auto v3 = _mm_mul_ps(row3, e3);

                const auto a0 = _mm_add_ps(v0, v1);
                const auto a1 = _mm_add_ps(v2, v3);
                _mm_store_ps(&result.m[i * 4], _mm_add_ps(a0, a1));
            }
#elif defined(__ARM_NEON__)
            const auto row0 = vld1q_f32(&m[0]);
            const auto row1 = vld1q_f32(&m[4]);
            const auto row2 = vld1q_f32(&m[8]);
            const auto row3 = vld1q_f32(&m[12]);

            for (std::size_t i = 0; i < 4; ++i)
            {
                const auto e0 = vdupq_n_f32(mat.m[i * 4 + 0]);
                const auto e1 = vdupq_n_f32(mat.m[i * 4 + 1]);
                const auto e2 = vdupq_n_f32(mat.m[i * 4 + 2]);
                const auto e3 = vdupq_n_f32(mat.m[i * 4 + 3]);

                const auto v0 = vmulq_f32(row0, e0);
                const auto v1 = vmulq_f32(row1, e1);
                const auto v2 = vmulq_f32(row2, e2);
                const auto v3 = vmulq_f32(row3, e3);

                const auto a0 = vaddq_f32(v0, v1);
                const auto a1 = vaddq_f32(v2, v3);
                vst1q_f32(&result.m[i * 4], vaddq_f32(a0, a1));
            }
#endif
            return result;
        }

        template <std::size_t rows2, std::size_t cols2, bool simd2>
        [[nodiscard]] const auto operator*(const Matrix<float, rows2, cols2, simd2>& mat) const noexcept
        {
            static_assert(rows2 == 4);

            Matrix<float, 4, cols2, simd2> result{};

            for (std::size_t i = 0; i < 4; ++i)
                for (std::size_t j = 0; j < cols2; ++j)
                    for (std::size_t k = 0; k < rows2; ++k)
                        result.m[i * cols2 + j] += m[i * 4 + k] * mat.m[k * cols2 + j];

            return result;
        }

        auto& operator*=(const Matrix& mat) noexcept
        {
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            const auto row0 = _mm_load_ps(&m[0]);
            const auto row1 = _mm_load_ps(&m[4]);
            const auto row2 = _mm_load_ps(&m[8]);
            const auto row3 = _mm_load_ps(&m[12]);

            for (std::size_t i = 0; i < 4; ++i)
            {
                const auto e0 = _mm_set1_ps(mat.m[i * 4 + 0]);
                const auto e1 = _mm_set1_ps(mat.m[i * 4 + 1]);
                const auto e2 = _mm_set1_ps(mat.m[i * 4 + 2]);
                const auto e3 = _mm_set1_ps(mat.m[i * 4 + 3]);

                const auto v0 = _mm_mul_ps(row0, e0);
                const auto v1 = _mm_mul_ps(row1, e1);
                const auto v2 = _mm_mul_ps(row2, e2);
                const auto v3 = _mm_mul_ps(row3, e3);

                const auto a0 = _mm_add_ps(v0, v1);
                const auto a1 = _mm_add_ps(v2, v3);
                _mm_store_ps(&m[i * 4], _mm_add_ps(a0, a1));
            }
#elif defined(__ARM_NEON__)
            const auto row0 = vld1q_f32(&m[0]);
            const auto row1 = vld1q_f32(&m[4]);
            const auto row2 = vld1q_f32(&m[8]);
            const auto row3 = vld1q_f32(&m[12]);

            for (std::size_t i = 0; i < 4; ++i)
            {
                const auto e0 = vdupq_n_f32(mat.m[i * 4 + 0]);
                const auto e1 = vdupq_n_f32(mat.m[i * 4 + 1]);
                const auto e2 = vdupq_n_f32(mat.m[i * 4 + 2]);
                const auto e3 = vdupq_n_f32(mat.m[i * 4 + 3]);

                const auto v0 = vmulq_f32(row0, e0);
                const auto v1 = vmulq_f32(row1, e1);
                const auto v2 = vmulq_f32(row2, e2);
                const auto v3 = vmulq_f32(row3, e3);

                const auto a0 = vaddq_f32(v0, v1);
                const auto a1 = vaddq_f32(v2, v3);
                vst1q_f32(&m[i * 4], vaddq_f32(a0, a1));
            }
#endif
            return *this;
        }

        void transpose() noexcept
        {
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            const auto tmp0 = _mm_shuffle_ps(_mm_load_ps(&m[0]), _mm_load_ps(&m[4]), _MM_SHUFFLE(1, 0, 1, 0));
            const auto tmp1 = _mm_shuffle_ps(_mm_load_ps(&m[8]), _mm_load_ps(&m[12]), _MM_SHUFFLE(1, 0, 1, 0));
            const auto tmp2 = _mm_shuffle_ps(_mm_load_ps(&m[0]), _mm_load_ps(&m[4]), _MM_SHUFFLE(3, 2, 3, 2));
            const auto tmp3 = _mm_shuffle_ps(_mm_load_ps(&m[8]), _mm_load_ps(&m[12]), _MM_SHUFFLE(3, 2, 3, 2));
            _mm_store_ps(&m[0], _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(2, 0, 2, 0)));
            _mm_store_ps(&m[4], _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(3, 1, 3, 1)));
            _mm_store_ps(&m[8], _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(2, 0, 2, 0)));
            _mm_store_ps(&m[12], _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(3, 1, 3, 1)));
#elif defined(__ARM_NEON__)
            const auto tmp0 = vtrnq_f32(vld1q_f32(&m[0]), vld1q_f32(&m[4]));
            const auto tmp1 = vtrnq_f32(vld1q_f32(&m[8]), vld1q_f32(&m[12]));
            vst1q_f32(&m[0], vextq_f32(vextq_f32(tmp0.val[0], tmp0.val[0], 2), tmp1.val[0], 2));
            vst1q_f32(&m[4], vextq_f32(vextq_f32(tmp0.val[1], tmp0.val[1], 2), tmp1.val[1], 2));
            vst1q_f32(&m[8], vextq_f32(tmp0.val[0], vextq_f32(tmp1.val[0], tmp1.val[0], 2), 2));
            vst1q_f32(&m[12], vextq_f32(tmp0.val[1], vextq_f32(tmp1.val[1], tmp1.val[1], 2), 2));
#endif
        }

        [[nodiscard]] auto transposed() const noexcept
        {
            Matrix result;
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
            const auto tmp0 = _mm_shuffle_ps(_mm_load_ps(&m[0]), _mm_load_ps(&m[4]), _MM_SHUFFLE(1, 0, 1, 0));
            const auto tmp1 = _mm_shuffle_ps(_mm_load_ps(&m[8]), _mm_load_ps(&m[12]), _MM_SHUFFLE(1, 0, 1, 0));
            const auto tmp2 = _mm_shuffle_ps(_mm_load_ps(&m[0]), _mm_load_ps(&m[4]), _MM_SHUFFLE(3, 2, 3, 2));
            const auto tmp3 = _mm_shuffle_ps(_mm_load_ps(&m[8]), _mm_load_ps(&m[12]), _MM_SHUFFLE(3, 2, 3, 2));
            _mm_store_ps(&result.m[0], _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(2, 0, 2, 0)));
            _mm_store_ps(&result.m[4], _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(3, 1, 3, 1)));
            _mm_store_ps(&result.m[8], _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(2, 0, 2, 0)));
            _mm_store_ps(&result.m[12], _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(3, 1, 3, 1)));
#elif defined(__ARM_NEON__)
            const auto tmp0 = vtrnq_f32(vld1q_f32(&m[0]), vld1q_f32(&m[4]));
            const auto tmp1 = vtrnq_f32(vld1q_f32(&m[8]), vld1q_f32(&m[12]));
            vst1q_f32(&result.m[0], vextq_f32(vextq_f32(tmp0.val[0], tmp0.val[0], 2), tmp1.val[0], 2));
            vst1q_f32(&result.m[4], vextq_f32(vextq_f32(tmp0.val[1], tmp0.val[1], 2), tmp1.val[1], 2));
            vst1q_f32(&result.m[8], vextq_f32(tmp0.val[0], vextq_f32(tmp1.val[0], tmp1.val[0], 2), 2));
            vst1q_f32(&result.m[12], vextq_f32(tmp0.val[1], vextq_f32(tmp1.val[1], tmp1.val[1], 2), 2));
#endif
            return result;
        }

        [[nodiscard]] auto determinant() const noexcept
        {
            return 0.0F; // TODO: implement
        }
    };

    template <typename T, std::size_t rows, std::size_t cols, bool simd>
    [[nodiscard]] auto operator*(const T scalar,
                                 const Matrix<T, rows, cols, simd>& mat) noexcept
    {
        return mat * scalar;
    }

    template <
        typename T, std::size_t dims, bool simdVector,
        std::size_t size, bool simdMatrix
    >
    [[nodiscard]] auto operator*(const Vector<T, dims, simdVector>& vec,
                                 const Matrix<T, size, size, simdMatrix>& mat) noexcept
    {
        static_assert(dims <= size);
        Vector<T, dims, simdVector && simdMatrix> result{};

        for (std::size_t i = 0; i < dims; ++i)
            for (std::size_t j = 0; j < dims; ++j)
                result.v[i] += vec.v[j] * mat.m[j * size + i];

        return result;
    }

    template <>
    [[nodiscard]] auto operator*(const Vector<float, 4, true>& vec,
                                 const Matrix<float, 4, 4, true>& mat) noexcept
    {
        Vector<float, 4, true> result;

#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
        const auto col0 = _mm_set1_ps(vec.v[0]);
        const auto col1 = _mm_set1_ps(vec.v[1]);
        const auto col2 = _mm_set1_ps(vec.v[2]);
        const auto col3 = _mm_set1_ps(vec.v[3]);

        const auto row0 = _mm_load_ps(&mat.m[0]);
        const auto row1 = _mm_load_ps(&mat.m[4]);
        const auto row2 = _mm_load_ps(&mat.m[8]);
        const auto row3 = _mm_load_ps(&mat.m[12]);

        const auto s = _mm_add_ps(_mm_add_ps(_mm_mul_ps(row0, col0),
                                             _mm_mul_ps(row1, col1)),
                                  _mm_add_ps(_mm_mul_ps(row2, col2),
                                             _mm_mul_ps(row3, col3)));
        _mm_store_ps(result.v.data(), s);
#elif defined(__ARM_NEON__)
        const auto col0 = vdupq_n_f32(vec.v[0]);
        const auto col1 = vdupq_n_f32(vec.v[1]);
        const auto col2 = vdupq_n_f32(vec.v[2]);
        const auto col3 = vdupq_n_f32(vec.v[3]);

        const auto row0 = vld1q_f32(&mat.m[0]);
        const auto row1 = vld1q_f32(&mat.m[4]);
        const auto row2 = vld1q_f32(&mat.m[8]);
        const auto row3 = vld1q_f32(&mat.m[12]);

        const auto s = vaddq_f32(vaddq_f32(vmulq_f32(row0, col0),
                                           vmulq_f32(row1, col1)),
                                 vaddq_f32(vmulq_f32(row2, col2),
                                           vmulq_f32(row3, col3)));
        vst1q_f32(result.v.data(), s);
#endif

        return result;
    }

    template <
        typename T, std::size_t dims, bool simdVector,
        std::size_t size, bool simdMatrix
    >
    auto& operator*=(Vector<T, dims, simdVector>& vec,
                     const Matrix<T, size, size, simdMatrix>& mat) noexcept
    {
        static_assert(dims <= size);
        const auto temp = vec.v;
        vec.v = {};

        for (std::size_t i = 0; i < dims; ++i)
            for (std::size_t j = 0; j < dims; ++j)
                vec.v[i] += temp[j] * mat.m[j * size + i];

        return vec;
    }

    template <>
    auto& operator*=(Vector<float, 4, true>& vec,
                     const Matrix<float, 4, 4, true>& mat) noexcept
    {
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
        const auto col0 = _mm_set1_ps(vec.v[0]);
        const auto col1 = _mm_set1_ps(vec.v[1]);
        const auto col2 = _mm_set1_ps(vec.v[2]);
        const auto col3 = _mm_set1_ps(vec.v[3]);

        const auto row0 = _mm_load_ps(&mat.m[0]);
        const auto row1 = _mm_load_ps(&mat.m[4]);
        const auto row2 = _mm_load_ps(&mat.m[8]);
        const auto row3 = _mm_load_ps(&mat.m[12]);

        const auto s = _mm_add_ps(_mm_add_ps(_mm_mul_ps(row0, col0),
                                             _mm_mul_ps(row1, col1)),
                                  _mm_add_ps(_mm_mul_ps(row2, col2),
                                             _mm_mul_ps(row3, col3)));
        _mm_store_ps(vec.v.data(), s);
#elif defined(__ARM_NEON__)
        const auto col0 = vdupq_n_f32(vec.v[0]);
        const auto col1 = vdupq_n_f32(vec.v[1]);
        const auto col2 = vdupq_n_f32(vec.v[2]);
        const auto col3 = vdupq_n_f32(vec.v[3]);

        const auto row0 = vld1q_f32(&mat.m[0]);
        const auto row1 = vld1q_f32(&mat.m[4]);
        const auto row2 = vld1q_f32(&mat.m[8]);
        const auto row3 = vld1q_f32(&mat.m[12]);

        const auto s = vaddq_f32(vaddq_f32(vmulq_f32(row0, col0),
                                           vmulq_f32(row1, col1)),
                                 vaddq_f32(vmulq_f32(row2, col2),
                                           vmulq_f32(row3, col3)));
        vst1q_f32(vec.v.data(), s);
#endif

        return vec;
    }
}

#endif
