//
// elnormous/math
//

#ifndef OMATH_MATRIX
#define OMATH_MATRIX

#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
#  include <xmmintrin.h>
#elif defined(__ARM_NEON__)
#  include <arm_neon.h>
#endif

namespace omath
{
    template <class T, std::size_t cols, std::size_t rows>
    struct CanMatrixUseSimd: std::false_type {};

#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0 || defined(__ARM_NEON__)
    template <>
    struct CanMatrixUseSimd<float, 4, 4>: std::true_type {};
#endif

    template <class T, std::size_t cols, std::size_t rows>
    inline constexpr bool canMatrixUseSimd = CanMatrixUseSimd<T, cols, rows>::value;

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

            if constexpr (simd)
            {
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
                const __m128 tmp0 = _mm_shuffle_ps(_mm_load_ps(&m[0]), _mm_load_ps(&m[4]), _MM_SHUFFLE(1, 0, 1, 0));
                const __m128 tmp1 = _mm_shuffle_ps(_mm_load_ps(&m[8]), _mm_load_ps(&m[12]), _MM_SHUFFLE(1, 0, 1, 0));
                const __m128 tmp2 = _mm_shuffle_ps(_mm_load_ps(&m[0]), _mm_load_ps(&m[4]), _MM_SHUFFLE(3, 2, 3, 2));
                const __m128 tmp3 = _mm_shuffle_ps(_mm_load_ps(&m[8]), _mm_load_ps(&m[12]), _MM_SHUFFLE(3, 2, 3, 2));
                _mm_store_ps(&m[0], _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(2, 0, 2, 0)));
                _mm_store_ps(&m[4], _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(3, 1, 3, 1)));
                _mm_store_ps(&m[8], _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(2, 0, 2, 0)));
                _mm_store_ps(&m[12], _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(3, 1, 3, 1)));
#elif defined(__ARM_NEON__)
                const float32x4x2_t tmp0 = vtrnq_f32(vld1q_f32(&m[0]), vld1q_f32(&m[4]));
                const float32x4x2_t tmp1 = vtrnq_f32(vld1q_f32(&m[8]), vld1q_f32(&m[12]));
                vst1q_f32(&m[0], vextq_f32(vextq_f32(tmp0.val[0], tmp0.val[0], 2), tmp1.val[0], 2));
                vst1q_f32(&m[4], vextq_f32(vextq_f32(tmp0.val[1], tmp0.val[1], 2), tmp1.val[1], 2));
                vst1q_f32(&m[8], vextq_f32(tmp0.val[0], vextq_f32(tmp1.val[0], tmp1.val[0], 2), 2));
                vst1q_f32(&m[12], vextq_f32(tmp0.val[1], vextq_f32(tmp1.val[1], tmp1.val[1], 2), 2));
#endif
            }
            else
                for (std::size_t row = 0; row < rows; ++row)
                    for (std::size_t col = row + 1; col < cols; ++col)
                        std::swap(m[row * cols + col], m[col * rows + row]);
        }

        [[nodiscard]] auto determinant() const noexcept
        {
            static_assert(rows > 0 && cols > 0 && rows == cols);
            static_assert(rows <= 2 && cols <= 2);

            if (rows == 1 && cols == 1)
                return m[0];
            else if (rows == 2 && cols == 2)
                return m[0] * m[3] - m[1] * m[2];
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
                Matrix result;
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
                const __m128 z = _mm_setzero_ps();
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
            else
                return generateNegative(std::make_index_sequence<cols * rows>{});
        }

        [[nodiscard]] constexpr const auto operator+(const Matrix& mat) const noexcept
        {
            if constexpr (simd)
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
            else
                return generateSum(std::make_index_sequence<cols * rows>{}, mat);
        }

        auto& operator+=(const Matrix& mat) noexcept
        {
            if constexpr (simd)
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
            }
            else
                for (std::size_t i = 0; i < cols * rows; ++i)
                    m[i] += mat.m[i];
            return *this;
        }

        [[nodiscard]] constexpr const auto operator-(const Matrix& mat) const noexcept
        {
            if constexpr (simd)
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
            else
                return generateDiff(std::make_index_sequence<cols * rows>{}, mat);
        }

        auto& operator-=(const Matrix& mat) noexcept
        {
            if constexpr (simd)
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
            }
            else
                for (std::size_t i = 0; i < cols * rows; ++i)
                    m[i] -= mat.m[i];
            return *this;
        }

        [[nodiscard]] const auto operator*(const T scalar) const noexcept
        {
            if constexpr (simd)
            {
                Matrix result;
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
                const __m128 s = _mm_set1_ps(scalar);
                _mm_store_ps(&result.m[0], _mm_mul_ps(_mm_load_ps(&m[0]), s));
                _mm_store_ps(&result.m[4], _mm_mul_ps(_mm_load_ps(&m[4]), s));
                _mm_store_ps(&result.m[8], _mm_mul_ps(_mm_load_ps(&m[8]), s));
                _mm_store_ps(&result.m[12], _mm_mul_ps(_mm_load_ps(&m[12]), s));
#elif defined(__ARM_NEON__)
                const float32x4_t s = vdupq_n_f32(scalar);
                vst1q_f32(&result.m[0], vmulq_f32(vld1q_f32(&m[0]), s));
                vst1q_f32(&result.m[4], vmulq_f32(vld1q_f32(&m[4]), s));
                vst1q_f32(&result.m[8], vmulq_f32(vld1q_f32(&m[8]), s));
                vst1q_f32(&result.m[12], vmulq_f32(vld1q_f32(&m[12]), s));
#endif
                return result;
            }
            else
                return generateMul(std::make_index_sequence<cols * rows>{}, scalar);
        }

        auto& operator*=(const T scalar) noexcept
        {
            if constexpr (simd)
            {
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
                const __m128 s = _mm_set1_ps(scalar);
                _mm_store_ps(&m[0], _mm_mul_ps(_mm_load_ps(&m[0]), s));
                _mm_store_ps(&m[4], _mm_mul_ps(_mm_load_ps(&m[4]), s));
                _mm_store_ps(&m[8], _mm_mul_ps(_mm_load_ps(&m[8]), s));
                _mm_store_ps(&m[12], _mm_mul_ps(_mm_load_ps(&m[12]), s));
#elif defined(__ARM_NEON__)
                const float32x4_t s = vdupq_n_f32(scalar);
                vst1q_f32(&m[0], vmulq_f32(vld1q_f32(&m[0]), s));
                vst1q_f32(&m[4], vmulq_f32(vld1q_f32(&m[4]), s));
                vst1q_f32(&m[8], vmulq_f32(vld1q_f32(&m[8]), s));
                vst1q_f32(&m[12], vmulq_f32(vld1q_f32(&m[12]), s));
#endif
            }
            else
                for (std::size_t i = 0; i < cols * rows; ++i)
                    m[i] *= scalar;

            return *this;
        }

        template <std::size_t cols2, std::size_t rows2, bool simd2>
        [[nodiscard]] auto operator*(const Matrix<T, cols2, rows2, simd2>& mat) const noexcept
        {
            static_assert(rows == cols2);

            if constexpr (simd && simd2)
            {
                Matrix<T, cols, rows2, simd && simd2> result;

                for (std::size_t i = 0; i < 4; ++i)
                {
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
                    const __m128 e0 = _mm_set1_ps(mat.m[i * 4 + 0]);
                    const __m128 e1 = _mm_set1_ps(mat.m[i * 4 + 1]);
                    const __m128 e2 = _mm_set1_ps(mat.m[i * 4 + 2]);
                    const __m128 e3 = _mm_set1_ps(mat.m[i * 4 + 3]);

                    const __m128 v0 = _mm_mul_ps(_mm_load_ps(&m[0]), e0);
                    const __m128 v1 = _mm_mul_ps(_mm_load_ps(&m[4]), e1);
                    const __m128 v2 = _mm_mul_ps(_mm_load_ps(&m[8]), e2);
                    const __m128 v3 = _mm_mul_ps(_mm_load_ps(&m[12]), e3);

                    const __m128 a0 = _mm_add_ps(v0, v1);
                    const __m128 a1 = _mm_add_ps(v2, v3);
                    _mm_store_ps(&result.m[i * 4], _mm_add_ps(a0, a1));
#elif defined(__ARM_NEON__)
                    const float32x4_t e0 = vdupq_n_f32(mat.m[i * 4 + 0]);
                    const float32x4_t e1 = vdupq_n_f32(mat.m[i * 4 + 1]);
                    const float32x4_t e2 = vdupq_n_f32(mat.m[i * 4 + 2]);
                    const float32x4_t e3 = vdupq_n_f32(mat.m[i * 4 + 3]);

                    const float32x4_t v0 = vmulq_f32(vld1q_f32(&m[0]), e0);
                    const float32x4_t v1 = vmulq_f32(vld1q_f32(&m[4]), e1);
                    const float32x4_t v2 = vmulq_f32(vld1q_f32(&m[8]), e2);
                    const float32x4_t v3 = vmulq_f32(vld1q_f32(&m[12]), e3);

                    const float32x4_t a0 = vaddq_f32(v0, v1);
                    const float32x4_t a1 = vaddq_f32(v2, v3);
                    vst1q_f32(&result.m[i * 4], vaddq_f32(a0, a1));
#endif
                }
                return result;
            }
            else
            {
                Matrix<T, cols, rows2, simd && simd2> result{};

                for (std::size_t row = 0; row < rows2; ++row)
                    for (std::size_t col = 0; col < cols; ++col)
                        for (std::size_t i = 0; i < rows; ++i)
                            result.m[row * cols + col] += m[i * cols + col] * mat.m[row * cols2 + i];

                return result;
            }
        }

        auto& operator*=(const Matrix& mat) noexcept
        {
            static_assert(rows == cols);

            if constexpr (simd)
            {
                const auto temp = m;

                for (std::size_t i = 0; i < 4; ++i)
                {
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP != 0
                    const __m128 e0 = _mm_set1_ps(mat.m[i * 4 + 0]);
                    const __m128 e1 = _mm_set1_ps(mat.m[i * 4 + 1]);
                    const __m128 e2 = _mm_set1_ps(mat.m[i * 4 + 2]);
                    const __m128 e3 = _mm_set1_ps(mat.m[i * 4 + 3]);

                    const __m128 v0 = _mm_mul_ps(_mm_load_ps(&temp[0]), e0);
                    const __m128 v1 = _mm_mul_ps(_mm_load_ps(&temp[4]), e1);
                    const __m128 v2 = _mm_mul_ps(_mm_load_ps(&temp[8]), e2);
                    const __m128 v3 = _mm_mul_ps(_mm_load_ps(&temp[12]), e3);

                    const __m128 a0 = _mm_add_ps(v0, v1);
                    const __m128 a1 = _mm_add_ps(v2, v3);
                    _mm_store_ps(&m[i * 4], _mm_add_ps(a0, a1));
#elif defined(__ARM_NEON__)
                    const float32x4_t e0 = vdupq_n_f32(mat.m[i * 4 + 0]);
                    const float32x4_t e1 = vdupq_n_f32(mat.m[i * 4 + 1]);
                    const float32x4_t e2 = vdupq_n_f32(mat.m[i * 4 + 2]);
                    const float32x4_t e3 = vdupq_n_f32(mat.m[i * 4 + 3]);

                    const float32x4_t v0 = vmulq_f32(vld1q_f32(&temp[0]), e0);
                    const float32x4_t v1 = vmulq_f32(vld1q_f32(&temp[4]), e1);
                    const float32x4_t v2 = vmulq_f32(vld1q_f32(&temp[8]), e2);
                    const float32x4_t v3 = vmulq_f32(vld1q_f32(&temp[12]), e3);

                    const float32x4_t a0 = vaddq_f32(v0, v1);
                    const float32x4_t a1 = vaddq_f32(v2, v3);
                    vst1q_f32(&m[i * 4], vaddq_f32(a0, a1));
#endif
                }
                return *this;
            }
            else
            {
                const auto temp = m;
                m = {};

                for (std::size_t row = 0; row < rows; ++row)
                    for (std::size_t col = 0; col < cols; ++col)
                        for (std::size_t i = 0; i < rows; ++i)
                            m[row * cols + col] += temp[i * cols + col] * mat.m[row * cols + i];

                return *this;
            }
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

    template <typename T, std::size_t cols, std::size_t rows, bool simd>
    [[nodiscard]] auto operator*(const T scalar, const Matrix<T, cols, rows, simd>& m) noexcept
    {
        return m * scalar;
    }
}

#endif
