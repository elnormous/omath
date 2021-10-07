//
// elnormous/omath
//

#ifndef OMATH_MATRIX
#define OMATH_MATRIX

#include <array>
#include <type_traits>
#include "Simd.hpp"
#include "Vector.hpp"

namespace omath
{
    template <typename T, std::size_t rows, std::size_t cols = rows>
    class Matrix final
    {
    public:
#if defined(OMATH_SIMD_SSE) || defined(OMATH_SIMD_NEON)
        alignas(std::is_same_v<T, float> && rows == 4 && cols == 4 ? cols * sizeof(T) : alignof(T))
#endif
#if defined(OMATH_SIMD_SSE2) || defined(OMATH_SIMD_NEON64)
        alignas(std::is_same_v<T, double> && rows == 4 && cols == 4 ? cols * sizeof(T) : alignof(T))
#endif
        std::array<T, cols * rows> m; // row-major matrix (transformation is pre-multiplying)

        [[nodiscard]] auto operator[](const std::size_t row) noexcept { return &m[row * cols]; }
        [[nodiscard]] constexpr auto operator[](const std::size_t row) const noexcept { return &m[row * cols]; }
    };

    template <typename T, std::size_t size>
    [[nodiscard]] constexpr auto identityMatrix() noexcept
    {
        Matrix<T, size, size> result;
        for (std::size_t i = 0; i < size; ++i)
            for (std::size_t j = 0; j < size; ++j)
                result.m[j * size + i] = (j == i) ? T(1) : T(0);
        return result;
    }

    template <typename T, std::size_t size>
    constexpr void setIdentity(Matrix<T, size, size>& matrix) noexcept
    {
        for (std::size_t i = 0; i < size; ++i)
            for (std::size_t j = 0; j < size; ++j)
                matrix.m[j * size + i] = (j == i) ? T(1) : T(0);
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator==(const Matrix<T, rows, cols>& matrix1,
                                            const Matrix<T, rows, cols>& matrix2) noexcept
    {
        for (std::size_t i = 0; i < rows * cols; ++i)
            if (matrix1.m[i] != matrix2.m[i]) return false;
        return true;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator!=(const Matrix<T, rows, cols>& matrix1,
                                            const Matrix<T, rows, cols>& matrix2) noexcept
    {
        for (std::size_t i = 0; i < rows * cols; ++i)
            if (matrix1.m[i] != matrix2.m[i]) return true;
        return false;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator+(const Matrix<T, rows, cols>& matrix) noexcept
    {
        return matrix;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator-(const Matrix<T, rows, cols>& matrix)noexcept
    {
        Matrix<T, rows, cols> result;
        for (std::size_t i = 0; i < rows * cols; ++i) result.m[i] = -matrix.m[i];
        return result;
    }

#ifdef OMATH_SIMD_SSE2
    template <>
    [[nodiscard]] inline auto operator-(const Matrix<double, 4, 4>& matrix) noexcept
    {
        Matrix<double, 4, 4> result;
        const auto z = _mm_setzero_pd();
        _mm_store_pd(&result.m[0], _mm_sub_pd(z, _mm_load_pd(&matrix.m[0])));
        _mm_store_pd(&result.m[2], _mm_sub_pd(z, _mm_load_pd(&matrix.m[2])));
        _mm_store_pd(&result.m[4], _mm_sub_pd(z, _mm_load_pd(&matrix.m[4])));
        _mm_store_pd(&result.m[6], _mm_sub_pd(z, _mm_load_pd(&matrix.m[6])));
        _mm_store_pd(&result.m[8], _mm_sub_pd(z, _mm_load_pd(&matrix.m[8])));
        _mm_store_pd(&result.m[10], _mm_sub_pd(z, _mm_load_pd(&matrix.m[10])));
        _mm_store_pd(&result.m[12], _mm_sub_pd(z, _mm_load_pd(&matrix.m[12])));
        _mm_store_pd(&result.m[14], _mm_sub_pd(z, _mm_load_pd(&matrix.m[14])));
        return result;
    }
#endif

#ifdef OMATH_SIMD_NEON64
    template <>
    [[nodiscard]] inline auto operator-(const Matrix<double, 4, 4>& matrix) noexcept
    {
        Matrix<double, 4, 4> result;
        vst1q_f64(&result.m[0], vnegq_f64(vld1q_f64(&matrix.m[0])));
        vst1q_f64(&result.m[2], vnegq_f64(vld1q_f64(&matrix.m[2])));
        vst1q_f64(&result.m[4], vnegq_f64(vld1q_f64(&matrix.m[4])));
        vst1q_f64(&result.m[6], vnegq_f64(vld1q_f64(&matrix.m[6])));
        vst1q_f64(&result.m[8], vnegq_f64(vld1q_f64(&matrix.m[8])));
        vst1q_f64(&result.m[10], vnegq_f64(vld1q_f64(&matrix.m[10])));
        vst1q_f64(&result.m[12], vnegq_f64(vld1q_f64(&matrix.m[12])));
        vst1q_f64(&result.m[14], vnegq_f64(vld1q_f64(&matrix.m[14])));
        return result;
    }
#endif

    template <typename T, std::size_t rows, std::size_t cols>
    constexpr void negate(Matrix<T, rows, cols>& matrix) noexcept
    {
        for (auto& c : matrix.m) c = -c;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator+(const Matrix<T, rows, cols>& matrix1,
                                           const Matrix<T, rows, cols>& matrix2) noexcept
    {
        Matrix<T, rows, cols> result;
        for (std::size_t i = 0; i < rows * cols; ++i)
            result.m[i] = matrix1.m[i] + matrix2.m[i];
        return result;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    auto& operator+=(Matrix<T, rows, cols>& matrix1,
                     const Matrix<T, rows, cols>& matrix2) noexcept
    {
        for (std::size_t i = 0; i < cols * rows; ++i)
            matrix1.m[i] += matrix2.m[i];
        return matrix1;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator-(const Matrix<T, rows, cols>& matrix1,
                                           const Matrix<T, rows, cols>& matrix2) noexcept
    {
        Matrix<T, rows, cols> result;
        for (std::size_t i = 0; i < rows * cols; ++i)
            result.m[i] = matrix1.m[i] - matrix2.m[i];
        return result;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    auto& operator-=(Matrix<T, rows, cols>& matrix1,
                     const Matrix<T, rows, cols>& matrix2) noexcept
    {
        for (std::size_t i = 0; i < cols * rows; ++i)
            matrix1.m[i] -= matrix2.m[i];
        return matrix1;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator*(const Matrix<T, rows, cols>& matrix,
                                           const T scalar) noexcept
    {
        Matrix<T, rows, cols> result;
        for (std::size_t i = 0; i < rows * cols; ++i)
            result.m[i] = matrix.m[i] * scalar;
        return result;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    auto& operator*=(Matrix<T, rows, cols>& matrix,
                     const T scalar) noexcept
    {
        for (std::size_t i = 0; i < cols * rows; ++i)
            matrix.m[i] *= scalar;
        return matrix;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator/(const Matrix<T, rows, cols>& matrix,
                                           const T scalar) noexcept
    {
        Matrix<T, rows, cols> result;
        for (std::size_t i = 0; i < rows * cols; ++i)
            result.m[i] = matrix.m[i] / scalar;
        return result;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    auto& operator/=(Matrix<T, rows, cols>& matrix,
                     const T scalar) noexcept
    {
        for (std::size_t i = 0; i < cols * rows; ++i)
            matrix.m[i] /= scalar;
        return matrix;
    }

    template <typename T, std::size_t rows, std::size_t cols, std::size_t cols2>
    [[nodiscard]] constexpr auto operator*(const Matrix<T, rows, cols>& matrix1,
                                           const Matrix<T, cols, cols2>& matrix2) noexcept
    {
        Matrix<T, rows, cols2> result{};

        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols2; ++j)
                for (std::size_t k = 0; k < cols; ++k)
                    result.m[i * cols2 + j] += matrix1.m[i * cols + k] * matrix2.m[k * cols2 + j];

        return result;
    }

    template <typename T, std::size_t size>
    auto& operator*=(Matrix<T, size, size>& matrix1,
                     const Matrix<T, size, size>& matrix2) noexcept
    {
        const auto temp = matrix1.m;
        matrix1.m = {};

        for (std::size_t i = 0; i < size; ++i)
            for (std::size_t j = 0; j < size; ++j)
                for (std::size_t k = 0; k < size; ++k)
                    matrix1.m[i * size + j] += temp[i * size + k] * matrix2.m[k * size + j];

        return matrix1;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] auto operator*(const T scalar,
                                 const Matrix<T, rows, cols>& mat) noexcept
    {
        return mat * scalar;
    }

    template <
        typename T, std::size_t dims,
        std::size_t size,
        std::enable_if<(size <= dims)>* = nullptr
    >
    [[nodiscard]] auto operator*(const Vector<T, dims>& vector,
                                 const Matrix<T, size, size>& matrix) noexcept
    {
        Vector<T, dims> result{};

        for (std::size_t i = 0; i < dims; ++i)
            for (std::size_t j = 0; j < dims; ++j)
                result.v[i] += vector.v[j] * matrix.m[j * size + i];

        return result;
    }

    template <
        typename T, std::size_t dims,
        std::size_t size
    >
    auto& operator*=(Vector<T, dims>& vector,
                     const Matrix<T, size, size>& matrix) noexcept
    {
        static_assert(dims <= size);
        const auto temp = vector.v;
        vector.v = {};

        for (std::size_t i = 0; i < dims; ++i)
            for (std::size_t j = 0; j < dims; ++j)
                vector.v[i] += temp[j] * matrix.m[j * size + i];

        return vector;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto transposed(const Matrix<T, rows, cols>& matrix) noexcept
    {
        Matrix<T, cols, rows> result;
        for (std::size_t i = 0; i < cols; ++i)
            for (std::size_t j = 0; j < rows; ++j)
                result.m[i * rows + j] = matrix.m[j * cols + i];
        return result;
    }

#ifdef OMATH_SIMD_SSE
    template <>
    [[nodiscard]] inline auto transposed(const Matrix<float, 4, 4>& matrix) noexcept
    {
        Matrix<float, 4, 4> result;
        const auto tmp0 = _mm_shuffle_ps(_mm_load_ps(&matrix.m[0]), _mm_load_ps(&matrix.m[4]), _MM_SHUFFLE(1, 0, 1, 0));
        const auto tmp1 = _mm_shuffle_ps(_mm_load_ps(&matrix.m[8]), _mm_load_ps(&matrix.m[12]), _MM_SHUFFLE(1, 0, 1, 0));
        const auto tmp2 = _mm_shuffle_ps(_mm_load_ps(&matrix.m[0]), _mm_load_ps(&matrix.m[4]), _MM_SHUFFLE(3, 2, 3, 2));
        const auto tmp3 = _mm_shuffle_ps(_mm_load_ps(&matrix.m[8]), _mm_load_ps(&matrix.m[12]), _MM_SHUFFLE(3, 2, 3, 2));
        _mm_store_ps(&result.m[0], _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(2, 0, 2, 0)));
        _mm_store_ps(&result.m[4], _mm_shuffle_ps(tmp0, tmp1, _MM_SHUFFLE(3, 1, 3, 1)));
        _mm_store_ps(&result.m[8], _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(2, 0, 2, 0)));
        _mm_store_ps(&result.m[12], _mm_shuffle_ps(tmp2, tmp3, _MM_SHUFFLE(3, 1, 3, 1)));
        return result;
    }
#endif

    template <typename T, std::size_t size>
    void transpose(Matrix<T, size, size>& matrix) noexcept
    {
        for (std::size_t i = 1; i < size; ++i)
            for (std::size_t j = 0; j < i; ++j)
            {
                T temp = std::move(matrix.m[i * size + j]);
                matrix.m[i * size + j] = std::move(matrix.m[j * size + i]);
                matrix.m[j * size + i] = std::move(temp);
            }
    }

    template <typename T, std::size_t size, std::enable_if<(size <= 4)>* = nullptr>
    [[nodiscard]] constexpr auto determinant(const Matrix<T, size, size>& matrix) noexcept
    {
        if constexpr (size == 0)
            return T(1);
        if constexpr (size == 1)
            return matrix.m[0];
        else if constexpr (size == 2)
            return matrix.m[0] * matrix.m[3] - matrix.m[1] * matrix.m[2];
        else if constexpr (size == 3)
            return matrix.m[0] * matrix.m[4] * matrix.m[8] +
                matrix.m[1] * matrix.m[5] * matrix.m[6] +
                matrix.m[2] * matrix.m[3] * matrix.m[7] -
                matrix.m[2] * matrix.m[4] * matrix.m[6] -
                matrix.m[1] * matrix.m[3] * matrix.m[8] -
                matrix.m[0] * matrix.m[5] * matrix.m[7];
        else if constexpr (size == 4)
        {
            const auto a0 = matrix.m[0] * matrix.m[5] - matrix.m[1] * matrix.m[4];
            const auto a1 = matrix.m[0] * matrix.m[6] - matrix.m[2] * matrix.m[4];
            const auto a2 = matrix.m[0] * matrix.m[7] - matrix.m[3] * matrix.m[4];
            const auto a3 = matrix.m[1] * matrix.m[6] - matrix.m[2] * matrix.m[5];
            const auto a4 = matrix.m[1] * matrix.m[7] - matrix.m[3] * matrix.m[5];
            const auto a5 = matrix.m[2] * matrix.m[7] - matrix.m[3] * matrix.m[6];
            const auto b0 = matrix.m[8] * matrix.m[13] - matrix.m[9] * matrix.m[12];
            const auto b1 = matrix.m[8] * matrix.m[14] - matrix.m[10] * matrix.m[12];
            const auto b2 = matrix.m[8] * matrix.m[15] - matrix.m[11] * matrix.m[12];
            const auto b3 = matrix.m[9] * matrix.m[14] - matrix.m[10] * matrix.m[13];
            const auto b4 = matrix.m[9] * matrix.m[15] - matrix.m[11] * matrix.m[13];
            const auto b5 = matrix.m[10] * matrix.m[15] - matrix.m[11] * matrix.m[14];

            return a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0;
        }
    }

    template <typename T, std::size_t size>
    void invert(Matrix<T, size, size>& matrix) noexcept
    {
        static_assert(size <= 4);

        if constexpr (size == 1)
            matrix.m[0] = 1.0F / matrix.m[0];
        else if constexpr (size == 2)
        {
            const auto det = matrix.m[0] * matrix.m[3] - matrix.m[1] * matrix.m[2];
            const std::array<T, size * size> adjugate{
                matrix.m[3],
                -matrix.m[1],
                -matrix.m[2],
                matrix.m[0]
            };

            matrix.m[0] = adjugate[0] / det;
            matrix.m[1] = adjugate[1] / det;
            matrix.m[2] = adjugate[2] / det;
            matrix.m[3] = adjugate[3] / det;
        }
        else if constexpr (size == 3)
        {
            const auto a0 = matrix.m[4] * matrix.m[8] - matrix.m[5] * matrix.m[7];
            const auto a1 = matrix.m[3] * matrix.m[8] - matrix.m[5] * matrix.m[6];
            const auto a2 = matrix.m[3] * matrix.m[7] - matrix.m[4] * matrix.m[6];

            const auto det = matrix.m[0] * a0 - matrix.m[1] * a1 + matrix.m[2] * a2;

            const std::array<T, size * size> adjugate{
                a0,
                -matrix.m[1] * matrix.m[8] + matrix.m[2] * matrix.m[7],
                matrix.m[1] * matrix.m[5] - matrix.m[2] * matrix.m[4],

                -a1,
                matrix.m[0] * matrix.m[8] - matrix.m[2] * matrix.m[6],
                -matrix.m[0] * matrix.m[5] + matrix.m[2] * matrix.m[3],

                a2,
                -matrix.m[0] * matrix.m[7] + matrix.m[1] * matrix.m[6],
                matrix.m[0] * matrix.m[4] - matrix.m[1] * matrix.m[3]
            };

            matrix.m[0] = adjugate[0] / det;
            matrix.m[1] = adjugate[1] / det;
            matrix.m[2] = adjugate[2] / det;
            matrix.m[3] = adjugate[3] / det;
            matrix.m[4] = adjugate[4] / det;
            matrix.m[5] = adjugate[5] / det;
            matrix.m[6] = adjugate[6] / det;
            matrix.m[7] = adjugate[7] / det;
            matrix.m[8] = adjugate[8] / det;
        }
        else if constexpr (size == 4)
        {
            const auto a0 = matrix.m[0] * matrix.m[5] - matrix.m[1] * matrix.m[4];
            const auto a1 = matrix.m[0] * matrix.m[6] - matrix.m[2] * matrix.m[4];
            const auto a2 = matrix.m[0] * matrix.m[7] - matrix.m[3] * matrix.m[4];
            const auto a3 = matrix.m[1] * matrix.m[6] - matrix.m[2] * matrix.m[5];
            const auto a4 = matrix.m[1] * matrix.m[7] - matrix.m[3] * matrix.m[5];
            const auto a5 = matrix.m[2] * matrix.m[7] - matrix.m[3] * matrix.m[6];
            const auto b0 = matrix.m[8] * matrix.m[13] - matrix.m[9] * matrix.m[12];
            const auto b1 = matrix.m[8] * matrix.m[14] - matrix.m[10] * matrix.m[12];
            const auto b2 = matrix.m[8] * matrix.m[15] - matrix.m[11] * matrix.m[12];
            const auto b3 = matrix.m[9] * matrix.m[14] - matrix.m[10] * matrix.m[13];
            const auto b4 = matrix.m[9] * matrix.m[15] - matrix.m[11] * matrix.m[13];
            const auto b5 = matrix.m[10] * matrix.m[15] - matrix.m[11] * matrix.m[14];

            const auto det = a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0;

            const std::array<T, size * size> adjugate{
                matrix.m[5] * b5 - matrix.m[6] * b4 + matrix.m[7] * b3,
                -(matrix.m[1] * b5 - matrix.m[2] * b4 + matrix.m[3] * b3),
                matrix.m[13] * a5 - matrix.m[14] * a4 + matrix.m[15] * a3,
                -(matrix.m[9] * a5 - matrix.m[10] * a4 + matrix.m[11] * a3),

                -(matrix.m[4] * b5 - matrix.m[6] * b2 + matrix.m[7] * b1),
                matrix.m[0] * b5 - matrix.m[2] * b2 + matrix.m[3] * b1,
                -(matrix.m[12] * a5 - matrix.m[14] * a2 + matrix.m[15] * a1),
                matrix.m[8] * a5 - matrix.m[10] * a2 + matrix.m[11] * a1,

                matrix.m[4] * b4 - matrix.m[5] * b2 + matrix.m[7] * b0,
                -(matrix.m[0] * b4 - matrix.m[1] * b2 + matrix.m[3] * b0),
                matrix.m[12] * a4 - matrix.m[13] * a2 + matrix.m[15] * a0,
                -(matrix.m[8] * a4 - matrix.m[9] * a2 + matrix.m[11] * a0),

                -(matrix.m[4] * b3 - matrix.m[5] * b1 + matrix.m[6] * b0),
                matrix.m[0] * b3 - matrix.m[1] * b1 + matrix.m[2] * b0,
                -(matrix.m[12] * a3 - matrix.m[13] * a1 + matrix.m[14] * a0),
                matrix.m[8] * a3 - matrix.m[9] * a1 + matrix.m[10] * a0
            };

            matrix.m[0] = adjugate[0] / det;
            matrix.m[1] = adjugate[1] / det;
            matrix.m[2] = adjugate[2] / det;
            matrix.m[3] = adjugate[3] / det;
            matrix.m[4] = adjugate[4] / det;
            matrix.m[5] = adjugate[5] / det;
            matrix.m[6] = adjugate[6] / det;
            matrix.m[7] = adjugate[7] / det;
            matrix.m[8] = adjugate[8] / det;
            matrix.m[9] = adjugate[9] / det;
            matrix.m[10] = adjugate[10] / det;
            matrix.m[11] = adjugate[11] / det;
            matrix.m[12] = adjugate[12] / det;
            matrix.m[13] = adjugate[13] / det;
            matrix.m[14] = adjugate[14] / det;
            matrix.m[15] = adjugate[15] / det;
        }
    }

    template <typename T, std::size_t size>
    [[nodiscard]] constexpr auto inverse(const Matrix<T, size, size>& matrix) noexcept
    {
        static_assert(size <= 4);

        Matrix<T, size, size> result;

        if constexpr (size == 1)
            result.m[0] = 1.0F / matrix.m[0];
        else if constexpr (size == 2)
        {
            const auto det = matrix.m[0] * matrix.m[3] - matrix.m[1] * matrix.m[2];
            result.m[0] = matrix.m[3] / det;
            result.m[1] = -matrix.m[1] / det;
            result.m[2] = -matrix.m[2] / det;
            result.m[3] = matrix.m[0] / det;
        }
        else if constexpr (size == 3)
        {
            const auto a0 = matrix.m[4] * matrix.m[8] - matrix.m[5] * matrix.m[7];
            const auto a1 = matrix.m[3] * matrix.m[8] - matrix.m[5] * matrix.m[6];
            const auto a2 = matrix.m[3] * matrix.m[7] - matrix.m[4] * matrix.m[6];

            const auto det = matrix.m[0] * a0 - matrix.m[1] * a1 + matrix.m[2] * a2;

            result.m[0] = a0 / det;
            result.m[1] = -(matrix.m[1] * matrix.m[8] - matrix.m[2] * matrix.m[7]) / det;
            result.m[2] = (matrix.m[1] * matrix.m[5] - matrix.m[2] * matrix.m[4]) / det;

            result.m[3] = -a1 / det;
            result.m[4] = (matrix.m[0] * matrix.m[8] - matrix.m[2] * matrix.m[6]) / det;
            result.m[5] = -(matrix.m[0] * matrix.m[5] - matrix.m[2] * matrix.m[3]) / det;

            result.m[6] = a2 / det;
            result.m[7] = -(matrix.m[0] * matrix.m[7] - matrix.m[1] * matrix.m[6]) / det;
            result.m[8] = (matrix.m[0] * matrix.m[4] - matrix.m[1] * matrix.m[3]) / det;
        }
        else if constexpr (size == 4)
        {
            const auto a0 = matrix.m[0] * matrix.m[5] - matrix.m[1] * matrix.m[4];
            const auto a1 = matrix.m[0] * matrix.m[6] - matrix.m[2] * matrix.m[4];
            const auto a2 = matrix.m[0] * matrix.m[7] - matrix.m[3] * matrix.m[4];
            const auto a3 = matrix.m[1] * matrix.m[6] - matrix.m[2] * matrix.m[5];
            const auto a4 = matrix.m[1] * matrix.m[7] - matrix.m[3] * matrix.m[5];
            const auto a5 = matrix.m[2] * matrix.m[7] - matrix.m[3] * matrix.m[6];
            const auto b0 = matrix.m[8] * matrix.m[13] - matrix.m[9] * matrix.m[12];
            const auto b1 = matrix.m[8] * matrix.m[14] - matrix.m[10] * matrix.m[12];
            const auto b2 = matrix.m[8] * matrix.m[15] - matrix.m[11] * matrix.m[12];
            const auto b3 = matrix.m[9] * matrix.m[14] - matrix.m[10] * matrix.m[13];
            const auto b4 = matrix.m[9] * matrix.m[15] - matrix.m[11] * matrix.m[13];
            const auto b5 = matrix.m[10] * matrix.m[15] - matrix.m[11] * matrix.m[14];

            const auto det = a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0;

            result.m[0] = (matrix.m[5] * b5 - matrix.m[6] * b4 + matrix.m[7] * b3) / det;
            result.m[1] = -(matrix.m[1] * b5 - matrix.m[2] * b4 + matrix.m[3] * b3) / det;
            result.m[2] = (matrix.m[13] * a5 - matrix.m[14] * a4 + matrix.m[15] * a3) / det;
            result.m[3] = -(matrix.m[9] * a5 - matrix.m[10] * a4 + matrix.m[11] * a3) / det;

            result.m[4] = -(matrix.m[4] * b5 - matrix.m[6] * b2 + matrix.m[7] * b1) / det;
            result.m[5] = (matrix.m[0] * b5 - matrix.m[2] * b2 + matrix.m[3] * b1) / det;
            result.m[6] = -(matrix.m[12] * a5 - matrix.m[14] * a2 + matrix.m[15] * a1) / det;
            result.m[7] = (matrix.m[8] * a5 - matrix.m[10] * a2 + matrix.m[11] * a1) / det;

            result.m[8] = (matrix.m[4] * b4 - matrix.m[5] * b2 + matrix.m[7] * b0) / det;
            result.m[9] = -(matrix.m[0] * b4 - matrix.m[1] * b2 + matrix.m[3] * b0) / det;
            result.m[10] = (matrix.m[12] * a4 - matrix.m[13] * a2 + matrix.m[15] * a0) / det;
            result.m[11] = -(matrix.m[8] * a4 - matrix.m[9] * a2 + matrix.m[11] * a0) / det;

            result.m[12] = -(matrix.m[4] * b3 - matrix.m[5] * b1 + matrix.m[6] * b0) / det;
            result.m[13] = (matrix.m[0] * b3 - matrix.m[1] * b1 + matrix.m[2] * b0) / det;
            result.m[14] = -(matrix.m[12] * a3 - matrix.m[13] * a1 + matrix.m[14] * a0) / det;
            result.m[15] = (matrix.m[8] * a3 - matrix.m[9] * a1 + matrix.m[10] * a0) / det;
        }

        return result;
    }
}

#include "MatrixNeon.hpp"
#include "MatrixSse.hpp"

#endif
