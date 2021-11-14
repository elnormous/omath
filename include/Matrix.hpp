//
// elnormous/omath
//

#ifndef OMATH_MATRIX_HPP
#define OMATH_MATRIX_HPP

#include <type_traits>
#include <utility>
#include "Vector.hpp"

namespace omath
{
    template <typename T, std::size_t rows, std::size_t cols = rows>
    struct MatrixElements final
    {
        T v[rows * cols];
    };

    template <typename T, std::size_t rows, std::size_t cols, std::size_t ...i>
    constexpr auto transpose(const MatrixElements<T, cols, rows> a,
                             const std::index_sequence<i...>) noexcept
    {
        return MatrixElements<T, cols, rows>{a.v[((i % rows) * cols + i / rows)]...};
    }

    template <typename T, std::size_t size, std::size_t ...i>
    constexpr auto identity(std::index_sequence<i...>) noexcept
    {
        return MatrixElements<T, size, size>{(i % size == i / size) ? T(1) : T(0)...};
    }

    template <typename T, std::size_t rows, std::size_t cols = rows>
    class Matrix final
    {
    public:
#ifndef OMATH_DISABLE_SIMD
#  if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP >= 1 || defined(__ARM_NEON__)
        alignas(std::is_same_v<T, float> && rows == 4 && cols == 4 ? cols * sizeof(T) : alignof(T))
#  endif
#  if (defined(__SSE2__) || defined(_M_X64) || _M_IX86_FP >= 2) || (defined(__ARM_NEON__) && defined(__aarch64__))
        alignas(std::is_same_v<T, double> && rows == 4 && cols == 4 ? cols * sizeof(T) : alignof(T))
#  endif
#endif // OMATH_DISABLE_SIMD
        MatrixElements<T, cols, rows> m; // column-major matrix

        constexpr Matrix() noexcept = default;

        template <typename ...A>
        explicit constexpr Matrix(const A... args) noexcept:
            m{transpose(MatrixElements<T, cols, rows>{args...}, std::make_index_sequence<rows * cols>{})}
        {
        }

        [[nodiscard]] auto& operator()(const std::size_t row, const std::size_t col) noexcept { return m.v[col * rows + row]; }
        [[nodiscard]] constexpr auto operator()(const std::size_t row, const std::size_t col) const noexcept { return m.v[col * rows + row]; }
    };

    template <typename T, std::size_t size>
    constexpr auto identityMatrix = Matrix<T, size, size>{identity<T, size>(std::make_index_sequence<size * size>{})};

    template <typename T, std::size_t size>
    constexpr void setIdentity(Matrix<T, size, size>& matrix) noexcept
    {
        for (std::size_t i = 0; i < size; ++i)
            for (std::size_t j = 0; j < size; ++j)
                matrix.m.v[j * size + i] = (j == i) ? T(1) : T(0);
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator==(const Matrix<T, rows, cols>& matrix1,
                                            const Matrix<T, rows, cols>& matrix2) noexcept
    {
        for (std::size_t i = 0; i < rows * cols; ++i)
            if (matrix1.m.v[i] != matrix2.m.v[i]) return false;
        return true;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator!=(const Matrix<T, rows, cols>& matrix1,
                                            const Matrix<T, rows, cols>& matrix2) noexcept
    {
        for (std::size_t i = 0; i < rows * cols; ++i)
            if (matrix1.m.v[i] != matrix2.m.v[i]) return true;
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
        for (std::size_t i = 0; i < rows * cols; ++i) result.m.v[i] = -matrix.m.v[i];
        return result;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    constexpr void negate(Matrix<T, rows, cols>& matrix) noexcept
    {
        for (auto& c : matrix.m.v) c = -c;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator+(const Matrix<T, rows, cols>& matrix1,
                                           const Matrix<T, rows, cols>& matrix2) noexcept
    {
        Matrix<T, rows, cols> result;
        for (std::size_t i = 0; i < rows * cols; ++i)
            result.m.v[i] = matrix1.m.v[i] + matrix2.m.v[i];
        return result;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    auto& operator+=(Matrix<T, rows, cols>& matrix1,
                     const Matrix<T, rows, cols>& matrix2) noexcept
    {
        for (std::size_t i = 0; i < rows * cols; ++i)
            matrix1.m.v[i] += matrix2.m.v[i];
        return matrix1;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator-(const Matrix<T, rows, cols>& matrix1,
                                           const Matrix<T, rows, cols>& matrix2) noexcept
    {
        Matrix<T, rows, cols> result;
        for (std::size_t i = 0; i < rows * cols; ++i)
            result.m.v[i] = matrix1.m.v[i] - matrix2.m.v[i];
        return result;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    auto& operator-=(Matrix<T, rows, cols>& matrix1,
                     const Matrix<T, rows, cols>& matrix2) noexcept
    {
        for (std::size_t i = 0; i < rows * cols; ++i)
            matrix1.m.v[i] -= matrix2.m.v[i];
        return matrix1;
    }

    template <typename T, std::size_t rows, std::size_t cols, std::size_t cols2>
    [[nodiscard]] constexpr auto operator*(const Matrix<T, rows, cols>& matrix1,
                                           const Matrix<T, cols, cols2>& matrix2) noexcept
    {
        Matrix<T, rows, cols2> result{};

        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols2; ++j)
                for (std::size_t k = 0; k < cols; ++k)
                    result.m.v[j * rows + i] += matrix1.m.v[k * rows + i] * matrix2.m.v[j * cols + k];

        return result;
    }

    template <typename T, std::size_t size>
    auto& operator*=(Matrix<T, size, size>& matrix1,
                     const Matrix<T, size, size>& matrix2) noexcept
    {
        Matrix<T, size, size> result{};

        for (std::size_t i = 0; i < size; ++i)
            for (std::size_t j = 0; j < size; ++j)
                for (std::size_t k = 0; k < size; ++k)
                    result.m.v[j * size + i] += matrix1.m.v[k * size + i] * matrix2.m.v[j * size + k];

        matrix1 = std::move(result);
        return matrix1;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator*(const Matrix<T, rows, cols>& matrix,
                                           const T scalar) noexcept
    {
        Matrix<T, rows, cols> result;
        for (std::size_t i = 0; i < rows * cols; ++i)
            result.m.v[i] = matrix.m.v[i] * scalar;
        return result;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    auto& operator*=(Matrix<T, rows, cols>& matrix,
                     const T scalar) noexcept
    {
        for (std::size_t i = 0; i < rows * cols; ++i)
            matrix.m.v[i] *= scalar;
        return matrix;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto operator/(const Matrix<T, rows, cols>& matrix,
                                           const T scalar) noexcept
    {
        Matrix<T, rows, cols> result;
        for (std::size_t i = 0; i < rows * cols; ++i)
            result.m.v[i] = matrix.m.v[i] / scalar;
        return result;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    auto& operator/=(Matrix<T, rows, cols>& matrix,
                     const T scalar) noexcept
    {
        for (std::size_t i = 0; i < rows * cols; ++i)
            matrix.m.v[i] /= scalar;
        return matrix;
    }

    template <
        typename T,
        std::size_t size,
        std::size_t dims,
        std::enable_if_t<(dims <= size)>* = nullptr
    >
    [[nodiscard]] auto operator*(const Matrix<T, size, size>& matrix,
                                 const Vector<T, dims>& vector) noexcept
    {
        Vector<T, dims> result{};

        for (std::size_t i = 0; i < dims; ++i)
            for (std::size_t j = 0; j < dims; ++j)
                result.v[i] += matrix.m.v[j * size + i] * vector.v[j];

        return result;
    }

    template <
        typename T,
        std::size_t size,
        std::size_t dims,
        std::enable_if_t<(dims <= size)>* = nullptr
    >
    void transformVector(const Matrix<T, size, size>& matrix,
                         Vector<T, dims>& vector) noexcept
    {
        Vector<T, dims> result{};

        for (std::size_t i = 0; i < dims; ++i)
            for (std::size_t j = 0; j < dims; ++j)
                result[i] += matrix.m.v[j * size + i] * vector.v[j];

        vector = std::move(result);
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] auto operator*(const T scalar,
                                 const Matrix<T, rows, cols>& mat) noexcept
    {
        return mat * scalar;
    }

    template <
        typename T,
        std::size_t dims,
        std::size_t size,
        std::enable_if_t<(dims <= size)>* = nullptr
    >
    [[nodiscard]] auto operator*(const Vector<T, dims>& vector,
                                 const Matrix<T, size, size>& matrix) noexcept
    {
        Vector<T, dims> result{};

        for (std::size_t i = 0; i < dims; ++i)
            for (std::size_t j = 0; j < dims; ++j)
                result.v[i] += vector.v[j] * matrix.m.v[i * size + j];

        return result;
    }

    template <
        typename T,
        std::size_t dims,
        std::size_t size,
        std::enable_if_t<(dims <= size)>* = nullptr
    >
    auto& operator*=(Vector<T, dims>& vector,
                     const Matrix<T, size, size>& matrix) noexcept
    {
        static_assert(dims <= size);
        Vector<T, dims> result{};

        for (std::size_t i = 0; i < dims; ++i)
            for (std::size_t j = 0; j < dims; ++j)
                result[i] += vector[j] * matrix.m.v[i * size + j];

        vector = std::move(result);
        return vector;
    }

    template <typename T, std::size_t rows, std::size_t cols>
    [[nodiscard]] constexpr auto transposed(const Matrix<T, rows, cols>& matrix) noexcept
    {
        Matrix<T, cols, rows> result;
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; ++j)
                result.m.v[i * cols + j] = matrix.m.v[j * rows + i];
        return result;
    }

    template <typename T, std::size_t size>
    void transpose(Matrix<T, size, size>& matrix) noexcept
    {
        for (std::size_t i = 1; i < size; ++i)
            for (std::size_t j = 0; j < i; ++j)
            {
                T temp = std::move(matrix.m.v[i * size + j]);
                matrix.m.v[i * size + j] = std::move(matrix.m.v[j * size + i]);
                matrix.m.v[j * size + i] = std::move(temp);
            }
    }

    template <typename T, std::size_t size, std::enable_if<(size <= 4)>* = nullptr>
    [[nodiscard]] constexpr auto determinant(const Matrix<T, size, size>& matrix) noexcept
    {
        if constexpr (size == 0)
            return T(1);
        if constexpr (size == 1)
            return matrix.m.v[0];
        else if constexpr (size == 2)
            return matrix.m.v[0] * matrix.m.v[3] - matrix.m.v[1] * matrix.m.v[2];
        else if constexpr (size == 3)
            return matrix.m.v[0] * matrix.m.v[4] * matrix.m.v[8] +
                matrix.m.v[1] * matrix.m.v[5] * matrix.m.v[6] +
                matrix.m.v[2] * matrix.m.v[3] * matrix.m.v[7] -
                matrix.m.v[2] * matrix.m.v[4] * matrix.m.v[6] -
                matrix.m.v[1] * matrix.m.v[3] * matrix.m.v[8] -
                matrix.m.v[0] * matrix.m.v[5] * matrix.m.v[7];
        else if constexpr (size == 4)
        {
            const auto a0 = matrix.m.v[0] * matrix.m.v[5] - matrix.m.v[1] * matrix.m.v[4];
            const auto a1 = matrix.m.v[0] * matrix.m.v[6] - matrix.m.v[2] * matrix.m.v[4];
            const auto a2 = matrix.m.v[0] * matrix.m.v[7] - matrix.m.v[3] * matrix.m.v[4];
            const auto a3 = matrix.m.v[1] * matrix.m.v[6] - matrix.m.v[2] * matrix.m.v[5];
            const auto a4 = matrix.m.v[1] * matrix.m.v[7] - matrix.m.v[3] * matrix.m.v[5];
            const auto a5 = matrix.m.v[2] * matrix.m.v[7] - matrix.m.v[3] * matrix.m.v[6];
            const auto b0 = matrix.m.v[8] * matrix.m.v[13] - matrix.m.v[9] * matrix.m.v[12];
            const auto b1 = matrix.m.v[8] * matrix.m.v[14] - matrix.m.v[10] * matrix.m.v[12];
            const auto b2 = matrix.m.v[8] * matrix.m.v[15] - matrix.m.v[11] * matrix.m.v[12];
            const auto b3 = matrix.m.v[9] * matrix.m.v[14] - matrix.m.v[10] * matrix.m.v[13];
            const auto b4 = matrix.m.v[9] * matrix.m.v[15] - matrix.m.v[11] * matrix.m.v[13];
            const auto b5 = matrix.m.v[10] * matrix.m.v[15] - matrix.m.v[11] * matrix.m.v[14];

            return a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0;
        }
    }

    template <typename T, std::size_t size>
    void invert(Matrix<T, size, size>& matrix) noexcept
    {
        static_assert(size <= 4);

        if constexpr (size == 1)
            matrix.m.v[0] = 1.0F / matrix.m.v[0];
        else if constexpr (size == 2)
        {
            const auto det = matrix.m.v[0] * matrix.m.v[3] - matrix.m.v[1] * matrix.m.v[2];
            const T adjugate[size * size]{
                matrix.m.v[3],
                -matrix.m.v[1],
                -matrix.m.v[2],
                matrix.m.v[0]
            };

            matrix.m.v[0] = adjugate[0] / det;
            matrix.m.v[1] = adjugate[1] / det;
            matrix.m.v[2] = adjugate[2] / det;
            matrix.m.v[3] = adjugate[3] / det;
        }
        else if constexpr (size == 3)
        {
            const auto a0 = matrix.m.v[4] * matrix.m.v[8] - matrix.m.v[5] * matrix.m.v[7];
            const auto a1 = matrix.m.v[3] * matrix.m.v[8] - matrix.m.v[5] * matrix.m.v[6];
            const auto a2 = matrix.m.v[3] * matrix.m.v[7] - matrix.m.v[4] * matrix.m.v[6];

            const auto det = matrix.m.v[0] * a0 - matrix.m.v[1] * a1 + matrix.m.v[2] * a2;

            const T adjugate[size * size]{
                a0,
                -matrix.m.v[1] * matrix.m.v[8] + matrix.m.v[2] * matrix.m.v[7],
                matrix.m.v[1] * matrix.m.v[5] - matrix.m.v[2] * matrix.m.v[4],

                -a1,
                matrix.m.v[0] * matrix.m.v[8] - matrix.m.v[2] * matrix.m.v[6],
                -matrix.m.v[0] * matrix.m.v[5] + matrix.m.v[2] * matrix.m.v[3],

                a2,
                -matrix.m.v[0] * matrix.m.v[7] + matrix.m.v[1] * matrix.m.v[6],
                matrix.m.v[0] * matrix.m.v[4] - matrix.m.v[1] * matrix.m.v[3]
            };

            matrix.m.v[0] = adjugate[0] / det;
            matrix.m.v[1] = adjugate[1] / det;
            matrix.m.v[2] = adjugate[2] / det;
            matrix.m.v[3] = adjugate[3] / det;
            matrix.m.v[4] = adjugate[4] / det;
            matrix.m.v[5] = adjugate[5] / det;
            matrix.m.v[6] = adjugate[6] / det;
            matrix.m.v[7] = adjugate[7] / det;
            matrix.m.v[8] = adjugate[8] / det;
        }
        else if constexpr (size == 4)
        {
            const auto a0 = matrix.m.v[0] * matrix.m.v[5] - matrix.m.v[1] * matrix.m.v[4];
            const auto a1 = matrix.m.v[0] * matrix.m.v[6] - matrix.m.v[2] * matrix.m.v[4];
            const auto a2 = matrix.m.v[0] * matrix.m.v[7] - matrix.m.v[3] * matrix.m.v[4];
            const auto a3 = matrix.m.v[1] * matrix.m.v[6] - matrix.m.v[2] * matrix.m.v[5];
            const auto a4 = matrix.m.v[1] * matrix.m.v[7] - matrix.m.v[3] * matrix.m.v[5];
            const auto a5 = matrix.m.v[2] * matrix.m.v[7] - matrix.m.v[3] * matrix.m.v[6];
            const auto b0 = matrix.m.v[8] * matrix.m.v[13] - matrix.m.v[9] * matrix.m.v[12];
            const auto b1 = matrix.m.v[8] * matrix.m.v[14] - matrix.m.v[10] * matrix.m.v[12];
            const auto b2 = matrix.m.v[8] * matrix.m.v[15] - matrix.m.v[11] * matrix.m.v[12];
            const auto b3 = matrix.m.v[9] * matrix.m.v[14] - matrix.m.v[10] * matrix.m.v[13];
            const auto b4 = matrix.m.v[9] * matrix.m.v[15] - matrix.m.v[11] * matrix.m.v[13];
            const auto b5 = matrix.m.v[10] * matrix.m.v[15] - matrix.m.v[11] * matrix.m.v[14];

            const auto det = a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0;

            const T adjugate[size * size]{
                matrix.m.v[5] * b5 - matrix.m.v[6] * b4 + matrix.m.v[7] * b3,
                -(matrix.m.v[1] * b5 - matrix.m.v[2] * b4 + matrix.m.v[3] * b3),
                matrix.m.v[13] * a5 - matrix.m.v[14] * a4 + matrix.m.v[15] * a3,
                -(matrix.m.v[9] * a5 - matrix.m.v[10] * a4 + matrix.m.v[11] * a3),

                -(matrix.m.v[4] * b5 - matrix.m.v[6] * b2 + matrix.m.v[7] * b1),
                matrix.m.v[0] * b5 - matrix.m.v[2] * b2 + matrix.m.v[3] * b1,
                -(matrix.m.v[12] * a5 - matrix.m.v[14] * a2 + matrix.m.v[15] * a1),
                matrix.m.v[8] * a5 - matrix.m.v[10] * a2 + matrix.m.v[11] * a1,

                matrix.m.v[4] * b4 - matrix.m.v[5] * b2 + matrix.m.v[7] * b0,
                -(matrix.m.v[0] * b4 - matrix.m.v[1] * b2 + matrix.m.v[3] * b0),
                matrix.m.v[12] * a4 - matrix.m.v[13] * a2 + matrix.m.v[15] * a0,
                -(matrix.m.v[8] * a4 - matrix.m.v[9] * a2 + matrix.m.v[11] * a0),

                -(matrix.m.v[4] * b3 - matrix.m.v[5] * b1 + matrix.m.v[6] * b0),
                matrix.m.v[0] * b3 - matrix.m.v[1] * b1 + matrix.m.v[2] * b0,
                -(matrix.m.v[12] * a3 - matrix.m.v[13] * a1 + matrix.m.v[14] * a0),
                matrix.m.v[8] * a3 - matrix.m.v[9] * a1 + matrix.m.v[10] * a0
            };

            matrix.m.v[0] = adjugate[0] / det;
            matrix.m.v[1] = adjugate[1] / det;
            matrix.m.v[2] = adjugate[2] / det;
            matrix.m.v[3] = adjugate[3] / det;
            matrix.m.v[4] = adjugate[4] / det;
            matrix.m.v[5] = adjugate[5] / det;
            matrix.m.v[6] = adjugate[6] / det;
            matrix.m.v[7] = adjugate[7] / det;
            matrix.m.v[8] = adjugate[8] / det;
            matrix.m.v[9] = adjugate[9] / det;
            matrix.m.v[10] = adjugate[10] / det;
            matrix.m.v[11] = adjugate[11] / det;
            matrix.m.v[12] = adjugate[12] / det;
            matrix.m.v[13] = adjugate[13] / det;
            matrix.m.v[14] = adjugate[14] / det;
            matrix.m.v[15] = adjugate[15] / det;
        }
    }

    template <typename T, std::size_t size>
    [[nodiscard]] constexpr auto inverse(const Matrix<T, size, size>& matrix) noexcept
    {
        static_assert(size <= 4);

        Matrix<T, size, size> result;

        if constexpr (size == 1)
            result.m.v[0] = 1.0F / matrix.m.v[0];
        else if constexpr (size == 2)
        {
            const auto det = matrix.m.v[0] * matrix.m.v[3] - matrix.m.v[1] * matrix.m.v[2];
            result.m.v[0] = matrix.m.v[3] / det;
            result.m.v[1] = -matrix.m.v[1] / det;
            result.m.v[2] = -matrix.m.v[2] / det;
            result.m.v[3] = matrix.m.v[0] / det;
        }
        else if constexpr (size == 3)
        {
            const auto a0 = matrix.m.v[4] * matrix.m.v[8] - matrix.m.v[5] * matrix.m.v[7];
            const auto a1 = matrix.m.v[3] * matrix.m.v[8] - matrix.m.v[5] * matrix.m.v[6];
            const auto a2 = matrix.m.v[3] * matrix.m.v[7] - matrix.m.v[4] * matrix.m.v[6];

            const auto det = matrix.m.v[0] * a0 - matrix.m.v[1] * a1 + matrix.m.v[2] * a2;

            result.m.v[0] = a0 / det;
            result.m.v[1] = -(matrix.m.v[1] * matrix.m.v[8] - matrix.m.v[2] * matrix.m.v[7]) / det;
            result.m.v[2] = (matrix.m.v[1] * matrix.m.v[5] - matrix.m.v[2] * matrix.m.v[4]) / det;

            result.m.v[3] = -a1 / det;
            result.m.v[4] = (matrix.m.v[0] * matrix.m.v[8] - matrix.m.v[2] * matrix.m.v[6]) / det;
            result.m.v[5] = -(matrix.m.v[0] * matrix.m.v[5] - matrix.m.v[2] * matrix.m.v[3]) / det;

            result.m.v[6] = a2 / det;
            result.m.v[7] = -(matrix.m.v[0] * matrix.m.v[7] - matrix.m.v[1] * matrix.m.v[6]) / det;
            result.m.v[8] = (matrix.m.v[0] * matrix.m.v[4] - matrix.m.v[1] * matrix.m.v[3]) / det;
        }
        else if constexpr (size == 4)
        {
            const auto a0 = matrix.m.v[0] * matrix.m.v[5] - matrix.m.v[1] * matrix.m.v[4];
            const auto a1 = matrix.m.v[0] * matrix.m.v[6] - matrix.m.v[2] * matrix.m.v[4];
            const auto a2 = matrix.m.v[0] * matrix.m.v[7] - matrix.m.v[3] * matrix.m.v[4];
            const auto a3 = matrix.m.v[1] * matrix.m.v[6] - matrix.m.v[2] * matrix.m.v[5];
            const auto a4 = matrix.m.v[1] * matrix.m.v[7] - matrix.m.v[3] * matrix.m.v[5];
            const auto a5 = matrix.m.v[2] * matrix.m.v[7] - matrix.m.v[3] * matrix.m.v[6];
            const auto b0 = matrix.m.v[8] * matrix.m.v[13] - matrix.m.v[9] * matrix.m.v[12];
            const auto b1 = matrix.m.v[8] * matrix.m.v[14] - matrix.m.v[10] * matrix.m.v[12];
            const auto b2 = matrix.m.v[8] * matrix.m.v[15] - matrix.m.v[11] * matrix.m.v[12];
            const auto b3 = matrix.m.v[9] * matrix.m.v[14] - matrix.m.v[10] * matrix.m.v[13];
            const auto b4 = matrix.m.v[9] * matrix.m.v[15] - matrix.m.v[11] * matrix.m.v[13];
            const auto b5 = matrix.m.v[10] * matrix.m.v[15] - matrix.m.v[11] * matrix.m.v[14];

            const auto det = a0 * b5 - a1 * b4 + a2 * b3 + a3 * b2 - a4 * b1 + a5 * b0;

            result.m.v[0] = (matrix.m.v[5] * b5 - matrix.m.v[6] * b4 + matrix.m.v[7] * b3) / det;
            result.m.v[1] = -(matrix.m.v[1] * b5 - matrix.m.v[2] * b4 + matrix.m.v[3] * b3) / det;
            result.m.v[2] = (matrix.m.v[13] * a5 - matrix.m.v[14] * a4 + matrix.m.v[15] * a3) / det;
            result.m.v[3] = -(matrix.m.v[9] * a5 - matrix.m.v[10] * a4 + matrix.m.v[11] * a3) / det;

            result.m.v[4] = -(matrix.m.v[4] * b5 - matrix.m.v[6] * b2 + matrix.m.v[7] * b1) / det;
            result.m.v[5] = (matrix.m.v[0] * b5 - matrix.m.v[2] * b2 + matrix.m.v[3] * b1) / det;
            result.m.v[6] = -(matrix.m.v[12] * a5 - matrix.m.v[14] * a2 + matrix.m.v[15] * a1) / det;
            result.m.v[7] = (matrix.m.v[8] * a5 - matrix.m.v[10] * a2 + matrix.m.v[11] * a1) / det;

            result.m.v[8] = (matrix.m.v[4] * b4 - matrix.m.v[5] * b2 + matrix.m.v[7] * b0) / det;
            result.m.v[9] = -(matrix.m.v[0] * b4 - matrix.m.v[1] * b2 + matrix.m.v[3] * b0) / det;
            result.m.v[10] = (matrix.m.v[12] * a4 - matrix.m.v[13] * a2 + matrix.m.v[15] * a0) / det;
            result.m.v[11] = -(matrix.m.v[8] * a4 - matrix.m.v[9] * a2 + matrix.m.v[11] * a0) / det;

            result.m.v[12] = -(matrix.m.v[4] * b3 - matrix.m.v[5] * b1 + matrix.m.v[6] * b0) / det;
            result.m.v[13] = (matrix.m.v[0] * b3 - matrix.m.v[1] * b1 + matrix.m.v[2] * b0) / det;
            result.m.v[14] = -(matrix.m.v[12] * a3 - matrix.m.v[13] * a1 + matrix.m.v[14] * a0) / det;
            result.m.v[15] = (matrix.m.v[8] * a3 - matrix.m.v[9] * a1 + matrix.m.v[10] * a0) / det;
        }

        return result;
    }
}

#include "MatrixAvx.hpp"
#include "MatrixNeon.hpp"
#include "MatrixSse.hpp"

#endif // OMATH_MATRIX_HPP
