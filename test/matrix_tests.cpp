#include "catch2/catch.hpp"
#include "Matrix.hpp"

TEST_CASE("Matrix zero initalization", "matrix")
{
    const omath::Matrix<float, 4> matrix{};
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            REQUIRE(matrix[row][column] == 0.0F);
}

TEST_CASE("Matrix value initalization", "matrix")
{
    const omath::Matrix<float, 2, 2> matrix{
        0.0F, 1.0F,
        2.0F, 3.0F
    };

    REQUIRE(matrix[0][0] == 0.0F);
    REQUIRE(matrix[0][1] == 1.0F);
    REQUIRE(matrix[1][0] == 2.0F);
    REQUIRE(matrix[1][1] == 3.0F);
}

TEST_CASE("Matrix identity", "matrix")
{
    const auto matrix = omath::Matrix<float, 2>::identity();

    REQUIRE(matrix[0][0] == 1.0F);
    REQUIRE(matrix[0][1] == 0.0F);
    REQUIRE(matrix[1][0] == 0.0F);
    REQUIRE(matrix[1][1] == 1.0F);
}

TEST_CASE("Matrix element setter", "matrix")
{
    omath::Matrix<float, 2> matrix;

    matrix[0][0] = 1.0F;
    matrix[0][1] = 2.0F;
    matrix[1][0] = 3.0F;
    matrix[1][1] = 4.0F;

    REQUIRE(matrix[0][0] == 1.0F);
    REQUIRE(matrix[0][1] == 2.0F);
    REQUIRE(matrix[1][0] == 3.0F);
    REQUIRE(matrix[1][1] == 4.0F);
}

TEST_CASE("Matrix comparison operators", "vector")
{
    const omath::Matrix<float, 2> matrix1{2.0F, 4.0F, 3.0F, 5.0F};
    const omath::Matrix<float, 2> matrix2{2.0F, 5.0F, 3.0F, 5.0F};
    const omath::Matrix<float, 2> matrix3{2.0F, 4.0F, 3.0F, 5.0F};

    SECTION("Equal")
    {
        REQUIRE_FALSE(matrix1 == matrix2);
        REQUIRE(matrix1 == matrix3);
    }

    SECTION("Not equal")
    {
        REQUIRE(matrix1 != matrix2);
        REQUIRE_FALSE(matrix1 != matrix3);
    }
}

TEST_CASE("1x1 matrix transpose", "matrix")
{
    omath::Matrix<float, 1> matrix{1.0F};
    matrix.transpose();

    REQUIRE(matrix[0][0] == 1.0F);
}

TEST_CASE("2x2 matrix transpose", "matrix")
{
    omath::Matrix<float, 2> matrix{
        1.0F, 2.0F,
        3.0F, 4.0F
    };
    matrix.transpose();

    REQUIRE(matrix[0][0] == 1.0F);
    REQUIRE(matrix[0][1] == 3.0F);
    REQUIRE(matrix[1][0] == 2.0F);
    REQUIRE(matrix[1][1] == 4.0F);
}

TEST_CASE("3x3 matrix transpose", "matrix")
{
    omath::Matrix<float, 3> matrix{
        1.0F, 2.0F, 3.0F,
        4.0F, 6.0F, 6.0F,
        7.0F, 8.0F, 9.0F
    };
    matrix.transpose();

    REQUIRE(matrix[0][0] == 1.0F);
    REQUIRE(matrix[0][1] == 4.0F);
    REQUIRE(matrix[0][2] == 7.0F);
    REQUIRE(matrix[1][0] == 2.0F);
    REQUIRE(matrix[1][1] == 6.0F);
    REQUIRE(matrix[1][2] == 8.0F);
    REQUIRE(matrix[2][0] == 3.0F);
    REQUIRE(matrix[2][1] == 6.0F);
    REQUIRE(matrix[2][2] == 9.0F);
}

TEST_CASE("4x4 matrix transpose", "matrix")
{
    omath::Matrix<float, 4, 4, false> matrix{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        9.0F, 10.0F, 11.0F, 12.0F,
        13.0F, 14.0F, 15.0F, 16.0F
    };
    matrix.transpose();

    REQUIRE(matrix[0][0] == 1.0F);
    REQUIRE(matrix[0][1] == 5.0F);
    REQUIRE(matrix[0][2] == 9.0F);
    REQUIRE(matrix[0][3] == 13.0F);
    REQUIRE(matrix[1][0] == 2.0F);
    REQUIRE(matrix[1][1] == 6.0F);
    REQUIRE(matrix[1][2] == 10.0F);
    REQUIRE(matrix[1][3] == 14.0F);
    REQUIRE(matrix[2][0] == 3.0F);
    REQUIRE(matrix[2][1] == 7.0F);
    REQUIRE(matrix[2][2] == 11.0F);
    REQUIRE(matrix[2][3] == 15.0F);
    REQUIRE(matrix[3][0] == 4.0F);
    REQUIRE(matrix[3][1] == 8.0F);
    REQUIRE(matrix[3][2] == 12.0F);
    REQUIRE(matrix[3][3] == 16.0F);
}

TEST_CASE("4x4 matrix transpose using SIMD", "matrix")
{
    omath::Matrix<float, 4, 4, true> matrix{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        9.0F, 10.0F, 11.0F, 12.0F,
        13.0F, 14.0F, 15.0F, 16.0F
    };
    matrix.transpose();

    REQUIRE(matrix[0][0] == 1.0F);
    REQUIRE(matrix[0][1] == 5.0F);
    REQUIRE(matrix[0][2] == 9.0F);
    REQUIRE(matrix[0][3] == 13.0F);
    REQUIRE(matrix[1][0] == 2.0F);
    REQUIRE(matrix[1][1] == 6.0F);
    REQUIRE(matrix[1][2] == 10.0F);
    REQUIRE(matrix[1][3] == 14.0F);
    REQUIRE(matrix[2][0] == 3.0F);
    REQUIRE(matrix[2][1] == 7.0F);
    REQUIRE(matrix[2][2] == 11.0F);
    REQUIRE(matrix[2][3] == 15.0F);
    REQUIRE(matrix[3][0] == 4.0F);
    REQUIRE(matrix[3][1] == 8.0F);
    REQUIRE(matrix[3][2] == 12.0F);
    REQUIRE(matrix[3][3] == 16.0F);
}

TEST_CASE("2x2 matrix comparison", "matrix")
{
    const omath::Matrix<float, 2> matrix1{
        0.0F, 1.0F,
        2.0F, 3.0F
    };

    const omath::Matrix<float, 2> matrix2{
        0.0F, 1.0F,
        2.0F, 3.0F
    };

    const omath::Matrix<float, 2> matrix3{
        1.0F, 2.0F,
        3.0F, 4.0F
    };

    REQUIRE(matrix1 == matrix2);
    REQUIRE(matrix1 != matrix3);
}

TEST_CASE("2x2 matrix negation", "matrix")
{
    const omath::Matrix<float, 2> matrix{
        0.0F, 1.0F,
        2.0F, -3.0F
    };

    const auto result = -matrix;

    REQUIRE(result[0][0] == 0.0F);
    REQUIRE(result[0][1] == -1.0F);
    REQUIRE(result[1][0] == -2.0F);
    REQUIRE(result[1][1] == 3.0F);
}

TEST_CASE("4x4 matrix negation", "matrix")
{
    const omath::Matrix<float, 4, 4, false> matrix{
        0.0F, 1.0F, 2.0F, 3.0F,
        2.0F, -3.0F, 4.0F, 5.0F,
        3.0F, 4.0F, 5.0F, 6.0F,
        4.0F, 5.0F, 6.0F, 7.0F
    };

    const auto result = -matrix;

    REQUIRE(result[0][0] == 0.0F);
    REQUIRE(result[0][1] == -1.0F);
    REQUIRE(result[0][2] == -2.0F);
    REQUIRE(result[0][3] == -3.0F);

    REQUIRE(result[1][0] == -2.0F);
    REQUIRE(result[1][1] == 3.0F);
    REQUIRE(result[1][2] == -4.0F);
    REQUIRE(result[1][3] == -5.0F);

    REQUIRE(result[2][0] == -3.0F);
    REQUIRE(result[2][1] == -4.0F);
    REQUIRE(result[2][2] == -5.0F);
    REQUIRE(result[2][3] == -6.0F);

    REQUIRE(result[3][0] == -4.0F);
    REQUIRE(result[3][1] == -5.0F);
    REQUIRE(result[3][2] == -6.0F);
    REQUIRE(result[3][3] == -7.0F);
}

TEST_CASE("4x4 matrix negation using SIMD", "matrix")
{
    const omath::Matrix<float, 4, 4, true> matrix{
        0.0F, 1.0F, 2.0F, 3.0F,
        2.0F, -3.0F, 4.0F, 5.0F,
        3.0F, 4.0F, 5.0F, 6.0F,
        4.0F, 5.0F, 6.0F, 7.0F
    };

    const auto result = -matrix;

    REQUIRE(result[0][0] == 0.0F);
    REQUIRE(result[0][1] == -1.0F);
    REQUIRE(result[0][2] == -2.0F);
    REQUIRE(result[0][3] == -3.0F);

    REQUIRE(result[1][0] == -2.0F);
    REQUIRE(result[1][1] == 3.0F);
    REQUIRE(result[1][2] == -4.0F);
    REQUIRE(result[1][3] == -5.0F);

    REQUIRE(result[2][0] == -3.0F);
    REQUIRE(result[2][1] == -4.0F);
    REQUIRE(result[2][2] == -5.0F);
    REQUIRE(result[2][3] == -6.0F);

    REQUIRE(result[3][0] == -4.0F);
    REQUIRE(result[3][1] == -5.0F);
    REQUIRE(result[3][2] == -6.0F);
    REQUIRE(result[3][3] == -7.0F);
}

TEST_CASE("2x2 matrix sum", "matrix")
{
    const omath::Matrix<float, 2> matrix1{
        0.0F, 1.0F,
        2.0F, -3.0F
    };

    const omath::Matrix<float, 2> matrix2{
        5.0F, -6.0F,
        7.0F, 8.0F
    };

    const auto result = matrix1 + matrix2;

    REQUIRE(result[0][0] == 5.0F);
    REQUIRE(result[0][1] == -5.0F);
    REQUIRE(result[1][0] == 9.0F);
    REQUIRE(result[1][1] == 5.0F);
}

TEST_CASE("4x4 matrix sum", "matrix")
{
    const omath::Matrix<float, 4, 4, false> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4, false> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    const auto result = matrix1 + matrix2;

    REQUIRE(result[0][0] == 5.0F);
    REQUIRE(result[0][1] == -5.0F);
    REQUIRE(result[0][2] == 5.0F);
    REQUIRE(result[0][3] == -5.0F);

    REQUIRE(result[1][0] == 9.0F);
    REQUIRE(result[1][1] == 5.0F);
    REQUIRE(result[1][2] == 9.0F);
    REQUIRE(result[1][3] == 5.0F);

    REQUIRE(result[2][0] == 5.0F);
    REQUIRE(result[2][1] == -5.0F);
    REQUIRE(result[2][2] == 5.0F);
    REQUIRE(result[2][3] == -5.0F);

    REQUIRE(result[3][0] == 9.0F);
    REQUIRE(result[3][1] == 5.0F);
    REQUIRE(result[3][2] == 9.0F);
    REQUIRE(result[3][3] == 5.0F);
}

TEST_CASE("4x4 matrix sum using SIMD", "matrix")
{
    const omath::Matrix<float, 4, 4, true> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4, true> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    const auto result = matrix1 + matrix2;

    REQUIRE(result[0][0] == 5.0F);
    REQUIRE(result[0][1] == -5.0F);
    REQUIRE(result[0][2] == 5.0F);
    REQUIRE(result[0][3] == -5.0F);

    REQUIRE(result[1][0] == 9.0F);
    REQUIRE(result[1][1] == 5.0F);
    REQUIRE(result[1][2] == 9.0F);
    REQUIRE(result[1][3] == 5.0F);

    REQUIRE(result[2][0] == 5.0F);
    REQUIRE(result[2][1] == -5.0F);
    REQUIRE(result[2][2] == 5.0F);
    REQUIRE(result[2][3] == -5.0F);

    REQUIRE(result[3][0] == 9.0F);
    REQUIRE(result[3][1] == 5.0F);
    REQUIRE(result[3][2] == 9.0F);
    REQUIRE(result[3][3] == 5.0F);
}

TEST_CASE("2x2 matrix increment", "matrix")
{
    omath::Matrix<float, 2> matrix1{
        0.0F, 1.0F,
        2.0F, -3.0F
    };

    const omath::Matrix<float, 2> matrix2{
        5.0F, -6.0F,
        7.0F, 8.0F
    };

    matrix1 += matrix2;

    REQUIRE(matrix1[0][0] == 5.0F);
    REQUIRE(matrix1[0][1] == -5.0F);
    REQUIRE(matrix1[1][0] == 9.0F);
    REQUIRE(matrix1[1][1] == 5.0F);
}

TEST_CASE("4x4 matrix increment", "matrix")
{
    omath::Matrix<float, 4, 4, false> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4, false> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    matrix1 += matrix2;

    REQUIRE(matrix1[0][0] == 5.0F);
    REQUIRE(matrix1[0][1] == -5.0F);
    REQUIRE(matrix1[0][2] == 5.0F);
    REQUIRE(matrix1[0][3] == -5.0F);

    REQUIRE(matrix1[1][0] == 9.0F);
    REQUIRE(matrix1[1][1] == 5.0F);
    REQUIRE(matrix1[1][2] == 9.0F);
    REQUIRE(matrix1[1][3] == 5.0F);

    REQUIRE(matrix1[2][0] == 5.0F);
    REQUIRE(matrix1[2][1] == -5.0F);
    REQUIRE(matrix1[2][2] == 5.0F);
    REQUIRE(matrix1[2][3] == -5.0F);

    REQUIRE(matrix1[3][0] == 9.0F);
    REQUIRE(matrix1[3][1] == 5.0F);
    REQUIRE(matrix1[3][2] == 9.0F);
    REQUIRE(matrix1[3][3] == 5.0F);
}

TEST_CASE("4x4 matrix increment using SIMD", "matrix")
{
    omath::Matrix<float, 4, 4, true> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4, true> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    matrix1 += matrix2;

    REQUIRE(matrix1[0][0] == 5.0F);
    REQUIRE(matrix1[0][1] == -5.0F);
    REQUIRE(matrix1[0][2] == 5.0F);
    REQUIRE(matrix1[0][3] == -5.0F);

    REQUIRE(matrix1[1][0] == 9.0F);
    REQUIRE(matrix1[1][1] == 5.0F);
    REQUIRE(matrix1[1][2] == 9.0F);
    REQUIRE(matrix1[1][3] == 5.0F);

    REQUIRE(matrix1[2][0] == 5.0F);
    REQUIRE(matrix1[2][1] == -5.0F);
    REQUIRE(matrix1[2][2] == 5.0F);
    REQUIRE(matrix1[2][3] == -5.0F);

    REQUIRE(matrix1[3][0] == 9.0F);
    REQUIRE(matrix1[3][1] == 5.0F);
    REQUIRE(matrix1[3][2] == 9.0F);
    REQUIRE(matrix1[3][3] == 5.0F);
}

TEST_CASE("2x2 matrix difference", "matrix")
{
    const omath::Matrix<float, 2> matrix1{
        0.0F, 1.0F,
        2.0F, -3.0F
    };

    const omath::Matrix<float, 2> matrix2{
        5.0F, -6.0F,
        7.0F, 8.0F
    };

    const auto result = matrix1 - matrix2;

    REQUIRE(result[0][0] == -5.0F);
    REQUIRE(result[0][1] == 7.0F);
    REQUIRE(result[1][0] == -5.0F);
    REQUIRE(result[1][1] == -11.0F);
}

TEST_CASE("4x4 matrix difference", "matrix")
{
    const omath::Matrix<float, 4, 4, false> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4, false> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    const auto result = matrix1 - matrix2;

    REQUIRE(result[0][0] == -5.0F);
    REQUIRE(result[0][1] == 7.0F);
    REQUIRE(result[0][2] == -5.0F);
    REQUIRE(result[0][3] == 7.0F);

    REQUIRE(result[1][0] == -5.0F);
    REQUIRE(result[1][1] == -11.0F);
    REQUIRE(result[1][2] == -5.0F);
    REQUIRE(result[1][3] == -11.0F);

    REQUIRE(result[2][0] == -5.0F);
    REQUIRE(result[2][1] == 7.0F);
    REQUIRE(result[2][2] == -5.0F);
    REQUIRE(result[2][3] == 7.0F);

    REQUIRE(result[3][0] == -5.0F);
    REQUIRE(result[3][1] == -11.0F);
    REQUIRE(result[3][2] == -5.0F);
    REQUIRE(result[3][3] == -11.0F);
}

TEST_CASE("4x4 matrix difference using SIMD", "matrix")
{
    const omath::Matrix<float, 4, 4, true> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4, true> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    const auto result = matrix1 - matrix2;

    REQUIRE(result[0][0] == -5.0F);
    REQUIRE(result[0][1] == 7.0F);
    REQUIRE(result[0][2] == -5.0F);
    REQUIRE(result[0][3] == 7.0F);

    REQUIRE(result[1][0] == -5.0F);
    REQUIRE(result[1][1] == -11.0F);
    REQUIRE(result[1][2] == -5.0F);
    REQUIRE(result[1][3] == -11.0F);

    REQUIRE(result[2][0] == -5.0F);
    REQUIRE(result[2][1] == 7.0F);
    REQUIRE(result[2][2] == -5.0F);
    REQUIRE(result[2][3] == 7.0F);

    REQUIRE(result[3][0] == -5.0F);
    REQUIRE(result[3][1] == -11.0F);
    REQUIRE(result[3][2] == -5.0F);
    REQUIRE(result[3][3] == -11.0F);
}

TEST_CASE("2x2 matrix decrement", "matrix")
{
    omath::Matrix<float, 2> matrix1{
        0.0F, 1.0F,
        2.0F, -3.0F
    };

    const omath::Matrix<float, 2> matrix2{
        5.0F, -6.0F,
        7.0F, 8.0F
    };

    matrix1 -= matrix2;

    REQUIRE(matrix1[0][0] == -5.0F);
    REQUIRE(matrix1[0][1] == 7.0F);
    REQUIRE(matrix1[1][0] == -5.0F);
    REQUIRE(matrix1[1][1] == -11.0F);
}

TEST_CASE("4x4 matrix decrement", "matrix")
{
    omath::Matrix<float, 4, 4, false> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4, false> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    matrix1 -= matrix2;

    REQUIRE(matrix1[0][0] == -5.0F);
    REQUIRE(matrix1[0][1] == 7.0F);
    REQUIRE(matrix1[0][2] == -5.0F);
    REQUIRE(matrix1[0][3] == 7.0F);

    REQUIRE(matrix1[1][0] == -5.0F);
    REQUIRE(matrix1[1][1] == -11.0F);
    REQUIRE(matrix1[1][2] == -5.0F);
    REQUIRE(matrix1[1][3] == -11.0F);

    REQUIRE(matrix1[2][0] == -5.0F);
    REQUIRE(matrix1[2][1] == 7.0F);
    REQUIRE(matrix1[2][2] == -5.0F);
    REQUIRE(matrix1[2][3] == 7.0F);

    REQUIRE(matrix1[3][0] == -5.0F);
    REQUIRE(matrix1[3][1] == -11.0F);
    REQUIRE(matrix1[3][2] == -5.0F);
    REQUIRE(matrix1[3][3] == -11.0F);
}

TEST_CASE("4x4 matrix decrement using SIMD", "matrix")
{
    omath::Matrix<float, 4, 4, true> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4, true> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    matrix1 -= matrix2;

    REQUIRE(matrix1[0][0] == -5.0F);
    REQUIRE(matrix1[0][1] == 7.0F);
    REQUIRE(matrix1[0][2] == -5.0F);
    REQUIRE(matrix1[0][3] == 7.0F);

    REQUIRE(matrix1[1][0] == -5.0F);
    REQUIRE(matrix1[1][1] == -11.0F);
    REQUIRE(matrix1[1][2] == -5.0F);
    REQUIRE(matrix1[1][3] == -11.0F);

    REQUIRE(matrix1[2][0] == -5.0F);
    REQUIRE(matrix1[2][1] == 7.0F);
    REQUIRE(matrix1[2][2] == -5.0F);
    REQUIRE(matrix1[2][3] == 7.0F);

    REQUIRE(matrix1[3][0] == -5.0F);
    REQUIRE(matrix1[3][1] == -11.0F);
    REQUIRE(matrix1[3][2] == -5.0F);
    REQUIRE(matrix1[3][3] == -11.0F);
}

TEST_CASE("2x2 matrix multiplication with scalar", "matrix")
{
    const omath::Matrix<float, 2> matrix{
        2.0F, 3.0F,
        4.0F, 5.0F
    };

    const auto result = matrix * 2.0F;

    REQUIRE(result[0][0] == 4.0F);
    REQUIRE(result[0][1] == 6.0F);
    REQUIRE(result[1][0] == 8.0F);
    REQUIRE(result[1][1] == 10.0F);
}

TEST_CASE("4x4 matrix multiplication with scalar", "matrix")
{
    const omath::Matrix<float, 4, 4, false> matrix{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const auto result = matrix * 2.0F;

    REQUIRE(result[0][0] == 0.0F);
    REQUIRE(result[0][1] == 2.0F);
    REQUIRE(result[0][2] == 0.0F);
    REQUIRE(result[0][3] == 2.0F);

    REQUIRE(result[1][0] == 4.0F);
    REQUIRE(result[1][1] == -6.0F);
    REQUIRE(result[1][2] == 4.0F);
    REQUIRE(result[1][3] == -6.0F);

    REQUIRE(result[2][0] == 0.0F);
    REQUIRE(result[2][1] == 2.0F);
    REQUIRE(result[2][2] == 0.0F);
    REQUIRE(result[2][3] == 2.0F);

    REQUIRE(result[3][0] == 4.0F);
    REQUIRE(result[3][1] == -6.0F);
    REQUIRE(result[3][2] == 4.0F);
    REQUIRE(result[3][3] == -6.0F);
}

TEST_CASE("4x4 matrix multiplication with scalar using SIMD", "matrix")
{
    const omath::Matrix<float, 4, 4, true> matrix{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4, true> result = matrix * 2.0F;

    REQUIRE(result[0][0] == 0.0F);
    REQUIRE(result[0][1] == 2.0F);
    REQUIRE(result[0][2] == 0.0F);
    REQUIRE(result[0][3] == 2.0F);

    REQUIRE(result[1][0] == 4.0F);
    REQUIRE(result[1][1] == -6.0F);
    REQUIRE(result[1][2] == 4.0F);
    REQUIRE(result[1][3] == -6.0F);

    REQUIRE(result[2][0] == 0.0F);
    REQUIRE(result[2][1] == 2.0F);
    REQUIRE(result[2][2] == 0.0F);
    REQUIRE(result[2][3] == 2.0F);

    REQUIRE(result[3][0] == 4.0F);
    REQUIRE(result[3][1] == -6.0F);
    REQUIRE(result[3][2] == 4.0F);
    REQUIRE(result[3][3] == -6.0F);
}

TEST_CASE("2x2 matrix multiplication assignment with scalar", "matrix")
{
    omath::Matrix<float, 2> matrix{
        2.0F, 3.0F,
        4.0F, 5.0F
    };

    matrix *= 2.0F;

    REQUIRE(matrix[0][0] == 4.0F);
    REQUIRE(matrix[0][1] == 6.0F);
    REQUIRE(matrix[1][0] == 8.0F);
    REQUIRE(matrix[1][1] == 10.0F);
}

TEST_CASE("4x4 matrix multiplication assignment with scalar", "matrix")
{
    omath::Matrix<float, 4, 4, false> matrix{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    matrix *= 2.0F;

    REQUIRE(matrix[0][0] == 0.0F);
    REQUIRE(matrix[0][1] == 2.0F);
    REQUIRE(matrix[0][2] == 0.0F);
    REQUIRE(matrix[0][3] == 2.0F);

    REQUIRE(matrix[1][0] == 4.0F);
    REQUIRE(matrix[1][1] == -6.0F);
    REQUIRE(matrix[1][2] == 4.0F);
    REQUIRE(matrix[1][3] == -6.0F);

    REQUIRE(matrix[2][0] == 0.0F);
    REQUIRE(matrix[2][1] == 2.0F);
    REQUIRE(matrix[2][2] == 0.0F);
    REQUIRE(matrix[2][3] == 2.0F);

    REQUIRE(matrix[3][0] == 4.0F);
    REQUIRE(matrix[3][1] == -6.0F);
    REQUIRE(matrix[3][2] == 4.0F);
    REQUIRE(matrix[3][3] == -6.0F);
}

TEST_CASE("1x1 matrix multiplication with matrix", "matrix")
{
    const omath::Matrix<float, 1> matrix1{
        2.0F
    };

    const omath::Matrix<float, 1> matrix2{
        3.0F
    };

    const auto result = matrix1 * matrix2;

    REQUIRE(result[0][0] == 6.0F);
}

TEST_CASE("2x2 matrix multiplication with matrix", "matrix")
{
    const omath::Matrix<float, 2> matrix1{
        2.0F, 3.0F,
        4.0F, 5.0F
    };

    const omath::Matrix<float, 2> matrix2{
        1.0F, 2.0F,
        3.0F, 4.0F
    };

    const auto result = matrix1 * matrix2;

    REQUIRE(result[0][0] == 10.0F);
    REQUIRE(result[0][1] == 13.0F);
    REQUIRE(result[1][0] == 22.0F);
    REQUIRE(result[1][1] == 29.0F);
}

TEST_CASE("1x2 matrix multiplication with 2x1 matrix", "matrix")
{
    const omath::Matrix<float, 1, 2> matrix1{
        2.0F,
        3.0F
    };

    const omath::Matrix<float, 2, 1> matrix2{
        1.0F, 2.0F
    };

    const omath::Matrix<float, 1, 1> result = matrix1 * matrix2;

    REQUIRE(result[0][0] == 8.0F);
}

TEST_CASE("1x3 matrix multiplication with 3x2 matrix", "matrix")
{
    const omath::Matrix<float, 1, 3> matrix1{
        2.0F,
        3.0F,
        4.0F
    };

    const omath::Matrix<float, 3, 2> matrix2{
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F
    };

    const omath::Matrix<float, 1, 2> result = matrix1 * matrix2;

    REQUIRE(result[0][0] == 20.0F);
    REQUIRE(result[0][1] == 47.0F);
}

TEST_CASE("2x1 matrix multiplication with 1x2 matrix", "matrix")
{
    const omath::Matrix<float, 2, 1> matrix1{
        2.0F, 3.0F
    };

    const omath::Matrix<float, 1, 2> matrix2{
        1.0F,
        2.0F
    };

    const omath::Matrix<float, 2, 2> result = matrix1 * matrix2;

    REQUIRE(result[0][0] == 2.0F);
    REQUIRE(result[0][1] == 3.0F);
    REQUIRE(result[1][0] == 4.0F);
    REQUIRE(result[1][1] == 6.0F);
}
