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

TEST_CASE("Matrix transpose", "matrix")
{
    SECTION("1x1")
    {
        omath::Matrix<float, 1> matrix{1.0F};
        matrix.transpose();

        REQUIRE(matrix[0][0] == 1.0F);
    }

    SECTION("2x2")
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

    SECTION("3x3")
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
}

TEST_CASE("Matrix comparison", "matrix")
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

TEST_CASE("Negative of matrix", "matrix")
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

TEST_CASE("Negative of matrix using SIMD", "matrix")
{
    const omath::Matrix<float, 4> matrix{
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

TEST_CASE("Matrix sum", "matrix")
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

TEST_CASE("Matrix difference", "matrix")
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

TEST_CASE("Matrix multiplication", "matrix")
{
    SECTION("float")
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

    SECTION("float assignment")
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

    SECTION("1x1")
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

    SECTION("2x2")
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

    SECTION("1x2")
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

    SECTION("1x3")
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

    SECTION("2x1")
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
}
