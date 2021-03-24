#include "catch2/catch.hpp"
#include "Matrix.hpp"

TEST_CASE("Matrix zero initalization", "matrix")
{
    const math::Matrix<float, 4> matrix;
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            REQUIRE(matrix[row][column] == 0.0F);
}

TEST_CASE("Matrix value initalization", "matrix")
{
    const math::Matrix<float, 2, 2> matrix{
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
    const auto matrix = math::Matrix<float, 2>::identity();

    REQUIRE(matrix[0][0] == 1.0F);
    REQUIRE(matrix[0][1] == 0.0F);
    REQUIRE(matrix[1][0] == 0.0F);
    REQUIRE(matrix[1][1] == 1.0F);
}

TEST_CASE("Matrix element setter", "matrix")
{
    math::Matrix<float, 2> matrix;

    matrix[0][0] = 1.0F;
    matrix[0][1] = 2.0F;
    matrix[1][0] = 3.0F;
    matrix[1][1] = 4.0F;

    REQUIRE(matrix[0][0] == 1.0F);
    REQUIRE(matrix[0][1] == 2.0F);
    REQUIRE(matrix[1][0] == 3.0F);
    REQUIRE(matrix[1][1] == 4.0F);
}

TEST_CASE("Matrix comparison", "matrix")
{
    const math::Matrix<float, 2> matrix1{
        0.0F, 1.0F,
        2.0F, 3.0F
    };

    const math::Matrix<float, 2> matrix2{
        0.0F, 1.0F,
        2.0F, 3.0F
    };

    const math::Matrix<float, 2> matrix3{
        1.0F, 2.0F,
        3.0F, 4.0F
    };

    REQUIRE(matrix1 == matrix2);
    REQUIRE(matrix1 != matrix3);
}

TEST_CASE("Matrix multiplication", "matrix")
{
    SECTION("1x1")
    {
        const math::Matrix<float, 1> matrix1{
            2.0F
        };

        const math::Matrix<float, 1> matrix2{
            3.0F
        };

        const auto result = matrix1 * matrix2;

        REQUIRE(result[0][0] == 6.0F);
    }

    SECTION("2x2")
    {
        const math::Matrix<float, 2> matrix1{
            2.0F, 3.0F,
            4.0F, 5.0F
        };

        const math::Matrix<float, 2> matrix2{
            1.0F, 2.0F,
            3.0F, 4.0F
        };

        const auto result = matrix1 * matrix2;

        REQUIRE(result[0][0] == 10.0F);
        REQUIRE(result[0][1] == 13.0F);
        REQUIRE(result[1][0] == 22.0F);
        REQUIRE(result[1][1] == 29.0F);
    }
}
