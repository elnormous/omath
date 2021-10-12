#include "catch2/catch.hpp"
#include "Matrix.hpp"

TEST_CASE("Matrix zero initalization", "matrix")
{
    const omath::Matrix<float, 4> matrix{};
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            REQUIRE(matrix[row][column] == 0.0F);
}

TEST_CASE("Matrix element assignment", "matrix")
{
    omath::Matrix<float, 2> matrix;
    matrix[0][0] = 0.0F;
    matrix[0][1] = 1.0F;
    matrix[1][0] = 2.0F;
    matrix[1][1] = 3.0F;

    REQUIRE(matrix == omath::Matrix<float, 2, 2>{
        0.0F, 1.0F,
        2.0F, 3.0F
    });
}

TEST_CASE("Matrix value initalization", "matrix")
{
    const omath::Matrix<float, 2, 2> matrix{
        0.0F, 1.0F,
        2.0F, 3.0F
    };

    REQUIRE(matrix == omath::Matrix<float, 2, 2>{
        0.0F, 1.0F,
        2.0F, 3.0F
    });
}

TEST_CASE("Matrix identity", "matrix")
{
    omath::Matrix<float, 2, 2> matrix{};

    setIdentity(matrix);

    REQUIRE(matrix == omath::Matrix<float, 2, 2>{
        1.0F, 0.0F,
        0.0F, 1.0F
    });
}

TEST_CASE("Matrix set identity", "matrix")
{
    const auto matrix = omath::identityMatrix<float, 2>();

    REQUIRE(matrix == omath::Matrix<float, 2, 2>{
        1.0F, 0.0F,
        0.0F, 1.0F
    });
}

TEST_CASE("Matrix element setter", "matrix")
{
    omath::Matrix<float, 2> matrix;

    matrix[0][0] = 1.0F;
    matrix[0][1] = 2.0F;
    matrix[1][0] = 3.0F;
    matrix[1][1] = 4.0F;

    REQUIRE(matrix == omath::Matrix<float, 2, 2>{
        1.0F, 2.0F,
        3.0F, 4.0F
    });
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
    transpose(matrix);

    REQUIRE(matrix == omath::Matrix<float, 1, 1>{
        1.0F
    });
}

TEST_CASE("2x2 matrix transpose", "matrix")
{
    omath::Matrix<float, 2> matrix{
        1.0F, 2.0F,
        3.0F, 4.0F
    };
    transpose(matrix);

    REQUIRE(matrix == omath::Matrix<float, 2, 2>{
        1.0F, 3.0F,
        2.0F, 4.0F
    });
}

TEST_CASE("3x3 matrix transpose", "matrix")
{
    omath::Matrix<float, 3> matrix{
        1.0F, 2.0F, 3.0F,
        4.0F, 6.0F, 6.0F,
        7.0F, 8.0F, 9.0F
    };
    transpose(matrix);

    REQUIRE(matrix == omath::Matrix<float, 3, 3>{
        1.0F, 4.0F, 7.0F,
        2.0F, 6.0F, 8.0F,
        3.0F, 6.0F, 9.0F
    });
}

TEST_CASE("4x4 matrix transpose", "matrix")
{
    omath::Matrix<float, 4, 4> matrix{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        9.0F, 10.0F, 11.0F, 12.0F,
        13.0F, 14.0F, 15.0F, 16.0F
    };
    transpose(matrix);

    REQUIRE(matrix == omath::Matrix<float, 4, 4>{
        1.0F, 5.0F, 9.0F, 13.0F,
        2.0F, 6.0F, 10.0F, 14.0F,
        3.0F, 7.0F, 11.0F, 15.0F,
        4.0F, 8.0F, 12.0F, 16.0F
    });
}

TEST_CASE("4x4 double matrix transpose", "matrix")
{
    omath::Matrix<double, 4, 4> matrix{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };
    transpose(matrix);

    REQUIRE(matrix == omath::Matrix<double, 4, 4>{
        1.0, 5.0, 9.0, 13.0,
        2.0, 6.0, 10.0, 14.0,
        3.0, 7.0, 11.0, 15.0,
        4.0, 8.0, 12.0, 16.0
    });
}

TEST_CASE("1x1 matrix transposed", "matrix")
{
    const omath::Matrix<float, 1> matrix{1.0F};
    const omath::Matrix<float, 1> result = transposed(matrix);

    REQUIRE(result == omath::Matrix<float, 1, 1>{
        1.0F
    });
}

TEST_CASE("2x1 matrix transposed", "matrix")
{
    const omath::Matrix<float, 2, 1> matrix{
        1.0F,
        3.0F
    };
    const omath::Matrix<float, 1, 2> result = transposed(matrix);

    REQUIRE(result == omath::Matrix<float, 1, 2>{
        1.0F, 3.0F
    });
}

TEST_CASE("2x2 matrix transposed", "matrix")
{
    const omath::Matrix<float, 2> matrix{
        1.0F, 2.0F,
        3.0F, 4.0F
    };
    const omath::Matrix<float, 2> result = transposed(matrix);

    REQUIRE(result == omath::Matrix<float, 2, 2>{
        1.0F, 3.0F,
        2.0F, 4.0F
    });
}

TEST_CASE("2x3 matrix transposed", "matrix")
{
    const omath::Matrix<float, 2, 3> matrix{
        1.0F, 2.0F, 3.0F,
        4.0F, 6.0F, 6.0F
    };
    const omath::Matrix<float, 3, 2> result = transposed(matrix);

    REQUIRE(result == omath::Matrix<float, 3, 2>{
        1.0F, 4.0F,
        2.0F, 6.0F,
        3.0F, 6.0F
    });
}

TEST_CASE("3x3 matrix transposed", "matrix")
{
    const omath::Matrix<float, 3> matrix{
        1.0F, 2.0F, 3.0F,
        4.0F, 6.0F, 6.0F,
        7.0F, 8.0F, 9.0F
    };
    const omath::Matrix<float, 3> result = transposed(matrix);

    REQUIRE(result == omath::Matrix<float, 3, 3>{
        1.0F, 4.0F, 7.0F,
        2.0F, 6.0F, 8.0F,
        3.0F, 6.0F, 9.0F
    });
}

TEST_CASE("4x4 matrix transposed", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        9.0F, 10.0F, 11.0F, 12.0F,
        13.0F, 14.0F, 15.0F, 16.0F
    };
    const omath::Matrix<float, 4, 4> result = transposed(matrix);

    REQUIRE(result == omath::Matrix<float, 4, 4>{
        1.0F, 5.0F, 9.0F, 13.0F,
        2.0F, 6.0F, 10.0F, 14.0F,
        3.0F, 7.0F, 11.0F, 15.0F,
        4.0F, 8.0F, 12.0F, 16.0F
    });
}

TEST_CASE("4x4 double matrix transposed", "matrix")
{
    const omath::Matrix<double, 4, 4> matrix{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };
    const omath::Matrix<double, 4, 4> result = transposed(matrix);

    REQUIRE(result == omath::Matrix<double, 4, 4>{
        1.0, 5.0, 9.0, 13.0,
        2.0, 6.0, 10.0, 14.0,
        3.0, 7.0, 11.0, 15.0,
        4.0, 8.0, 12.0, 16.0
    });
}

TEST_CASE("1x1 matrix determinant", "matrix")
{
    const omath::Matrix<float, 1, 1> matrix{
        2.0F
    };

    REQUIRE(determinant(matrix) == 2.0F);
}

TEST_CASE("2x2 matrix determinant", "matrix")
{
    const omath::Matrix<float, 2, 2> matrix{
        1.0F, 2.0F,
        3.0F, 4.0F
    };

    REQUIRE(determinant(matrix) == -2.0F);
}

TEST_CASE("3x3 matrix determinant", "matrix")
{
    const omath::Matrix<float, 3, 3> matrix{
        2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F,
        8.0F, 9.0F, 0.0F
    };

    REQUIRE(determinant(matrix) == 30.0F);
}

TEST_CASE("4x4 matrix determinant", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        2.0F, 3.0F, 4.0F, 0.0F,
        5.0F, 6.0F, 7.0F, 0.0F,
        8.0F, 9.0F, 1.0F, 0.0F,
        2.0F, 3.0F, 4.0F, 1.0F
    };

    REQUIRE(determinant(matrix) == 27.0F);
}

TEST_CASE("2x2 matrix negative", "matrix")
{
    const omath::Matrix<float, 2> matrix{
        0.0F, 1.0F,
        2.0F, -3.0F
    };

    const auto result = -matrix;

    REQUIRE(result == omath::Matrix<float, 2, 2>{
        0.0F, -1.0F,
        -2.0F, 3.0F
    });
}

TEST_CASE("4x4 matrix negative", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        0.0F, 1.0F, 2.0F, 3.0F,
        2.0F, -3.0F, 4.0F, 5.0F,
        3.0F, 4.0F, 5.0F, 6.0F,
        4.0F, 5.0F, 6.0F, 7.0F
    };

    const auto result = -matrix;

    REQUIRE(result == omath::Matrix<float, 4, 4>{
        0.0F, -1.0F, -2.0F, -3.0F,
        -2.0F, 3.0F, -4.0F, -5.0F,
        -3.0F, -4.0F, -5.0F, -6.0F,
        -4.0F, -5.0F, -6.0F, -7.0F
    });
}

TEST_CASE("4x4 double matrix negative", "matrix")
{
    const omath::Matrix<double, 4, 4> matrix{
        0.0, 1.0, 2.0, 3.0,
        2.0, -3.0, 4.0, 5.0,
        3.0, 4.0, 5.0, 6.0,
        4.0, 5.0, 6.0, 7.0
    };

    const auto result = -matrix;

    REQUIRE(result == omath::Matrix<double, 4, 4>{
        0.0, -1.0, -2.0, -3.0,
        -2.0, 3.0, -4.0, -5.0,
        -3.0, -4.0, -5.0, -6.0,
        -4.0, -5.0, -6.0, -7.0
    });
}

TEST_CASE("2x2 matrix negate", "matrix")
{
    omath::Matrix<float, 2> matrix{
        0.0F, 1.0F,
        2.0F, -3.0F
    };

    negate(matrix);

    REQUIRE(matrix == omath::Matrix<float, 2, 2>{
        0.0F, -1.0F,
        -2.0F, 3.0F
    });
}

TEST_CASE("4x4 matrix negate", "matrix")
{
    omath::Matrix<float, 4, 4> matrix{
        0.0F, 1.0F, 2.0F, 3.0F,
        2.0F, -3.0F, 4.0F, 5.0F,
        3.0F, 4.0F, 5.0F, 6.0F,
        4.0F, 5.0F, 6.0F, 7.0F
    };

    negate(matrix);

    REQUIRE(matrix == omath::Matrix<float, 4, 4>{
        0.0F, -1.0F, -2.0F, -3.0F,
        -2.0F, 3.0F, -4.0F, -5.0F,
        -3.0F, -4.0F, -5.0F, -6.0F,
        -4.0F, -5.0F, -6.0F, -7.0F
    });
}

TEST_CASE("4x4 double matrix negate", "matrix")
{
    omath::Matrix<double, 4, 4> matrix{
        0.0, 1.0, 2.0, 3.0,
        2.0, -3.0, 4.0, 5.0,
        3.0, 4.0, 5.0, 6.0,
        4.0, 5.0, 6.0, 7.0
    };

    negate(matrix);

    REQUIRE(matrix == omath::Matrix<double, 4, 4>{
        0.0, -1.0, -2.0, -3.0,
        -2.0, 3.0, -4.0, -5.0,
        -3.0, -4.0, -5.0, -6.0,
        -4.0, -5.0, -6.0, -7.0
    });
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

    REQUIRE(result == omath::Matrix<float, 2, 2>{
        5.0F, -5.0F,
        9.0F, 5.0F
    });
}

TEST_CASE("4x4 matrix sum", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    const auto result = matrix1 + matrix2;

    REQUIRE(result == omath::Matrix<float, 4, 4>{
        5.0F, -5.0F, 5.0F, -5.0F,
        9.0F, 5.0F, 9.0F, 5.0F,
        5.0F, -5.0F, 5.0F, -5.0F,
        9.0F, 5.0F, 9.0F, 5.0F
    });
}

TEST_CASE("4x4 double matrix sum", "matrix")
{
    const omath::Matrix<double, 4, 4> matrix1{
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0,
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0
    };

    const omath::Matrix<double, 4, 4> matrix2{
        5.0, -6.0, 5.0, -6.0,
        7.0, 8.0, 7.0, 8.0,
        5.0, -6.0, 5.0, -6.0,
        7.0, 8.0, 7.0, 8.0
    };

    const auto result = matrix1 + matrix2;

    REQUIRE(result == omath::Matrix<double, 4, 4>{
        5.0, -5.0, 5.0, -5.0,
        9.0, 5.0, 9.0, 5.0,
        5.0, -5.0, 5.0, -5.0,
        9.0, 5.0, 9.0, 5.0
    });
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

    REQUIRE(matrix1 == omath::Matrix<float, 2, 2>{
        5.0F, -5.0F,
        9.0F, 5.0F
    });
}

TEST_CASE("4x4 matrix increment", "matrix")
{
    omath::Matrix<float, 4, 4> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    matrix1 += matrix2;

    REQUIRE(matrix1 == omath::Matrix<float, 4, 4>{
        5.0F, -5.0F, 5.0F, -5.0F,
        9.0F, 5.0F, 9.0F, 5.0F,
        5.0F, -5.0F, 5.0F, -5.0F,
        9.0F, 5.0F, 9.0F, 5.0F
    });
}

TEST_CASE("4x4 double matrix increment", "matrix")
{
    omath::Matrix<double, 4, 4> matrix1{
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0,
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0
    };

    const omath::Matrix<double, 4, 4> matrix2{
        5.0, -6.0, 5.0, -6.0,
        7.0, 8.0, 7.0, 8.0,
        5.0, -6.0, 5.0, -6.0,
        7.0, 8.0, 7.0, 8.0
    };

    matrix1 += matrix2;

    REQUIRE(matrix1 == omath::Matrix<double, 4, 4>{
        5.0, -5.0, 5.0, -5.0,
        9.0, 5.0, 9.0, 5.0,
        5.0, -5.0, 5.0, -5.0,
        9.0, 5.0, 9.0, 5.0
    });
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

    REQUIRE(result == omath::Matrix<float, 2, 2>{
        -5.0F, 7.0F,
        -5.0F, -11.0F
    });
}

TEST_CASE("4x4 matrix difference", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    const auto result = matrix1 - matrix2;

    REQUIRE(result == omath::Matrix<float, 4, 4>{
        -5.0F, 7.0F, -5.0F, 7.0F,
        -5.0F, -11.0F, -5.0F, -11.0F,
        -5.0F, 7.0F, -5.0F, 7.0F,
        -5.0F, -11.0F, -5.0F, -11.0F
    });
}

TEST_CASE("4x4 double matrix difference", "matrix")
{
    const omath::Matrix<double, 4, 4> matrix1{
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0,
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0
    };

    const omath::Matrix<double, 4, 4> matrix2{
        5.0, -6.0, 5.0, -6.0,
        7.0, 8.0, 7.0, 8.0,
        5.0, -6.0, 5.0, -6.0,
        7.0, 8.0, 7.0, 8.0
    };

    const auto result = matrix1 - matrix2;

    REQUIRE(result == omath::Matrix<double, 4, 4>{
        -5.0, 7.0, -5.0, 7.0,
        -5.0, -11.0, -5.0, -11.0,
        -5.0, 7.0, -5.0, 7.0,
        -5.0, -11.0, -5.0, -11.0
    });
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

    REQUIRE(matrix1 == omath::Matrix<float, 2, 2>{
        -5.0F, 7.0F,
        -5.0F, -11.0F
    });
}

TEST_CASE("4x4 matrix decrement", "matrix")
{
    omath::Matrix<float, 4, 4> matrix1{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4> matrix2{
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F,
        5.0F, -6.0F, 5.0F, -6.0F,
        7.0F, 8.0F, 7.0F, 8.0F
    };

    matrix1 -= matrix2;

    REQUIRE(matrix1 == omath::Matrix<float, 4, 4>{
        -5.0F, 7.0F, -5.0F, 7.0F,
        -5.0F, -11.0F, -5.0F, -11.0F,
        -5.0F, 7.0F, -5.0F, 7.0F,
        -5.0F, -11.0F, -5.0F, -11.0F
    });
}

TEST_CASE("4x4 double matrix decrement", "matrix")
{
    omath::Matrix<double, 4, 4> matrix1{
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0,
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0
    };

    const omath::Matrix<double, 4, 4> matrix2{
        5.0, -6.0, 5.0, -6.0,
        7.0, 8.0, 7.0, 8.0,
        5.0, -6.0, 5.0, -6.0,
        7.0, 8.0, 7.0, 8.0
    };

    matrix1 -= matrix2;

    REQUIRE(matrix1 == omath::Matrix<double, 4, 4>{
        -5.0, 7.0, -5.0, 7.0,
        -5.0, -11.0, -5.0, -11.0,
        -5.0, 7.0, -5.0, 7.0,
        -5.0, -11.0, -5.0, -11.0
    });
}

TEST_CASE("2x2 matrix multiplication with scalar", "matrix")
{
    const omath::Matrix<float, 2> matrix{
        2.0F, 3.0F,
        4.0F, 5.0F
    };

    const auto result = matrix * 2.0F;

    REQUIRE(result == omath::Matrix<float, 2, 2>{
        4.0F, 6.0F,
        8.0F, 10.0F
    });
}

TEST_CASE("4x4 matrix multiplication with scalar", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const auto result = matrix * 2.0F;

    REQUIRE(result == omath::Matrix<float, 4, 4>{
        0.0F, 2.0F, 0.0F, 2.0F,
        4.0F, -6.0F, 4.0F, -6.0F,
        0.0F, 2.0F, 0.0F, 2.0F,
        4.0F, -6.0F, 4.0F, -6.0F
    });
}

TEST_CASE("4x4 double matrix multiplication with scalar", "matrix")
{
    const omath::Matrix<double, 4, 4> matrix{
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0,
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0
    };

    const auto result = matrix * 2.0;

    REQUIRE(result == omath::Matrix<double, 4, 4>{
        0.0, 2.0, 0.0, 2.0,
        4.0, -6.0, 4.0, -6.0,
        0.0, 2.0, 0.0, 2.0,
        4.0, -6.0, 4.0, -6.0
    });
}

TEST_CASE("2x2 matrix multiplication assignment with scalar", "matrix")
{
    omath::Matrix<float, 2> matrix{
        2.0F, 3.0F,
        4.0F, 5.0F
    };

    matrix *= 2.0F;

    REQUIRE(matrix == omath::Matrix<float, 2, 2>{
        4.0F, 6.0F,
        8.0F, 10.0F
    });
}

TEST_CASE("4x4 matrix multiplication assignment with scalar", "matrix")
{
    omath::Matrix<float, 4, 4> matrix{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    matrix *= 2.0F;

    REQUIRE(matrix == omath::Matrix<float, 4, 4>{
        0.0F, 2.0F, 0.0F, 2.0F,
        4.0F, -6.0F, 4.0F, -6.0F,
        0.0F, 2.0F, 0.0F, 2.0F,
        4.0F, -6.0F, 4.0F, -6.0F
    });
}

TEST_CASE("4x4 double matrix multiplication assignment with scalar", "matrix")
{
    omath::Matrix<double, 4, 4> matrix{
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0,
        0.0, 1.0, 0.0, 1.0,
        2.0, -3.0, 2.0, -3.0
    };

    matrix *= 2.0;

    REQUIRE(matrix == omath::Matrix<double, 4, 4>{
        0.0, 2.0, 0.0, 2.0,
        4.0, -6.0, 4.0, -6.0,
        0.0, 2.0, 0.0, 2.0,
        4.0, -6.0, 4.0, -6.0
    });
}

TEST_CASE("2x2 matrix divison with scalar", "matrix")
{
    const omath::Matrix<float, 2> matrix{
        2.0F, 4.0F,
        4.0F, 10.0F
    };

    const auto result = matrix / 2.0F;

    REQUIRE(result == omath::Matrix<float, 2, 2>{
        1.0F, 2.0F,
        2.0F, 5.0F
    });
}

TEST_CASE("4x4 matrix divison with scalar", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        0.0F, 2.0F, 0.0F, 2.0F,
        2.0F, -4.0F, 2.0F, -4.0F,
        0.0F, 2.0F, 0.0F, 2.0F,
        2.0F, -4.0F, 2.0F, -4.0F
    };

    const auto result = matrix / 2.0F;

    REQUIRE(result == omath::Matrix<float, 4, 4>{
        0.0F, 1.0F, 0.0F, 1.0F,
        1.0F, -2.0F, 1.0F, -2.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        1.0F, -2.0F, 1.0F, -2.0F
    });
}

TEST_CASE("4x4 double matrix divison with scalar", "matrix")
{
    const omath::Matrix<double, 4, 4> matrix{
        0.0, 2.0, 0.0, 2.0,
        2.0, -4.0, 2.0, -4.0,
        0.0, 2.0, 0.0, 2.0,
        2.0, -4.0, 2.0, -4.0
    };

    const auto result = matrix / 2.0;

    REQUIRE(result == omath::Matrix<double, 4, 4>{
        0.0, 1.0, 0.0, 1.0,
        1.0, -2.0, 1.0, -2.0,
        0.0, 1.0, 0.0, 1.0,
        1.0, -2.0, 1.0, -2.0
    });
}

TEST_CASE("2x2 matrix division assignment with scalar", "matrix")
{
    omath::Matrix<float, 2> matrix{
        2.0F, 4.0F,
        4.0F, 10.0F
    };

    matrix /= 2.0F;

    REQUIRE(matrix == omath::Matrix<float, 2, 2>{
        1.0F, 2.0F,
        2.0F, 5.0F
    });
}

TEST_CASE("4x4 matrix division assignment with scalar", "matrix")
{
    omath::Matrix<float, 4, 4> matrix{
        0.0F, 2.0F, 0.0F, 2.0F,
        2.0F, -4.0F, 2.0F, -4.0F,
        0.0F, 2.0F, 0.0F, 2.0F,
        2.0F, -4.0F, 2.0F, -4.0F
    };

    matrix /= 2.0F;

    REQUIRE(matrix == omath::Matrix<float, 4, 4>{
        0.0F, 1.0F, 0.0F, 1.0F,
        1.0F, -2.0F, 1.0F, -2.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        1.0F, -2.0F, 1.0F, -2.0F
    });
}

TEST_CASE("4x4 double matrix divison assignment with scalar", "matrix")
{
    omath::Matrix<double, 4, 4> matrix{
        0.0, 2.0, 0.0, 2.0,
        2.0, -4.0, 2.0, -4.0,
        0.0, 2.0, 0.0, 2.0,
        2.0, -4.0, 2.0, -4.0
    };

    matrix /= 2.0;

    REQUIRE(matrix == omath::Matrix<double, 4, 4>{
        0.0, 1.0, 0.0, 1.0,
        1.0, -2.0, 1.0, -2.0,
        0.0, 1.0, 0.0, 1.0,
        1.0, -2.0, 1.0, -2.0
    });
}

TEST_CASE("Scalar multiplication with 2x2 matrix", "matrix")
{
    const omath::Matrix<float, 2> matrix{
        2.0F, 3.0F,
        4.0F, 5.0F
    };

    const auto result = 2.0F * matrix;

    REQUIRE(result == omath::Matrix<float, 2, 2>{
        4.0F, 6.0F,
        8.0F, 10.0F
    });
}

TEST_CASE("1x1 matrix multiplication", "matrix")
{
    const omath::Matrix<float, 1> matrix1{2.0F};
    const omath::Matrix<float, 1> matrix2{3.0F};

    const auto result = matrix1 * matrix2;

    REQUIRE(result == omath::Matrix<float, 1, 1>{6.0F});
}

TEST_CASE("2x2 matrix multiplication", "matrix")
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

    REQUIRE(result == omath::Matrix<float, 2, 2>{
        11.0F, 16.0F,
        19.0F, 28.0F
    });
}

TEST_CASE("1x2 matrix multiplication with 2x1 matrix", "matrix")
{
    const omath::Matrix<float, 1, 2> matrix1{
        2.0F, 3.0F
    };

    const omath::Matrix<float, 2, 1> matrix2{
        1.0F,
        2.0F
    };

    const omath::Matrix<float, 1, 1> result = matrix1 * matrix2;

    REQUIRE(result == omath::Matrix<float, 1, 1>{8.0F});
}

TEST_CASE("1x3 matrix multiplication with 3x2 matrix", "matrix")
{
    const omath::Matrix<float, 1, 3> matrix1{
        2.0F, 3.0F, 4.0F
    };

    const omath::Matrix<float, 3, 2> matrix2{
        1.0F, 2.0F,
        3.0F, 4.0F,
        5.0F, 6.0F
    };

    const omath::Matrix<float, 1, 2> result = matrix1 * matrix2;

    REQUIRE(result == omath::Matrix<float, 1, 2>{
        31.0F,
        40.0F
    });
}

TEST_CASE("2x1 matrix multiplication with 1x2 matrix", "matrix")
{
    const omath::Matrix<float, 2, 1> matrix1{
        2.0F,
        3.0F
    };

    const omath::Matrix<float, 1, 2> matrix2{
        1.0F, 2.0F
    };

    const omath::Matrix<float, 2, 2> result = matrix1 * matrix2;

    REQUIRE(result == omath::Matrix<float, 2, 2>{
        2.0F, 4.0F,
        3.0F, 6.0F
    });
}

TEST_CASE("4x4 matrix multiplication", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix1{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F
    };

    const omath::Matrix<float, 4, 4> matrix2{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F
    };

    const omath::Matrix<float, 4, 4> result = matrix1 * matrix2;

    REQUIRE(result == omath::Matrix<float, 4, 4>{
        34.0F, 44.0F, 54.0F, 64.0F,
        82.0F, 108.0F, 134.0F, 160.0F,
        34.0F, 44.0F, 54.0F, 64.0F,
        82.0F, 108.0F, 134.0F, 160.0F
    });
}

TEST_CASE("4x4 double matrix multiplication", "matrix")
{
    const omath::Matrix<double, 4, 4> matrix1{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    };

    const omath::Matrix<double, 4, 4> matrix2{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    };

    const omath::Matrix<double, 4, 4> result = matrix1 * matrix2;

    REQUIRE(result == omath::Matrix<double, 4, 4>{
        34.0, 44.0, 54.0, 64.0,
        82.0, 108.0, 134.0, 160.0,
        34.0, 44.0, 54.0, 64.0,
        82.0, 108.0, 134.0, 160.0
    });
}

TEST_CASE("4x4 matrix multiplication with 4x1 matrix", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix1{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F
    };

    const omath::Matrix<float, 4, 1> matrix2{
        1.0F,
        2.0F,
        3.0F,
        4.0F
    };

    const omath::Matrix<float, 4, 1> result = matrix1 * matrix2;

    REQUIRE(result == omath::Matrix<float, 4, 1>{
        30.0F,
        70.0F,
        30.0F,
        70.0F
    });
}

TEST_CASE("1x1 matrix multiplication assignment", "matrix")
{
    omath::Matrix<float, 1> matrix1{2.0F};
    const omath::Matrix<float, 1> matrix2{3.0F};

    matrix1 *= matrix2;

    REQUIRE(matrix1 == omath::Matrix<float, 1, 1>{6.0F});
}

TEST_CASE("2x2 matrix multiplication assignment", "matrix")
{
    omath::Matrix<float, 2> matrix1{
        2.0F, 3.0F,
        4.0F, 5.0F
    };

    const omath::Matrix<float, 2> matrix2{
        1.0F, 2.0F,
        3.0F, 4.0F
    };

    matrix1 *= matrix2;

    REQUIRE(matrix1 == omath::Matrix<float, 2, 2>{
        11.0F, 16.0F,
        19.0F, 28.0F
    });
}

TEST_CASE("2x2 matrix multiplication assignment with itself", "matrix")
{
    omath::Matrix<float, 2> matrix1{
        2.0F, 3.0F,
        4.0F, 5.0F
    };

    matrix1 *= matrix1;

    REQUIRE(matrix1 == omath::Matrix<float, 2, 2>{
        16.0F, 21.0F,
        28.0F, 37.0F
    });
}

TEST_CASE("4x4 matrix multiplication assignment", "matrix")
{
    omath::Matrix<float, 4, 4> matrix1{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F
    };

    const omath::Matrix<float, 4, 4> matrix2{
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F
    };

    matrix1 *= matrix2;

    REQUIRE(matrix1 == omath::Matrix<float, 4, 4>{
        34.0F, 44.0F, 54.0F, 64.0F,
        82.0F, 108.0F, 134.0F, 160.0F,
        34.0F, 44.0F, 54.0F, 64.0F,
        82.0F, 108.0F, 134.0F, 160.0F
    });
}

TEST_CASE("4x4 double matrix multiplication assignment", "matrix")
{
    omath::Matrix<double, 4, 4> matrix1{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    };

    const omath::Matrix<double, 4, 4> matrix2{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    };

    matrix1 *= matrix2;

    REQUIRE(matrix1 == omath::Matrix<double, 4, 4>{
        34.0, 44.0, 54.0, 64.0,
        82.0, 108.0, 134.0, 160.0,
        34.0, 44.0, 54.0, 64.0,
        82.0, 108.0, 134.0, 160.0
    });
}

TEST_CASE("2D vector transformation by 3x3 matrix", "matrix")
{
    const omath::Matrix<float, 3> matrix{
        1.0F, 2.0F, 0.0F,
        1.0F, 1.0F, 0.0F,
        1.0F, 3.0F, 1.0F
    };

    const omath::Vector<float, 2> vector{1.0F, 2.0F};

    const omath::Vector<float, 2> result = vector * matrix;

    REQUIRE(result == omath::Vector<float, 2>{
        3.0F, 4.0F
    });
}

TEST_CASE("3D vector transformation by 3x3 matrix", "matrix")
{
    const omath::Matrix<float, 3> matrix{
        1.0F, 2.0F, 0.0F,
        1.0F, 1.0F, 0.0F,
        1.0F, 3.0F, 1.0F
    };

    const omath::Vector<float, 3> vector{1.0F, 2.0F, 1.0F};

    const omath::Vector<float, 3> result = vector * matrix;

    REQUIRE(result == omath::Vector<float, 3>{
        4.0F, 7.0F, 1.0F
    });
}

TEST_CASE("3D vector transformation by 4x4 matrix", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        1.0F, 2.0F, 0.0F, 0.0F,
        1.0F, 1.0F, 0.0F, 0.0F,
        1.0F, 3.0F, 1.0F, 0.0F,
        1.0F, 1.0F, 3.0F, 1.0F
    };

    const omath::Vector<float, 3> vector{1.0F, 2.0F, 3.0F};

    const omath::Vector<float, 3> result = vector * matrix;

    REQUIRE(result == omath::Vector<float, 3>{
        6.0F, 13.0F, 3.0F
    });
}

TEST_CASE("4D vector transformation by 4x4 matrix", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        1.0F, 2.0F, 0.0F, 0.0F,
        1.0F, 1.0F, 0.0F, 0.0F,
        1.0F, 3.0F, 1.0F, 0.0F,
        1.0F, 1.0F, 3.0F, 1.0F
    };

    const omath::Vector<float, 4> vector{1.0F, 2.0F, 3.0F, 1.0F};

    const omath::Vector<float, 4> result = vector * matrix;

    REQUIRE(result == omath::Vector<float, 4>{
        7.0F, 14.0F, 6.0F, 1.0F
    });
}

TEST_CASE("2D vector transformation assignment by 3x3 matrix", "matrix")
{
    const omath::Matrix<float, 3> matrix{
        1.0F, 2.0F, 0.0F,
        1.0F, 1.0F, 0.0F,
        1.0F, 3.0F, 1.0F
    };

    omath::Vector<float, 2> vector{1.0F, 2.0F};

    vector *= matrix;

    REQUIRE(vector == omath::Vector<float, 2>{
        3.0F, 4.0F
    });
}

TEST_CASE("3D vector transformation assignment by 3x3 matrix", "matrix")
{
    const omath::Matrix<float, 3> matrix{
        1.0F, 2.0F, 0.0F,
        1.0F, 1.0F, 0.0F,
        1.0F, 3.0F, 1.0F
    };

    omath::Vector<float, 3> vector{1.0F, 2.0F, 1.0F};

    vector *= matrix;

    REQUIRE(vector == omath::Vector<float, 3>{
        4.0F, 7.0F, 1.0F
    });
}

TEST_CASE("3D vector transformation assignment by 4x4 matrix", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        1.0F, 2.0F, 0.0F, 0.0F,
        1.0F, 1.0F, 0.0F, 0.0F,
        1.0F, 3.0F, 1.0F, 0.0F,
        1.0F, 1.0F, 3.0F, 1.0F
    };

    omath::Vector<float, 3> vector{1.0F, 2.0F, 3.0F};

    vector *= matrix;

    REQUIRE(vector == omath::Vector<float, 3>{
        6.0F, 13.0F, 3.0F
    });
}

TEST_CASE("4D vector transformation assignment by 4x4 matrix", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        1.0F, 2.0F, 0.0F, 0.0F,
        1.0F, 1.0F, 0.0F, 0.0F,
        1.0F, 3.0F, 1.0F, 0.0F,
        1.0F, 1.0F, 3.0F, 1.0F
    };

    omath::Vector<float, 4> vector{1.0F, 2.0F, 3.0F, 1.0F};

    vector *= matrix;

    REQUIRE(vector == omath::Vector<float, 4>{
        7.0F, 14.0F, 6.0F, 1.0F
    });
}

TEST_CASE("1x1 matrix inversion", "matrix")
{
    omath::Matrix<float, 1, 1> matrix{
        2.0F
    };

    invert(matrix);

    REQUIRE(matrix == omath::Matrix<float, 1, 1>{
        0.5F
    });
}

TEST_CASE("2x2 matrix inversion", "matrix")
{
    omath::Matrix<float, 2, 2> matrix{
        2.0F, 3.0F,
        4.0F, 5.0F
    };

    invert(matrix);

    REQUIRE(matrix[0][0] == Approx(-5.0F / 2.0F));
    REQUIRE(matrix[0][1] == Approx(3.0F / 2.0F));
    REQUIRE(matrix[1][0] == Approx(2.0F));
    REQUIRE(matrix[1][1] == Approx(-1.0F));
}

TEST_CASE("3x3 matrix inversion", "matrix")
{
    omath::Matrix<float, 3, 3> matrix{
        2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F,
        8.0F, 9.0F, 0.0F
    };

    invert(matrix);

    REQUIRE(matrix[0][0] == Approx(-21.0F / 10.0F));
    REQUIRE(matrix[0][1] == Approx(6.0F / 5.0F));
    REQUIRE(matrix[0][2] == Approx(-1.0F / 10.0F));

    REQUIRE(matrix[1][0] == Approx(28.0F / 15.0F));
    REQUIRE(matrix[1][1] == Approx(-16.0F / 15.0F));
    REQUIRE(matrix[1][2] == Approx(1.0F / 5.0F));

    REQUIRE(matrix[2][0] == Approx(-1.0F / 10.0F));
    REQUIRE(matrix[2][1] == Approx(1.0F / 5.0F));
    REQUIRE(matrix[2][2] == Approx(-1.0F / 10.0F));
}

TEST_CASE("4x4 matrix inversion", "matrix")
{
    omath::Matrix<float, 4, 4> matrix{
        2.0F, 3.0F, 4.0F, 0.0F,
        5.0F, 6.0F, 7.0F, 0.0F,
        8.0F, 9.0F, 1.0F, 0.0F,
        2.0F, 3.0F, 4.0F, 1.0F
    };

    invert(matrix);

    REQUIRE(matrix[0][0] == Approx(-19.0F / 9.0F));
    REQUIRE(matrix[0][1] == Approx(11.0F / 9.0F));
    REQUIRE(matrix[0][2] == Approx(-1.0F / 9.0F));
    REQUIRE(matrix[0][3] == Approx(0.0F));

    REQUIRE(matrix[1][0] == Approx(17.0F / 9.0F));
    REQUIRE(matrix[1][1] == Approx(-10.0F / 9.0F));
    REQUIRE(matrix[1][2] == Approx(2.0F / 9.0F));
    REQUIRE(matrix[1][3] == Approx(0.0F));

    REQUIRE(matrix[2][0] == Approx(-1.0F / 9.0F));
    REQUIRE(matrix[2][1] == Approx(2.0F / 9.0F));
    REQUIRE(matrix[2][2] == Approx(-1.0F / 9.0F));
    REQUIRE(matrix[2][3] == Approx(0.0F));

    REQUIRE(matrix[3][0] == Approx(-1.0F));
    REQUIRE(matrix[3][1] == Approx(0.0F));
    REQUIRE(matrix[3][2] == Approx(0.0F));
    REQUIRE(matrix[3][3] == Approx(1.0F));
}

TEST_CASE("1x1 matrix inverse", "matrix")
{
    const omath::Matrix<float, 1, 1> matrix{
        2.0F
    };

    const omath::Matrix<float, 1, 1> result = inverse(matrix);

    REQUIRE(result == omath::Matrix<float, 1, 1>{
        0.5F
    });
}

TEST_CASE("2x2 matrix inverse", "matrix")
{
    const omath::Matrix<float, 2, 2> matrix{
        2.0F, 3.0F,
        4.0F, 5.0F
    };

    const omath::Matrix<float, 2, 2> result = inverse(matrix);

    REQUIRE(result[0][0] == Approx(-5.0F / 2.0F));
    REQUIRE(result[0][1] == Approx(3.0F / 2.0F));
    REQUIRE(result[1][0] == Approx(2.0F));
    REQUIRE(result[1][1] == Approx(-1.0F));
}

TEST_CASE("3x3 matrix inverse", "matrix")
{
    const omath::Matrix<float, 3, 3> matrix{
        2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F,
        8.0F, 9.0F, 0.0F
    };

    const omath::Matrix<float, 3, 3> result = inverse(matrix);

    REQUIRE(result[0][0] == Approx(-21.0F / 10.0F));
    REQUIRE(result[0][1] == Approx(6.0F / 5.0F));
    REQUIRE(result[0][2] == Approx(-1.0F / 10.0F));

    REQUIRE(result[1][0] == Approx(28.0F / 15.0F));
    REQUIRE(result[1][1] == Approx(-16.0F / 15.0F));
    REQUIRE(result[1][2] == Approx(1.0F / 5.0F));

    REQUIRE(result[2][0] == Approx(-1.0F / 10.0F));
    REQUIRE(result[2][1] == Approx(1.0F / 5.0F));
    REQUIRE(result[2][2] == Approx(-1.0F / 10.0F));
}

TEST_CASE("4x4 matrix inverse", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        2.0F, 3.0F, 4.0F, 0.0F,
        5.0F, 6.0F, 7.0F, 0.0F,
        8.0F, 9.0F, 1.0F, 0.0F,
        2.0F, 3.0F, 4.0F, 1.0F
    };

    const omath::Matrix<float, 4, 4> result = inverse(matrix);

    REQUIRE(result[0][0] == Approx(-19.0F / 9.0F));
    REQUIRE(result[0][1] == Approx(11.0F / 9.0F));
    REQUIRE(result[0][2] == Approx(-1.0F / 9.0F));
    REQUIRE(result[0][3] == Approx(0.0F));

    REQUIRE(result[1][0] == Approx(17.0F / 9.0F));
    REQUIRE(result[1][1] == Approx(-10.0F / 9.0F));
    REQUIRE(result[1][2] == Approx(2.0F / 9.0F));
    REQUIRE(result[1][3] == Approx(0.0F));

    REQUIRE(result[2][0] == Approx(-1.0F / 9.0F));
    REQUIRE(result[2][1] == Approx(2.0F / 9.0F));
    REQUIRE(result[2][2] == Approx(-1.0F / 9.0F));
    REQUIRE(result[2][3] == Approx(0.0F));

    REQUIRE(result[3][0] == Approx(-1.0F));
    REQUIRE(result[3][1] == Approx(0.0F));
    REQUIRE(result[3][2] == Approx(0.0F));
    REQUIRE(result[3][3] == Approx(1.0F));
}
