#include "catch2/catch.hpp"
#include "Matrix.hpp"

TEST_CASE("4x4 matrix transpose using SIMD", "matrix")
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

TEST_CASE("4x4 matrix transposed using SIMD", "matrix")
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

TEST_CASE("4x4 matrix negation using SIMD", "matrix")
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

TEST_CASE("4x4 matrix sum using SIMD", "matrix")
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

TEST_CASE("4x4 matrix increment using SIMD", "matrix")
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

TEST_CASE("4x4 matrix difference using SIMD", "matrix")
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

TEST_CASE("4x4 matrix decrement using SIMD", "matrix")
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

TEST_CASE("4x4 matrix multiplication with scalar using SIMD", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        2.0F, -3.0F, 2.0F, -3.0F
    };

    const omath::Matrix<float, 4, 4> result = matrix * 2.0F;

    REQUIRE(result == omath::Matrix<float, 4, 4>{
        0.0F, 2.0F, 0.0F, 2.0F,
        4.0F, -6.0F, 4.0F, -6.0F,
        0.0F, 2.0F, 0.0F, 2.0F,
        4.0F, -6.0F, 4.0F, -6.0F
    });
}

TEST_CASE("4x4 matrix divison with scalar using SIMD", "matrix")
{
    const omath::Matrix<float, 4, 4> matrix{
        0.0F, 2.0F, 0.0F, 2.0F,
        2.0F, -4.0F, 2.0F, -4.0F,
        0.0F, 2.0F, 0.0F, 2.0F,
        2.0F, -4.0F, 2.0F, -4.0F
    };

    const omath::Matrix<float, 4, 4> result = matrix / 2.0F;

    REQUIRE(result == omath::Matrix<float, 4, 4>{
        0.0F, 1.0F, 0.0F, 1.0F,
        1.0F, -2.0F, 1.0F, -2.0F,
        0.0F, 1.0F, 0.0F, 1.0F,
        1.0F, -2.0F, 1.0F, -2.0F
    });
}

TEST_CASE("4x4 matrix multiplication assignment with scalar using SIMD", "matrix")
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

TEST_CASE("4x4 matrix division assignment with scalar using SIMD", "matrix")
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

TEST_CASE("4x4 matrix multiplication using SIMD", "matrix")
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

TEST_CASE("4x4 SIMD matrix multiplication with 4x1 matrix", "matrix")
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

TEST_CASE("4x4 matrix multiplication assignment using SIMD", "matrix")
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

TEST_CASE("4D vector transformation by 4x4 matrix using SIMD", "matrix")
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

TEST_CASE("4D vector transformation assignment by 4x4 matrix using SIMD", "matrix")
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
