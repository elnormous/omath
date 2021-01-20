#include "catch2/catch.hpp"
#include "Matrix.hpp"

TEST_CASE("Matrix zero initalization", "matrix")
{
    const math::Matrix<float, 4> matrix;
    for (std::size_t i = 0; i < 16; ++i)
        REQUIRE(matrix[i] == 0.0F);
}

TEST_CASE("Matrix value initalization", "matrix")
{
    const math::Matrix<float, 2, 2> matrix{
        0.0F, 1.0F,
        2.0F, 3.0F
    };

    REQUIRE(matrix[0] == 0.0F);
    REQUIRE(matrix[1] == 1.0F);
    REQUIRE(matrix[2] == 2.0F);
    REQUIRE(matrix[3] == 3.0F);
}

TEST_CASE("Matrix identity", "matrix")
{
    const auto matrix = math::Matrix<float, 2>::identity();

    REQUIRE(matrix[0] == 1.0F);
    REQUIRE(matrix[1] == 0.0F);
    REQUIRE(matrix[2] == 0.0F);
    REQUIRE(matrix[3] == 1.0F);
}
