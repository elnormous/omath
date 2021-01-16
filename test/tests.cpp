#include "catch2/catch.hpp"
#include "Matrix.hpp"

TEST_CASE("Matrix", "matrix")
{
    math::Matrix<float, 4, 4> matrix;
    for (std::size_t i = 0; i < 16; ++i)
        REQUIRE(matrix[i] == 0.0F);
}
