#include "catch2/catch.hpp"
#include "Matrix.hpp"
#include "Plane.hpp"
#include "Quaternion.hpp"
#include "Vector.hpp"

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

TEST_CASE("Vector zero initalization", "vector")
{
    const math::Vector<float, 4> vector;
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(vector[i] == 0.0F);
}

TEST_CASE("Vector value initalization", "vector")
{
    const math::Vector<float, 2> vector{
        0.0F, 1.0F
    };

    REQUIRE(vector[0] == 0.0F);
    REQUIRE(vector[1] == 1.0F);
}

TEST_CASE("Quaternion zero initalization", "quaternion")
{
    const math::Quaternion<float> quaternion;
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(quaternion[i] == 0.0F);
}

TEST_CASE("Quaternion value initalization", "quaternion")
{
    const math::Quaternion<float> quaternion{
        0.0F, 1.0F, 2.0F, 3.0F
    };

    REQUIRE(quaternion[0] == 0.0F);
    REQUIRE(quaternion[1] == 1.0F);
    REQUIRE(quaternion[2] == 2.0F);
    REQUIRE(quaternion[3] == 3.0F);
}

TEST_CASE("Plane zero initalization", "plane")
{
    const math::Plane<float> plane;
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(plane[i] == 0.0F);
}

TEST_CASE("Plane value initalization", "plane")
{
    const math::Plane<float> plane{
        0.0F, 1.0F, 2.0F, 3.0F
    };

    REQUIRE(plane[0] == 0.0F);
    REQUIRE(plane[1] == 1.0F);
    REQUIRE(plane[2] == 2.0F);
    REQUIRE(plane[3] == 3.0F);
}
