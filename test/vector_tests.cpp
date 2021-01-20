#include "catch2/catch.hpp"
#include "Vector.hpp"

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

TEST_CASE("Vector accessors", "vector")
{
    const math::Vector<float, 4> vector{
        1.0F, 2.0F, 3.0F, 4.0F
    };

    REQUIRE(vector.x() == 1.0F);
    REQUIRE(vector.y() == 2.0F);
    REQUIRE(vector.z() == 3.0F);
    REQUIRE(vector.w() == 4.0F);
}
