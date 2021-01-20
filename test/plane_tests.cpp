#include "catch2/catch.hpp"
#include "Plane.hpp"

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

TEST_CASE("Plane accessors", "plane")
{
    const math::Plane<float> plane{
        1.0F, 2.0F, 3.0F, 4.0F
    };

    REQUIRE(plane.a() == 0.0F);
    REQUIRE(plane.b() == 1.0F);
    REQUIRE(plane.c() == 2.0F);
    REQUIRE(plane.d() == 3.0F);
}
