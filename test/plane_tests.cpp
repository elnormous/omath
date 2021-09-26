#include "catch2/catch.hpp"
#include "Plane.hpp"

TEST_CASE("Plane zero initalization", "plane")
{
    const omath::Plane<float> plane{};
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(plane[i] == 0.0F);
}

TEST_CASE("Plane value initalization", "plane")
{
    const omath::Plane<float> plane{
        0.0F, 1.0F, 2.0F, 3.0F
    };

    REQUIRE(plane[0] == 0.0F);
    REQUIRE(plane[1] == 1.0F);
    REQUIRE(plane[2] == 2.0F);
    REQUIRE(plane[3] == 3.0F);
}

TEST_CASE("Plane accessors", "plane")
{
    const omath::Plane<float> plane{
        1.0F, 2.0F, 3.0F, 4.0F
    };

    REQUIRE(plane.a() == 1.0F);
    REQUIRE(plane.b() == 2.0F);
    REQUIRE(plane.c() == 3.0F);
    REQUIRE(plane.d() == 4.0F);
}

TEST_CASE("Plane comparison", "plane")
{
    const omath::Plane<float> plane1{
        0.0F, 1.0F, 2.0F, 3.0F
    };

    const omath::Plane<float> plane2{
        0.0F, 1.0F,
        2.0F, 3.0F
    };

    const omath::Plane<float> plane3{
        1.0F, 2.0F, 3.0F, 4.0F
    };

    REQUIRE(plane1 == plane2);
    REQUIRE(plane1 != plane3);
}

TEST_CASE("Plane flip", "plane")
{
    const omath::Plane<float> plane{
        1.0F, 2.0F, 3.0F, 4.0F
    };

    const auto flippedPlane = -plane;

    REQUIRE(flippedPlane.a() == -1.0F);
    REQUIRE(flippedPlane.b() == -2.0F);
    REQUIRE(flippedPlane.c() == -3.0F);
    REQUIRE(flippedPlane.d() == -4.0F);
}

TEST_CASE("Vector dot", "plane")
{
    const omath::Plane<float> plane{
        2.0F, -2.0F, 5.0F, 8.0F
    };

    const omath::Vector<float, 3> vector{
        4.0F, -4.0F, 3.0F
    };

    const auto result = dot(plane, vector);

    REQUIRE(result == 39.0F);
}

TEST_CASE("Vector flip", "plane")
{
    SECTION("First")
    {
        const omath::Plane<float> plane{
            2.0F, -2.0F, 5.0F, 8.0F
        };

        const omath::Vector<float, 3> vector{
            4.0F, -4.0F, 3.0F
        };

        const auto result = distance(plane, vector);

        REQUIRE(result == Approx(6.78902858227F));
    }

    SECTION("Second")
    {
        const omath::Plane<float> plane{
            1.0F, -2.0F, -2.0F, -1.0F
        };

        const omath::Vector<float, 3> vector{
            2.0F, 8.0F, 5.0F
        };

        const auto result = distance(plane, vector);

        REQUIRE(result == Approx(8.33333333333F));
    }
}
