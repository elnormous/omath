#include "catch2/catch.hpp"
#include "Quaternion.hpp"

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

TEST_CASE("Quaternion accessors", "quaternion")
{
    const math::Quaternion<float> quaternion{
        1.0F, 2.0F, 3.0F, 4.0F
    };

    REQUIRE(quaternion.x() == 1.0F);
    REQUIRE(quaternion.y() == 2.0F);
    REQUIRE(quaternion.z() == 3.0F);
    REQUIRE(quaternion.w() == 4.0F);
}

TEST_CASE("Quaternion identity", "quaternion")
{
    const auto quaternion = math::Quaternion<float>::identity();

    REQUIRE(quaternion[0] == 0.0F);
    REQUIRE(quaternion[1] == 0.0F);
    REQUIRE(quaternion[2] == 0.0F);
    REQUIRE(quaternion[3] == 1.0F);
}
