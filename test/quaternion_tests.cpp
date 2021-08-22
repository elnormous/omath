#include "catch2/catch.hpp"
#include "Quaternion.hpp"

TEST_CASE("Quaternion zero initalization", "quaternion")
{
    const math::Quaternion<float> quaternion{};
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

TEST_CASE("Quaternion comparison", "quaternion")
{
    const math::Quaternion<float> quaternion1{
        0.0F, 1.0F, 2.0F, 3.0F
    };

    const math::Quaternion<float> quaternion2{
        0.0F, 1.0F,
        2.0F, 3.0F
    };

    const math::Quaternion<float> quaternion3{
        1.0F, 2.0F, 3.0F, 4.0F
    };

    REQUIRE(quaternion1 == quaternion2);
    REQUIRE(quaternion1 != quaternion3);
}


TEST_CASE("Quaternion arithmetic operators", "vector")
{
    const math::Quaternion<float> quaternion1{2.0F, 4.0F, -6.0F, 6.0F};
    const math::Quaternion<float> quaternion2{2.0F, 5.0F, 6.0F, 6.0F};

    SECTION("Negate")
    {
        const auto result = -quaternion1;
        REQUIRE(result.x() == -2.0F);
        REQUIRE(result.y() == -4.0F);
        REQUIRE(result.z() == 6.0F);
        REQUIRE(result.w() == -6.0F);
    }

    SECTION("Add")
    {
        const auto result = quaternion1 + quaternion2;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 9.0F);
        REQUIRE(result.z() == 0.0F);
        REQUIRE(result.w() == 12.0F);
    }

    SECTION("Subtract")
    {
        const auto result = quaternion1 - quaternion2;
        REQUIRE(result.x() == -0.0F);
        REQUIRE(result.y() == -1.0F);
        REQUIRE(result.z() == -12.0F);
        REQUIRE(result.w() == 0.0F);
    }

    SECTION("Scalar multiply")
    {
        const auto result = quaternion1 * 2.0F;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 8.0F);
        REQUIRE(result.z() == -12.0F);
        REQUIRE(result.w() == 12.0F);
    }

    SECTION("Quaternion multiply")
    {
        const auto result = quaternion1 * quaternion2;
        REQUIRE(result.x() == 78.0F);
        REQUIRE(result.y() == 30.0F);
        REQUIRE(result.z() == 2.0F);
        REQUIRE(result.w() == 48.0F);
    }

    SECTION("Divide")
    {
        const auto result = quaternion1 / 2.0F;
        REQUIRE(result.x() == 1.0F);
        REQUIRE(result.y() == 2.0F);
        REQUIRE(result.z() == -3.0F);
        REQUIRE(result.w() == 3.0F);
    }

    SECTION("Add assign")
    {
        math::Quaternion<float> result{2.0F, 4.0F, -6.0F, 6.0F};
        result += quaternion2;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 9.0F);
        REQUIRE(result.z() == 0.0F);
        REQUIRE(result.w() == 12.0F);
    }

    SECTION("Subtract assign")
    {
        math::Quaternion<float> result{2.0F, 4.0F, -6.0F, 6.0F};
        result -= quaternion2;
        REQUIRE(result.x() == -0.0F);
        REQUIRE(result.y() == -1.0F);
        REQUIRE(result.z() == -12.0F);
        REQUIRE(result.w() == 0.0F);
    }

    SECTION("Scalar multiply assign")
    {
        math::Quaternion<float> result{2.0F, 4.0F, -6.0F, 6.0F};
        result *= 2.0F;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 8.0F);
        REQUIRE(result.z() == -12.0F);
        REQUIRE(result.w() == 12.0F);
    }

    SECTION("Quaternion multiply assign")
    {
        math::Quaternion<float> result{2.0F, 4.0F, -6.0F, 6.0F};
        result *= quaternion2;
        REQUIRE(result.x() == 78.0F);
        REQUIRE(result.y() == 30.0F);
        REQUIRE(result.z() == 2.0F);
        REQUIRE(result.w() == 48.0F);
    }

    SECTION("Divide assign")
    {
        math::Quaternion<float> result{2.0F, 4.0F, -6.0F, 6.0F};
        result /= 2.0F;
        REQUIRE(result.x() == 1.0F);
        REQUIRE(result.y() == 2.0F);
        REQUIRE(result.z() == -3.0F);
        REQUIRE(result.w() == 3.0F);
    }
}
