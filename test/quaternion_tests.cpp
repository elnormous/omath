#include "catch2/catch.hpp"
#include "Quaternion.hpp"

TEST_CASE("Quaternion zero initalization", "quaternion")
{
    const omath::Quaternion<float> quaternion{};
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(quaternion[i] == 0.0F);
}

TEST_CASE("Quaternion value initalization", "quaternion")
{
    const omath::Quaternion<float> quaternion{
        0.0F, 1.0F, 2.0F, 3.0F
    };

    REQUIRE(quaternion[0] == 0.0F);
    REQUIRE(quaternion[1] == 1.0F);
    REQUIRE(quaternion[2] == 2.0F);
    REQUIRE(quaternion[3] == 3.0F);
}

TEST_CASE("Quaternion accessors", "quaternion")
{
    const omath::Quaternion<float> quaternion{
        1.0F, 2.0F, 3.0F, 4.0F
    };

    REQUIRE(quaternion.x() == 1.0F);
    REQUIRE(quaternion.y() == 2.0F);
    REQUIRE(quaternion.z() == 3.0F);
    REQUIRE(quaternion.w() == 4.0F);
}

TEST_CASE("Quaternion identity", "quaternion")
{
    const auto quaternion = omath::identityQuaternion<float>();

    REQUIRE(quaternion[0] == 0.0F);
    REQUIRE(quaternion[1] == 0.0F);
    REQUIRE(quaternion[2] == 0.0F);
    REQUIRE(quaternion[3] == 1.0F);
}

TEST_CASE("Quaternion comparison", "quaternion")
{
    const omath::Quaternion<float> quaternion1{
        0.0F, 1.0F, 2.0F, 3.0F
    };

    const omath::Quaternion<float> quaternion2{
        0.0F, 1.0F,
        2.0F, 3.0F
    };

    const omath::Quaternion<float> quaternion3{
        1.0F, 2.0F, 3.0F, 4.0F
    };

    REQUIRE(quaternion1 == quaternion2);
    REQUIRE(quaternion1 != quaternion3);
}


TEST_CASE("Quaternion arithmetic operators", "vector")
{
    const omath::Quaternion<float> quaternion1{2.0F, 4.0F, -6.0F, 6.0F};
    const omath::Quaternion<float> quaternion2{2.0F, 5.0F, 6.0F, 6.0F};

    SECTION("Negative")
    {
        const auto result = -quaternion1;
        REQUIRE(result == omath::Quaternion<float>{-2.0F, -4.0F, 6.0F, -6.0F});
    }

    SECTION("Negation")
    {
        omath::Quaternion<float> result{2.0F, 4.0F, -6.0F, 6.0F};
        negate(result);
        REQUIRE(result == omath::Quaternion<float>{-2.0F, -4.0F, 6.0F, -6.0F});
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
        REQUIRE(result == omath::Quaternion<float>{-0.0F, -1.0F, -12.0F, 0.0F});
    }

    SECTION("Scalar multiply")
    {
        const auto result = quaternion1 * 2.0F;
        REQUIRE(result == omath::Quaternion<float>{4.0F, 8.0F, -12.0F, 12.0F});
    }

    SECTION("Quaternion multiply")
    {
        const auto result = quaternion1 * quaternion2;
        REQUIRE(result == omath::Quaternion<float>{78.0F, 30.0F, 2.0F, 48.0F});
    }

    SECTION("Divide")
    {
        const auto result = quaternion1 / 2.0F;
        REQUIRE(result == omath::Quaternion<float>{1.0F, 2.0F, -3.0F, 3.0F});
    }

    SECTION("Add assign")
    {
        omath::Quaternion<float> result{2.0F, 4.0F, -6.0F, 6.0F};
        result += quaternion2;
        REQUIRE(result == omath::Quaternion<float>{4.0F, 9.0F, 0.0F, 12.0F});
    }

    SECTION("Subtract assign")
    {
        omath::Quaternion<float> result{2.0F, 4.0F, -6.0F, 6.0F};
        result -= quaternion2;
        REQUIRE(result == omath::Quaternion<float>{-0.0F, -1.0F, -12.0F, 0.0F});
    }

    SECTION("Scalar multiply assign")
    {
        omath::Quaternion<float> result{2.0F, 4.0F, -6.0F, 6.0F};
        result *= 2.0F;
        REQUIRE(result == omath::Quaternion<float>{4.0F, 8.0F, -12.0F, 12.0F});
    }

    SECTION("Quaternion multiply assign")
    {
        omath::Quaternion<float> result{2.0F, 4.0F, -6.0F, 6.0F};
        result *= quaternion2;
        REQUIRE(result == omath::Quaternion<float>{78.0F, 30.0F, 2.0F, 48.0F});
    }

    SECTION("Divide assign")
    {
        omath::Quaternion<float> result{2.0F, 4.0F, -6.0F, 6.0F};
        result /= 2.0F;
        REQUIRE(result == omath::Quaternion<float>{1.0F, 2.0F, -3.0F, 3.0F});
    }
}
