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

    REQUIRE(quaternion == omath::Quaternion<float>{0.0F, 1.0F, 2.0F, 3.0F});
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
    REQUIRE(quaternion[0] == 1.0F);
    REQUIRE(quaternion[1] == 2.0F);
    REQUIRE(quaternion[2] == 3.0F);
    REQUIRE(quaternion[3] == 4.0F);
    REQUIRE(quaternion(0) == 1.0F);
    REQUIRE(quaternion(1) == 2.0F);
    REQUIRE(quaternion(2) == 3.0F);
    REQUIRE(quaternion(3) == 4.0F);
}

TEST_CASE("Quaternion element assignment", "matrix")
{
    omath::Quaternion<float> quaternion;
    quaternion[0] = 0.0F;
    quaternion[1] = 1.0F;
    quaternion[2] = 2.0F;
    quaternion[3] = 3.0F;

    REQUIRE(quaternion == omath::Quaternion<float>{0.0F, 1.0F, 2.0F, 3.0F});

    quaternion(0) = 4.0F;
    quaternion(1) = 5.0F;
    quaternion(2) = 6.0F;
    quaternion(3) = 7.0F;

    REQUIRE(quaternion == omath::Quaternion<float>{4.0F, 5.0F, 6.0F, 7.0F});
}

TEST_CASE("Quaternion identity", "quaternion")
{
    const auto quaternion = omath::identityQuaternion<float>;

    REQUIRE(quaternion == omath::Quaternion<float>{0.0F, 0.0F, 0.0F, 1.0F});
}

TEST_CASE("Quaternion set identity", "quaternion")
{
    omath::Quaternion<float> quaternion{};
    setIdentity(quaternion);

    REQUIRE(quaternion == omath::Quaternion<float>{0.0F, 0.0F, 0.0F, 1.0F});
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
        omath::Quaternion<float> result = quaternion1;
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
        omath::Quaternion<float> result = quaternion1;
        result += quaternion2;
        REQUIRE(result == omath::Quaternion<float>{4.0F, 9.0F, 0.0F, 12.0F});
    }

    SECTION("Subtract assign")
    {
        omath::Quaternion<float> result = quaternion1;
        result -= quaternion2;
        REQUIRE(result == omath::Quaternion<float>{-0.0F, -1.0F, -12.0F, 0.0F});
    }

    SECTION("Scalar multiply assign")
    {
        omath::Quaternion<float> result = quaternion1;
        result *= 2.0F;
        REQUIRE(result == omath::Quaternion<float>{4.0F, 8.0F, -12.0F, 12.0F});
    }

    SECTION("Quaternion multiply assign")
    {
        omath::Quaternion<float> result = quaternion1;
        result *= quaternion2;
        REQUIRE(result == omath::Quaternion<float>{78.0F, 30.0F, 2.0F, 48.0F});
    }

    SECTION("Divide assign")
    {
        omath::Quaternion<float> result = quaternion1;
        result /= 2.0F;
        REQUIRE(result == omath::Quaternion<float>{1.0F, 2.0F, -3.0F, 3.0F});
    }

    SECTION("Norm")
    {
        const float result = norm(quaternion1);
        REQUIRE(result == Approx(9.591663046625438f));
    }

    SECTION("Invert")
    {
        omath::Quaternion<float> result = quaternion1;
        invert(result);
        REQUIRE(result[0] == Approx(-0.0217391F));
        REQUIRE(result[1] == Approx(-0.0434783F));
        REQUIRE(result[2] == Approx(0.0652174F));
        REQUIRE(result[3] == Approx(0.0652174F));
    }

    SECTION("Inverse")
    {
        const omath::Quaternion<float> result = inverse(quaternion1);
        REQUIRE(result[0] == Approx(-0.0217391F));
        REQUIRE(result[1] == Approx(-0.0434783F));
        REQUIRE(result[2] == Approx(0.0652174F));
        REQUIRE(result[3] == Approx(0.0652174F));
    }

    SECTION("Conjugation")
    {
        const omath::Quaternion<float> result = conjugated(quaternion1);
        REQUIRE(result == omath::Quaternion<float>{-2.0F, -4.0F, 6.0F, 6.0F});
    }

    SECTION("Conjugate")
    {
        omath::Quaternion<float> result = quaternion1;
        conjugate(result);
        REQUIRE(result == omath::Quaternion<float>{-2.0F, -4.0F, 6.0F, 6.0F});
    }
}

TEST_CASE("Double-precision quaternion arithmetic operators", "vector")
{
    const omath::Quaternion<double> quaternion1{2.0, 4.0, -6.0, 6.0};
    const omath::Quaternion<double> quaternion2{2.0, 5.0, 6.0, 6.0};

    SECTION("Negative")
    {
        const auto result = -quaternion1;
        REQUIRE(result == omath::Quaternion<double>{-2.0, -4.0, 6.0, -6.0});
    }

    SECTION("Negation")
    {
        omath::Quaternion<double> result = quaternion1;
        negate(result);
        REQUIRE(result == omath::Quaternion<double>{-2.0, -4.0, 6.0, -6.0});
    }

    SECTION("Add")
    {
        const auto result = quaternion1 + quaternion2;
        REQUIRE(result == omath::Quaternion<double>{4.0, 9.0, 0.0, 12.0});
    }

    SECTION("Subtract")
    {
        const auto result = quaternion1 - quaternion2;
        REQUIRE(result == omath::Quaternion<double>{-0.0, -1.0, -12.0, 0.0});
    }

    SECTION("Scalar multiply")
    {
        const auto result = quaternion1 * 2.0;
        REQUIRE(result == omath::Quaternion<double>{4.0, 8.0, -12.0, 12.0});
    }

    SECTION("Quaternion multiply")
    {
        const auto result = quaternion1 * quaternion2;
        REQUIRE(result == omath::Quaternion<double>{78.0, 30.0, 2.0, 48.0});
    }

    SECTION("Divide")
    {
        const auto result = quaternion1 / 2.0;
        REQUIRE(result == omath::Quaternion<double>{1.0, 2.0, -3.0, 3.0});
    }

    SECTION("Add assign")
    {
        omath::Quaternion<double> result = quaternion1;
        result += quaternion2;
        REQUIRE(result == omath::Quaternion<double>{4.0, 9.0, 0.0, 12.0});
    }

    SECTION("Subtract assign")
    {
        omath::Quaternion<double> result = quaternion1;
        result -= quaternion2;
        REQUIRE(result == omath::Quaternion<double>{-0.0, -1.0, -12.0, 0.0});
    }

    SECTION("Scalar multiply assign")
    {
        omath::Quaternion<double> result = quaternion1;
        result *= 2.0;
        REQUIRE(result == omath::Quaternion<double>{4.0, 8.0, -12.0, 12.0});
    }

    SECTION("Quaternion multiply assign")
    {
        omath::Quaternion<double> result = quaternion1;
        result *= quaternion2;
        REQUIRE(result == omath::Quaternion<double>{78.0, 30.0, 2.0, 48.0});
    }

    SECTION("Divide assign")
    {
        omath::Quaternion<double> result = quaternion1;
        result /= 2.0;
        REQUIRE(result == omath::Quaternion<double>{1.0, 2.0, -3.0, 3.0});
    }

    SECTION("Norm")
    {
        const double result = norm(quaternion1);
        REQUIRE(result == Approx(9.591663046625438));
    }

    SECTION("Invert")
    {
        omath::Quaternion<double> result = quaternion1;
        invert(result);
        REQUIRE(result[0] == Approx(-0.0217391));
        REQUIRE(result[1] == Approx(-0.0434783));
        REQUIRE(result[2] == Approx(0.0652174));
        REQUIRE(result[3] == Approx(0.0652174));
    }

    SECTION("Inverse")
    {
        const omath::Quaternion<double> result = inverse(quaternion1);
        REQUIRE(result[0] == Approx(-0.0217391));
        REQUIRE(result[1] == Approx(-0.0434783));
        REQUIRE(result[2] == Approx(0.0652174));
        REQUIRE(result[3] == Approx(0.0652174));
    }

    SECTION("Conjugation")
    {
        const omath::Quaternion<double> result = conjugated(quaternion1);
        REQUIRE(result == omath::Quaternion<double>{-2.0, -4.0, 6.0, 6.0});
    }

    SECTION("Conjugate")
    {
        omath::Quaternion<double> result = quaternion1;
        conjugate(result);
        REQUIRE(result == omath::Quaternion<double>{-2.0, -4.0, 6.0, 6.0});
    }
}
