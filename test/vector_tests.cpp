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

TEST_CASE("Vector arithmetic operators", "vector")
{
    const math::Vector<float, 2> v1{2.0F, 4.0F};
    const math::Vector<float, 2> v2{2.0F, 5.0F};

    SECTION("Negate")
    {
        const auto result = -v1;
        REQUIRE(result.x() == -2.0F);
        REQUIRE(result.y() == -4.0F);
    }

    SECTION("Add")
    {
        const auto result = v1 + v2;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 9.0F);
    }

    SECTION("Subtract")
    {
        const auto result = v1 - v2;
        REQUIRE(result.x() == -0.0F);
        REQUIRE(result.y() == -1.0F);
    }

    SECTION("Multiply")
    {
        const auto result = v1 * 2.0F;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 8.0F);
    }

    SECTION("Divide")
    {
        const auto result = v1 / 2.0F;
        REQUIRE(result.x() == 1.0F);
        REQUIRE(result.y() == 2.0F);
    }

    SECTION("Add assign")
    {
        math::Vector<float, 2> result{2.0F, 4.0F};
        result += v2;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 9.0F);
    }

    SECTION("Subtract assign")
    {
        math::Vector<float, 2> result{2.0F, 4.0F};
        result -= v2;
        REQUIRE(result.x() == -0.0F);
        REQUIRE(result.y() == -1.0F);
    }

    SECTION("Multiply assign")
    {
        math::Vector<float, 2> result{2.0F, 4.0F};
        result *= 2.0F;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 8.0F);
    }

    SECTION("Divide assign")
    {
        math::Vector<float, 2> result{2.0F, 4.0F};
        result /= 2.0F;
        REQUIRE(result.x() == 1.0F);
        REQUIRE(result.y() == 2.0F);
    }
}

TEST_CASE("Vector comparison operators", "vector")
{
    const math::Vector<float, 2> v1{2.0F, 4.0F};
    const math::Vector<float, 2> v2{2.0F, 5.0F};
    const math::Vector<float, 2> v3{2.0F, 4.0F};

    SECTION("Less than")
    {
        REQUIRE(v1 < v2);
        REQUIRE_FALSE(v1 < v3);
    }

    SECTION("Greater than")
    {
        REQUIRE_FALSE(v1 > v2);
        REQUIRE_FALSE(v1 > v3);
    }

    SECTION("Equal")
    {
        REQUIRE_FALSE(v1 == v2);
        REQUIRE(v1 == v3);
    }

    SECTION("Not equal")
    {
        REQUIRE(v1 != v2);
        REQUIRE_FALSE(v1 != v3);
    }
}
