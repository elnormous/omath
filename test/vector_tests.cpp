#include "catch2/catch.hpp"
#include "Vector.hpp"

TEST_CASE("Vector zero initalization", "vector")
{
    const omath::Vector<float, 4> vector{};
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(vector[i] == 0.0F);
}

TEST_CASE("Vector value initalization", "vector")
{
    const omath::Vector<float, 2> vector{
        0.0F, 1.0F
    };

    REQUIRE(vector[0] == 0.0F);
    REQUIRE(vector[1] == 1.0F);
}

TEST_CASE("Vector accessors", "vector")
{
    const omath::Vector<float, 4> vector{
        1.0F, 2.0F, 3.0F, 4.0F
    };

    REQUIRE(vector.x() == 1.0F);
    REQUIRE(vector.y() == 2.0F);
    REQUIRE(vector.z() == 3.0F);
    REQUIRE(vector.w() == 4.0F);
}

TEST_CASE("Vector comparison operators", "vector")
{
    const omath::Vector<float, 2> vector1{2.0F, 4.0F};
    const omath::Vector<float, 2> vector2{2.0F, 5.0F};
    const omath::Vector<float, 2> vector3{2.0F, 4.0F};

    SECTION("Less than")
    {
        REQUIRE(vector1 < vector2);
        REQUIRE_FALSE(vector1 < vector3);
    }

    SECTION("Greater than")
    {
        REQUIRE_FALSE(vector1 > vector2);
        REQUIRE_FALSE(vector1 > vector3);
    }

    SECTION("Equal")
    {
        REQUIRE_FALSE(vector1 == vector2);
        REQUIRE(vector1 == vector3);
    }

    SECTION("Not equal")
    {
        REQUIRE(vector1 != vector2);
        REQUIRE_FALSE(vector1 != vector3);
    }
}

TEST_CASE("Vector arithmetic operators", "vector")
{
    const omath::Vector<float, 2> vector1{2.0F, 4.0F};
    const omath::Vector<float, 2> vector2{2.0F, 5.0F};

    SECTION("Negate")
    {
        const auto result = -vector1;
        REQUIRE(result.x() == -2.0F);
        REQUIRE(result.y() == -4.0F);
    }

    SECTION("Add")
    {
        const auto result = vector1 + vector2;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 9.0F);
    }

    SECTION("Subtract")
    {
        const auto result = vector1 - vector2;
        REQUIRE(result.x() == -0.0F);
        REQUIRE(result.y() == -1.0F);
    }

    SECTION("Multiply")
    {
        const auto result = vector1 * 2.0F;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 8.0F);
    }

    SECTION("Divide")
    {
        const auto result = vector1 / 2.0F;
        REQUIRE(result.x() == 1.0F);
        REQUIRE(result.y() == 2.0F);
    }

    SECTION("Add assign")
    {
        omath::Vector<float, 2> result{2.0F, 4.0F};
        result += vector2;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 9.0F);
    }

    SECTION("Subtract assign")
    {
        omath::Vector<float, 2> result{2.0F, 4.0F};
        result -= vector2;
        REQUIRE(result.x() == -0.0F);
        REQUIRE(result.y() == -1.0F);
    }

    SECTION("Multiply assign")
    {
        omath::Vector<float, 2> result{2.0F, 4.0F};
        result *= 2.0F;
        REQUIRE(result.x() == 4.0F);
        REQUIRE(result.y() == 8.0F);
    }

    SECTION("Divide assign")
    {
        omath::Vector<float, 2> result{2.0F, 4.0F};
        result /= 2.0F;
        REQUIRE(result.x() == 1.0F);
        REQUIRE(result.y() == 2.0F);
    }
}

TEST_CASE("Vector length", "vector")
{
    const omath::Vector<float, 2> vector{3.0F, 4.0F};
    const auto result = vector.length();

    REQUIRE(result == Approx(5.0F));
}

TEST_CASE("Vector length squared", "vector")
{
    const omath::Vector<float, 2> vector{3.0F, 4.0F};
    const auto result = vector.lengthSquared();

    REQUIRE(result == Approx(25.0F));
}

TEST_CASE("Vector cross product", "vector")
{
    const omath::Vector<float, 3> vector1{2.0F, 3.0F, 4.0F};
    const omath::Vector<float, 3> vector2{5.0F, 6.0F, 7.0F};
    const auto result = vector1.cross(vector2);

    REQUIRE(result.x() == -3.0F);
    REQUIRE(result.y() == 6.0F);
    REQUIRE(result.z() == -3.0F);
}

TEST_CASE("Vector dot product", "vector")
{
    const omath::Vector<float, 2> vector1{1.0F, 2.0F};
    const omath::Vector<float, 2> vector2{4.0F, 5.0F};
    const auto result = vector1.dot(vector2);

    REQUIRE(result == 14.0F);
}

TEST_CASE("Vector distance", "vector")
{
    const omath::Vector<float, 2> vector1{1.0F, 2.0F};
    const omath::Vector<float, 2> vector2{4.0F, 6.0F};
    const auto result = vector1.distance(vector2);

    REQUIRE(result == 5.0F);
}

TEST_CASE("Vector distance squared", "vector")
{
    const omath::Vector<float, 2> vector1{1.0F, 2.0F};
    const omath::Vector<float, 2> vector2{4.0F, 6.0F};
    const auto result = vector1.distanceSquared(vector2);

    REQUIRE(result == 25.0F);
}
