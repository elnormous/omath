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

    REQUIRE(vector == omath::Vector<float, 2>{0.0F, 1.0F});
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
        REQUIRE_FALSE(vector2 < vector1);
    }

    SECTION("Greater than")
    {
        REQUIRE_FALSE(vector1 > vector2);
        REQUIRE_FALSE(vector1 > vector3);
        REQUIRE(vector2 > vector1);
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

    SECTION("Negative")
    {
        const auto result = -vector1;
        REQUIRE(result == omath::Vector<float, 2>{-2.0F, -4.0F});
    }

    SECTION("Negation")
    {
        omath::Vector<float, 2> result{2.0F, 4.0F};
        negate(result);
        REQUIRE(result == omath::Vector<float, 2>{-2.0F, -4.0F});
    }

    SECTION("Add")
    {
        const auto result = vector1 + vector2;
        REQUIRE(result == omath::Vector<float, 2>{4.0F, 9.0F});
    }

    SECTION("Subtract")
    {
        const auto result = vector1 - vector2;
        REQUIRE(result == omath::Vector<float, 2>{-0.0F, -1.0F});
    }

    SECTION("Multiply")
    {
        const auto result = vector1 * 2.0F;
        REQUIRE(result == omath::Vector<float, 2>{4.0F, 8.0F});
    }

    SECTION("Divide")
    {
        const auto result = vector1 / 2.0F;
        REQUIRE(result == omath::Vector<float, 2>{1.0F, 2.0F});
    }

    SECTION("Add assign")
    {
        omath::Vector<float, 2> result{2.0F, 4.0F};
        result += vector2;
        REQUIRE(result == omath::Vector<float, 2>{4.0F, 9.0F});
    }

    SECTION("Subtract assign")
    {
        omath::Vector<float, 2> result{2.0F, 4.0F};
        result -= vector2;
        REQUIRE(result == omath::Vector<float, 2>{-0.0F, -1.0F});
    }

    SECTION("Multiply assign")
    {
        omath::Vector<float, 2> result{2.0F, 4.0F};
        result *= 2.0F;
        REQUIRE(result == omath::Vector<float, 2>{4.0F, 8.0F});
    }

    SECTION("Divide assign")
    {
        omath::Vector<float, 2> result{2.0F, 4.0F};
        result /= 2.0F;
        REQUIRE(result == omath::Vector<float, 2>{1.0F, 2.0F});
    }
}

TEST_CASE("Vector length", "vector")
{
    const omath::Vector<float, 2> vector{3.0F, 4.0F};
    const auto result = length(vector);

    REQUIRE(result == Approx(5.0F));
}

TEST_CASE("Vector length squared", "vector")
{
    const omath::Vector<float, 2> vector{3.0F, 4.0F};
    const auto result = lengthSquared(vector);

    REQUIRE(result == Approx(25.0F));
}

TEST_CASE("Vector cross product", "vector")
{
    const omath::Vector<float, 3> vector1{2.0F, 3.0F, 4.0F};
    const omath::Vector<float, 3> vector2{5.0F, 6.0F, 7.0F};
    const auto result = cross(vector1, vector2);

    REQUIRE(result == omath::Vector<float, 3>{-3.0F, 6.0F, -3.0F});
}

TEST_CASE("2D vector dot product", "vector")
{
    const omath::Vector<float, 2> vector1{1.0F, 2.0F};
    const omath::Vector<float, 2> vector2{4.0F, 5.0F};
    const float result = dot(vector1, vector2);

    REQUIRE(result == 14.0F);
}

TEST_CASE("4D vector dot product", "vector")
{
    const omath::Vector<float, 4> vector1{1.0F, 2.0F, 3.0F, 1.0F};
    const omath::Vector<float, 4> vector2{4.0F, 5.0F, 6.0F, 1.0F};
    const float result = omath::dot(vector1, vector2);

    REQUIRE(result == 33.0F);
}

TEST_CASE("Vector distance", "vector")
{
    const omath::Vector<float, 2> vector1{1.0F, 2.0F};
    const omath::Vector<float, 2> vector2{4.0F, 6.0F};
    const auto result = distance(vector1, vector2);

    REQUIRE(result == 5.0F);
}

TEST_CASE("Vector distance squared", "vector")
{
    const omath::Vector<float, 2> vector1{1.0F, 2.0F};
    const omath::Vector<float, 2> vector2{4.0F, 6.0F};
    const auto result = distanceSquared(vector1, vector2);

    REQUIRE(result == 25.0F);
}

TEST_CASE("2D normalized vector normalize", "vector")
{
    omath::Vector<float, 2> vector{1.0F, 0.0F};
    normalize(vector);

    REQUIRE(vector == omath::Vector<float, 2>{1.0F, 0.0F});
}

TEST_CASE("2D vector normalize", "vector")
{
    omath::Vector<float, 2> vector{2.0F, 3.0F};
    normalize(vector);

    REQUIRE(vector.v[0] == Approx(0.5547001962252291));
    REQUIRE(vector.v[1] == Approx(0.8320502943378437));
}

TEST_CASE("3D normalized vector normalize", "vector")
{
    omath::Vector<float, 3> vector{1.0F, 0.0F, 0.0F};
    normalize(vector);

    REQUIRE(vector == omath::Vector<float, 3>{1.0F, 0.0F, 0.0F});
}

TEST_CASE("3D vector normalize", "vector")
{
    omath::Vector<float, 3> vector{2.0F, 3.0F, 4.0F};
    normalize(vector);

    REQUIRE(vector.v[0] == Approx(0.3713906763541037));
    REQUIRE(vector.v[1] == Approx(0.5570860145311556));
    REQUIRE(vector.v[2] == Approx(0.7427813527082074));
}

TEST_CASE("3D vector normalized", "vector")
{
    const omath::Vector<float, 3> vector{2.0F, 3.0F, 4.0F};
    const auto result = normalized(vector);

    REQUIRE(result.v[0] == Approx(0.3713906763541037));
    REQUIRE(result.v[1] == Approx(0.5570860145311556));
    REQUIRE(result.v[2] == Approx(0.7427813527082074));
}
