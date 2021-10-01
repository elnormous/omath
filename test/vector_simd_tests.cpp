#include "catch2/catch.hpp"
#include "Vector.hpp"

TEST_CASE("4D vector dot product using SIMD", "vector")
{
    const omath::Vector<float, 4, true> vector1{1.0F, 2.0F, 3.0F, 1.0F};
    const omath::Vector<float, 4, true> vector2{4.0F, 5.0F, 6.0F, 1.0F};
    const float result = dot(vector1, vector2);

    REQUIRE(result == 33.0F);
}
