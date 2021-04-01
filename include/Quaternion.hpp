//
// elnormous/math
//

#ifndef MATH_QUATERNION
#define MATH_QUATERNION

#include <array>
#include <type_traits>

namespace math
{
    template <typename T> class Quaternion final
    {
    public:
#if defined(__SSE__)
        alignas(4 * sizeof(T))
#endif
        std::array<T, 4> v{};

        constexpr Quaternion() noexcept {}

        constexpr Quaternion(const T x, const T y, const T z, const T w) noexcept:
            v{{x, y, z, w}}
        {
        }

        auto& operator[](const std::size_t index) noexcept { return v[index]; }
        constexpr auto operator[](const std::size_t index) const noexcept { return v[index]; }

        auto& x() noexcept { return v[0]; }
        constexpr auto x() const noexcept { return v[0]; }

        auto& y() noexcept { return v[1]; }
        constexpr auto y() const noexcept { return v[1]; }

        auto& z() noexcept { return v[2]; }
        constexpr auto z() const noexcept { return v[2]; }

        auto& w() noexcept { return v[3]; }
        constexpr auto w() const noexcept { return v[3]; }

        static constexpr auto identity() noexcept
        {
            return Quaternion{T(0), T(0), T(0), T(1)};
        }

        constexpr auto operator==(const Quaternion& q) const noexcept
        {
            return v[0] == q.v[0] && v[1] == q.v[1] && v[2] == q.v[2] && v[3] == q.v[3];
        }

        constexpr auto operator!=(const Quaternion& q) const noexcept
        {
            return v[0] != q.v[0] || v[1] != q.v[1] || v[2] != q.v[2] || v[3] != q.v[3];
        }

        constexpr auto operator-() const noexcept
        {
            return Quaternion{-v[0], -v[1], -v[2], -v[3]};
        }

        constexpr const auto operator+(const Quaternion& q) const noexcept
        {
            return Quaternion{
                v[0] + q.v[0],
                v[1] + q.v[1],
                v[2] + q.v[2],
                v[3] + q.v[3]
            };
        }

        constexpr auto& operator+=(const Quaternion& q) noexcept
        {
            v[0] += q.v[0];
            v[1] += q.v[1];
            v[2] += q.v[2];
            v[3] += q.v[3];

            return *this;
        }

        constexpr const auto operator-(const Quaternion& q) const noexcept
        {
            return Quaternion{
                v[0] - q.v[0],
                v[1] - q.v[1],
                v[2] - q.v[2],
                v[3] - q.v[3]
            };
        }

        constexpr auto& operator-=(const Quaternion& q) noexcept
        {
            v[0] -= q.v[0];
            v[1] -= q.v[1];
            v[2] -= q.v[2];
            v[3] -= q.v[3];

            return *this;
        }

        constexpr const auto operator*(const Quaternion& q) const noexcept
        {
            return Quaternion{
                v[0] * q.v[3] + v[1] * q.v[2] - v[2] * q.v[1] + v[3] * q.v[0],
                -v[0] * q.v[2] + v[1] * q.v[3] + v[2] * q.v[0] + v[3] * q.v[1],
                v[0] * q.v[1] - v[1] * q.v[0] + v[2] * q.v[3] + v[3] * q.v[2],
                -v[0] * q.v[0] - v[1] * q.v[1] - v[2] * q.v[2] + v[3] * q.v[3]
            };
        }

        constexpr auto& operator*=(const Quaternion& q) noexcept
        {
            v = {
                v[0] * q.v[3] + v[1] * q.v[2] - v[2] * q.v[1] + v[3] * q.v[0],
                -v[0] * q.v[2] + v[1] * q.v[3] + v[2] * q.v[0] + v[3] * q.v[1],
                v[0] * q.v[1] - v[1] * q.v[0] + v[2] * q.v[3] + v[3] * q.v[2],
                -v[0] * q.v[0] - v[1] * q.v[1] - v[2] * q.v[2] + v[3] * q.v[3]
            };

            return *this;
        }

        constexpr const auto operator*(const T scalar) const noexcept
        {
            return Quaternion{
                v[0] * scalar,
                v[1] * scalar,
                v[2] * scalar,
                v[3] * scalar
            };
        }

        constexpr auto& operator*=(const T scalar) noexcept
        {
            v[0] *= scalar;
            v[1] *= scalar;
            v[2] *= scalar;
            v[3] *= scalar;

            return *this;
        }

        constexpr const auto operator/(const T scalar) const noexcept
        {
            return Quaternion{
                v[0] / scalar,
                v[1] / scalar,
                v[2] / scalar,
                v[3] / scalar
            };
        }

        constexpr auto& operator/=(const T scalar) noexcept
        {
            v[0] /= scalar;
            v[1] /= scalar;
            v[2] /= scalar;
            v[3] /= scalar;

            return *this;
        }
    };
}

#endif
