//
// elnormous/omath
//

#ifndef OMATH_QUATERNION
#define OMATH_QUATERNION

#include <array>
#include <type_traits>

namespace omath
{
    template <typename T>
    class Quaternion final
    {
    public:
#if defined(__SSE__) || defined(_M_X64) || _M_IX86_FP >= 1 || defined(__ARM_NEON__)
        alignas(std::is_same_v<T, float> ? 4 * sizeof(T) : sizeof(T))
#endif
#if (defined(__SSE2__) || defined(_M_X64) || _M_IX86_FP >= 2) || (defined(__ARM_NEON__) && defined(__aarch64__))
        alignas(std::is_same_v<T, double> ? 4 * sizeof(T) : sizeof(T))
#endif
        std::array<T, 4> v;

        [[nodiscard]] auto& operator[](const std::size_t index) noexcept { return v[index]; }
        [[nodiscard]] constexpr auto operator[](const std::size_t index) const noexcept { return v[index]; }

        [[nodiscard]] auto& x() noexcept { return v[0]; }
        [[nodiscard]] constexpr auto x() const noexcept { return v[0]; }

        [[nodiscard]] auto& y() noexcept { return v[1]; }
        [[nodiscard]] constexpr auto y() const noexcept { return v[1]; }

        [[nodiscard]] auto& z() noexcept { return v[2]; }
        [[nodiscard]] constexpr auto z() const noexcept { return v[2]; }

        [[nodiscard]] auto& w() noexcept { return v[3]; }
        [[nodiscard]] constexpr auto w() const noexcept { return v[3]; }
    };

    template <typename T>
    constexpr auto identityQuaternion = Quaternion<T>{T(0), T(0), T(0), T(1)};

    template <typename T>
    constexpr void setIdentity(Quaternion<T>& quat) noexcept
    {
        quat.v = {T(0), T(0), T(0), T(1)};
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator==(const Quaternion<T>& quat1,
                                            const Quaternion<T>& quat2) noexcept
    {
        return quat1.v[0] == quat2.v[0] &&
            quat1.v[1] == quat2.v[1] &&
            quat1.v[2] == quat2.v[2] &&
            quat1.v[3] == quat2.v[3];
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator!=(const Quaternion<T>& quat1,
                                            const Quaternion<T>& quat2) noexcept
    {
        return quat1.v[0] != quat2.v[0] ||
            quat1.v[1] != quat2.v[1] ||
            quat1.v[2] != quat2.v[2] ||
            quat1.v[3] != quat2.v[3];
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator+(const Quaternion<T>& quat) noexcept
    {
        return quat;
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator-(const Quaternion<T>& quat) noexcept
    {
        return Quaternion<T>{-quat.v[0], -quat.v[1], -quat.v[2], -quat.v[3]};
    }

    template <typename T>
    constexpr void negate(Quaternion<T>& quat) noexcept
    {
        quat.v[0] = -quat.v[0];
        quat.v[1] = -quat.v[1];
        quat.v[2] = -quat.v[2];
        quat.v[3] = -quat.v[3];
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator+(const Quaternion<T>& quat1,
                                           const Quaternion<T>& quat2) noexcept
    {
        return Quaternion<T>{
            quat1.v[0] + quat2.v[0],
            quat1.v[1] + quat2.v[1],
            quat1.v[2] + quat2.v[2],
            quat1.v[3] + quat2.v[3]
        };
    }

    template <typename T>
    constexpr auto& operator+=(Quaternion<T>& quat1,
                               const Quaternion<T>& quat2) noexcept
    {
        quat1.v[0] += quat2.v[0];
        quat1.v[1] += quat2.v[1];
        quat1.v[2] += quat2.v[2];
        quat1.v[3] += quat2.v[3];

        return quat1;
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator-(const Quaternion<T>& quat1,
                                           const Quaternion<T>& quat2) noexcept
    {
        return Quaternion<T>{
            quat1.v[0] - quat2.v[0],
            quat1.v[1] - quat2.v[1],
            quat1.v[2] - quat2.v[2],
            quat1.v[3] - quat2.v[3]
        };
    }

    template <typename T>
    constexpr auto& operator-=(Quaternion<T>& quat1,
                               const Quaternion<T>& quat2) noexcept
    {
        quat1.v[0] -= quat2.v[0];
        quat1.v[1] -= quat2.v[1];
        quat1.v[2] -= quat2.v[2];
        quat1.v[3] -= quat2.v[3];

        return quat1;
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator*(const Quaternion<T>& quat1,
                                           const Quaternion<T>& quat2) noexcept
    {
        return Quaternion<T>{
            quat1.v[0] * quat2.v[3] + quat1.v[1] * quat2.v[2] - quat1.v[2] * quat2.v[1] + quat1.v[3] * quat2.v[0],
            -quat1.v[0] * quat2.v[2] + quat1.v[1] * quat2.v[3] + quat1.v[2] * quat2.v[0] + quat1.v[3] * quat2.v[1],
            quat1.v[0] * quat2.v[1] - quat1.v[1] * quat2.v[0] + quat1.v[2] * quat2.v[3] + quat1.v[3] * quat2.v[2],
            -quat1.v[0] * quat2.v[0] - quat1.v[1] * quat2.v[1] - quat1.v[2] * quat2.v[2] + quat1.v[3] * quat2.v[3]
        };
    }

    template <typename T>
    constexpr auto& operator*=(Quaternion<T>& quat1,
                               const Quaternion<T>& quat2) noexcept
    {
        quat1.v = {
            quat1.v[0] * quat2.v[3] + quat1.v[1] * quat2.v[2] - quat1.v[2] * quat2.v[1] + quat1.v[3] * quat2.v[0],
            -quat1.v[0] * quat2.v[2] + quat1.v[1] * quat2.v[3] + quat1.v[2] * quat2.v[0] + quat1.v[3] * quat2.v[1],
            quat1.v[0] * quat2.v[1] - quat1.v[1] * quat2.v[0] + quat1.v[2] * quat2.v[3] + quat1.v[3] * quat2.v[2],
            -quat1.v[0] * quat2.v[0] - quat1.v[1] * quat2.v[1] - quat1.v[2] * quat2.v[2] + quat1.v[3] * quat2.v[3]
        };

        return quat1;
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator*(const Quaternion<T>& quat,
                                           const T scalar) noexcept
    {
        return Quaternion<T>{
            quat.v[0] * scalar,
            quat.v[1] * scalar,
            quat.v[2] * scalar,
            quat.v[3] * scalar
        };
    }

    template <typename T>
    constexpr auto& operator*=(Quaternion<T>& quat,
                               const T scalar) noexcept
    {
        quat.v[0] *= scalar;
        quat.v[1] *= scalar;
        quat.v[2] *= scalar;
        quat.v[3] *= scalar;

        return quat;
    }

    template <typename T>
    [[nodiscard]] constexpr auto operator/(const Quaternion<T>& quat,
                                           const T scalar) noexcept
    {
        return Quaternion<T>{
            quat.v[0] / scalar,
            quat.v[1] / scalar,
            quat.v[2] / scalar,
            quat.v[3] / scalar
        };
    }

    template <typename T>
    constexpr auto& operator/=(Quaternion<T>& quat,
                               const T scalar) noexcept
    {
        quat.v[0] /= scalar;
        quat.v[1] /= scalar;
        quat.v[2] /= scalar;
        quat.v[3] /= scalar;

        return quat;
    }

    template <typename T>
    constexpr void conjugate(Quaternion<T>& quat) noexcept
    {
        quat.v[0] = -quat.v[0];
        quat.v[1] = -quat.v[1];
        quat.v[2] = -quat.v[2];
    }

    template <typename T>
    [[nodiscard]] constexpr auto conjugated(const Quaternion<T>& quat) noexcept
    {
        return Quaternion<T>{-quat.v[0], -quat.v[1], -quat.v[2], quat.v[3]};
    }
}

#include "QuaternionAvx.hpp"
#include "QuaternionNeon.hpp"
#include "QuaternionSse.hpp"

#endif
