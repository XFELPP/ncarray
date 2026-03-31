/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_CUSTOM_TYPES_HH
#define NCARRAY_CUSTOM_TYPES_HH

#include <cmath>
#include <concepts>
#include <ostream>
#include <type_traits>

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
#define DEFINE_VECTOR_OPS(VECDTYPE, ...)                             \
  NCA_HD constexpr VECDTYPE operator+(const VECDTYPE& other) const { \
    return { __VA_ARGS__(+) };                                       \
  }                                                                  \
  NCA_HD constexpr VECDTYPE operator-(const VECDTYPE& other) const { \
    return { __VA_ARGS__(-) };                                       \
  }                                                                  \
  NCA_HD constexpr VECDTYPE operator*(const VECDTYPE& other) const { \
    return { __VA_ARGS__(*) };                                       \
  }                                                                  \
  NCA_HD constexpr VECDTYPE operator/(const VECDTYPE& other) const { \
    return { __VA_ARGS__(/) };                                       \
  }                                                                  \
  NCA_HD constexpr VECDTYPE operator+=(const VECDTYPE& other) {      \
    if (this != &other) *this = *this + other;                       \
    return *this;                                                    \
  }                                                                  \
  NCA_HD constexpr VECDTYPE operator-=(const VECDTYPE& other) {      \
    if (this != &other) *this = *this - other;                       \
    return *this;                                                    \
  }                                                                  \
  NCA_HD constexpr VECDTYPE operator*=(const VECDTYPE& other) {      \
    if (this != &other) *this = *this * other;                       \
    return *this;                                                    \
  }                                                                  \
  NCA_HD constexpr VECDTYPE operator/=(const VECDTYPE& other) {      \
    if (this != &other) *this = *this / other;                       \
    return *this;                                                    \
  }

  template <typename T>
  concept Numeric = std::is_arithmetic<T>::value;

#define DEFINE_SCALAR_OPS(VECDTYPE, ...)                               \
  template <Numeric T>                                                 \
  NCA_HD friend constexpr VECDTYPE operator+(const VECDTYPE& v, T s) { \
    return { __VA_ARGS__(v, s, +) };                                   \
  }                                                                    \
  template <Numeric T>                                                 \
  NCA_HD friend constexpr VECDTYPE operator+(T s, const VECDTYPE& v) { \
    return { __VA_ARGS__(v, s, +) };                                   \
  }                                                                    \
  template <Numeric T>                                                 \
  NCA_HD friend constexpr VECDTYPE operator-(const VECDTYPE& v, T s) { \
    return { __VA_ARGS__(v, s, -) };                                   \
  }                                                                    \
  template <Numeric T>                                                 \
  NCA_HD friend constexpr VECDTYPE operator-(T s, const VECDTYPE& v) { \
    return { __VA_ARGS__(v, s, -) };                                   \
  }                                                                    \
  template <Numeric T>                                                 \
  NCA_HD friend constexpr VECDTYPE operator*(const VECDTYPE& v, T s) { \
    return { __VA_ARGS__(v, s, *) };                                   \
  }                                                                    \
  template <Numeric T>                                                 \
  NCA_HD friend constexpr VECDTYPE operator*(T s, const VECDTYPE& v) { \
    return { __VA_ARGS__(v, s, *) };                                   \
  }                                                                    \
  template <Numeric T>                                                 \
  NCA_HD friend constexpr VECDTYPE operator/(const VECDTYPE& v, T s) { \
    return { __VA_ARGS__(v, s, /) };                                   \
  }                                                                    \
  template <Numeric T>                                                 \
  NCA_HD friend constexpr VECDTYPE operator/(T s, const VECDTYPE& v) { \
    return { __VA_ARGS__(v, s, /) };                                   \
  }

#define DEFINE_VECTOR_COMPARISONS(VECDTYPE, ...)                        \
  friend NCA_HD bool operator==(const VECDTYPE& a, const VECDTYPE& b) { \
    return __VA_ARGS__(a, b, ==);                                       \
  }                                                                     \
  friend NCA_HD bool operator!=(const VECDTYPE& a, const VECDTYPE& b) { \
    return !(a == b);                                                   \
  }                                                                     \
  friend NCA_HD bool operator<(const VECDTYPE& a, const VECDTYPE& b) {  \
    return __VA_ARGS__(a, b, <);                                        \
  }                                                                     \
  friend NCA_HD bool operator>(const VECDTYPE& a, const VECDTYPE& b) {  \
    return __VA_ARGS__(a, b, >);                                        \
  }

#define BINOPS2(OP) x OP other.x, y OP other.y
#define BINOPS3(OP) x OP other.x, y OP other.y, z OP other.z
#define BINOPS4(OP) x OP other.x, y OP other.y, z OP other.z, w OP other.w

#define SCALAROPS2(v, s, OP) static_cast<decltype(v.x)>(v.x OP s), static_cast<decltype(v.y)>(v.y OP s)
#define SCALAROPS3(v, s, OP)            \
  static_cast<decltype(v.x)>(v.x OP s), \
  static_cast<decltype(v.y)>(v.y OP s), \
  static_cast<decltype(v.z)>(v.z OP s)
#define SCALAROPS4(v, s, OP)            \
  static_cast<decltype(v.x)>(v.x OP s), \
  static_cast<decltype(v.y)>(v.y OP s), \
  static_cast<decltype(v.z)>(v.z OP s), \
  static_cast<decltype(v.w)>(v.w OP s)

#define COMPOPS2(a, b, OP) a.x OP b.x && a.y OP b.y
#define COMPOPS3(a, b, OP) a.x OP b.x && a.y OP b.y && a.z OP b.z
#define COMPOPS4(a, b, OP) a.x OP b.x && a.y OP b.y && a.z OP b.z && a.w OP b.w

  template <typename T>
  concept Vector2DType = requires(T v) {
    v.x;
    v.y;
  };
  template <typename T>
  concept Vector3DType = Vector2DType<T> && requires(T v) {
    v.z;
  };
  template <typename T>
  concept Vector4DType = Vector3DType<T> && requires(T v) {
    v.w;
  };

#pragma pack(push, 8)
  struct Float2 {
    float x { 0.0f };
    float y { 0.0f };

    constexpr Float2() = default;
    constexpr Float2(const Float2&) = default;
    constexpr Float2(Float2&&) = default;
    constexpr Float2& operator=(const Float2&) = default;
    constexpr Float2& operator=(Float2&&) = default;

    NCA_HD constexpr Float2(float x_, float y_)
      : x(x_)
      , y(y_)
    {}

    // A broadcast constructor for T{0}, T{nan} and so on
    template <Numeric U>
    NCA_HD constexpr Float2(U s)
      : x(static_cast<float>(s))
      , y(static_cast<float>(s))
    {}

    // A constructor for static_cast<OtherVec>..
    template <Vector2DType U>
    requires (!std::is_same_v<std::decay_t<U>, Float2>)
    NCA_HD constexpr Float2(const U& other)
      : x(static_cast<float>(other.x))
      , y(static_cast<float>(other.y))
    {}

    DEFINE_VECTOR_OPS(Float2, BINOPS2)
    DEFINE_SCALAR_OPS(Float2, SCALAROPS2)
    DEFINE_VECTOR_COMPARISONS(Float2, COMPOPS2)
  };
#pragma pack(pop)

#pragma pack(push, 4)
  struct Float3 {
    float x { 0.0f };
    float y { 0.0f };
    float z { 0.0f };

    constexpr Float3() = default;
    constexpr Float3(const Float3&) = default;
    constexpr Float3(Float3&&) = default;
    constexpr Float3& operator=(const Float3&) = default;
    constexpr Float3& operator=(Float3&&) = default;

    NCA_HD constexpr Float3(float x_, float y_, float z_)
      : x(x_)
      , y(y_)
      , z(z_)
    {}

    // A broadcast constructor for T{0}, T{nan} and so on
    template <Numeric U>
    NCA_HD constexpr Float3(U s)
      : x(static_cast<float>(s))
      , y(static_cast<float>(s))
      , z(static_cast<float>(s))
    {}

    // A constructor for static_cast<OtherVec>..
    template <Vector2DType U>
    requires (!std::is_same_v<std::decay_t<U>, Float3>)
    NCA_HD constexpr Float3(const U& other)
      : x(static_cast<float>(other.x))
      , y(static_cast<float>(other.y))
    {
      if constexpr (Vector3DType<U>) {
        z = static_cast<float>(other.z);
      }
    }

    DEFINE_VECTOR_OPS(Float3, BINOPS3)
    DEFINE_SCALAR_OPS(Float3, SCALAROPS3)
    DEFINE_VECTOR_COMPARISONS(Float3, COMPOPS3)
  };
#pragma pack(pop)

#pragma pack(push, 8)
  struct Float4 {
    float x { 0.0f };
    float y { 0.0f };
    float z { 0.0f };
    float w { 0.0f };

    constexpr Float4() = default;
    constexpr Float4(const Float4&) = default;
    constexpr Float4(Float4&&) = default;
    constexpr Float4& operator=(const Float4&) = default;
    constexpr Float4& operator=(Float4&&) = default;

    NCA_HD constexpr Float4(float x_, float y_, float z_, float w_)
      : x(x_)
      , y(y_)
      , z(z_)
      , w(w_)
    {}

    // A broadcast constructor for T{0}, T{nan} and so on
    template <Numeric U>
    NCA_HD constexpr Float4(U s)
      : x(static_cast<float>(s))
      , y(static_cast<float>(s))
      , z(static_cast<float>(s))
      , w(static_cast<float>(s))
    {}

    // A constructor for static_cast<OtherVec>..
    template <Vector2DType U>
    requires (!std::is_same_v<std::decay_t<U>, Float4>)
    NCA_HD constexpr Float4(const U& other)
      : x(static_cast<float>(other.x))
      , y(static_cast<float>(other.y))
    {
      if constexpr (Vector3DType<U>) {
        z = static_cast<float>(other.z);
      }
      if constexpr (Vector4DType<U>) {
        w = static_cast<float>(other.w);
      }
    }

    DEFINE_VECTOR_OPS(Float4, BINOPS4)
    DEFINE_SCALAR_OPS(Float4, SCALAROPS4)
    DEFINE_VECTOR_COMPARISONS(Float4, COMPOPS4)
  };
#pragma pack(pop)

#pragma pack(push, 16)
  struct Double2 {
    double x { 0.0 };
    double y { 0.0 };

    constexpr Double2() = default;
    constexpr Double2(const Double2&) = default;
    constexpr Double2(Double2&&) = default;
    constexpr Double2& operator=(const Double2&) = default;
    constexpr Double2& operator=(Double2&&) = default;

    NCA_HD constexpr Double2(double x_, double y_)
      : x(x_)
      , y(y_)
    {}

    // A broadcast constructor for T{0}, T{nan} and so on
    template <Numeric U>
    NCA_HD constexpr Double2(U s)
      : x(static_cast<double>(s))
      , y(static_cast<double>(s))
    {}

    // A constructor for static_cast<OtherVec>..
    template <Vector2DType U>
    requires (!std::is_same_v<std::decay_t<U>, Double2>)
    NCA_HD constexpr Double2(const U& other)
      : x(static_cast<double>(other.x))
      , y(static_cast<double>(other.y))
    {}

    DEFINE_VECTOR_OPS(Double2, BINOPS2)
    DEFINE_SCALAR_OPS(Double2, SCALAROPS2)
    DEFINE_VECTOR_COMPARISONS(Double2, COMPOPS2)
  };
#pragma pack(pop)

#pragma pack(push, 8)
  struct Double3 {
    double x { 0.0 };
    double y { 0.0 };
    double z { 0.0 };

    constexpr Double3() = default;
    constexpr Double3(const Double3&) = default;
    constexpr Double3(Double3&&) = default;
    constexpr Double3& operator=(const Double3&) = default;
    constexpr Double3& operator=(Double3&&) = default;

    NCA_HD constexpr Double3(double x_, double y_, double z_)
      : x(x_)
      , y(y_)
      , z(z_)
    {}

    // A broadcast constructor for T{0}, T{nan} and so on
    template <Numeric U>
    NCA_HD constexpr Double3(U s)
      : x(static_cast<double>(s))
      , y(static_cast<double>(s))
      , z(static_cast<double>(s))
    {}

    // A constructor for static_cast<OtherVec>..
    template <Vector2DType U>
    requires (!std::is_same_v<std::decay_t<U>, Double3>)
    NCA_HD constexpr Double3(const U& other)
      : x(static_cast<double>(other.x))
      , y(static_cast<double>(other.y))
    {
      if constexpr (Vector3DType<U>) {
        z = static_cast<double>(other.z);
      }
    }

    DEFINE_VECTOR_OPS(Double3, BINOPS3)
    DEFINE_SCALAR_OPS(Double3, SCALAROPS3)
    DEFINE_VECTOR_COMPARISONS(Double3, COMPOPS3)
  };
#pragma pack(pop)

#pragma pack(push, 16)
  struct Double4 {
    double x { 0.0 };
    double y { 0.0 };
    double z { 0.0 };
    double w { 0.0 };

    constexpr Double4() = default;
    constexpr Double4(const Double4&) = default;
    constexpr Double4(Double4&&) = default;
    constexpr Double4& operator=(const Double4&) = default;
    constexpr Double4& operator=(Double4&&) = default;

    NCA_HD constexpr Double4(double x_, double y_, double z_, double w_)
      : x(x_)
      , y(y_)
      , z(z_)
      , w(w_)
    {}

    // A broadcast constructor for T{0}, T{nan} and so on
    template <Numeric U>
    NCA_HD constexpr Double4(U s)
      : x(static_cast<double>(s))
      , y(static_cast<double>(s))
      , z(static_cast<double>(s))
      , w(static_cast<double>(s))
    {}

    // A constructor for static_cast<OtherVec>..
    template <Vector2DType U>
    requires (!std::is_same_v<std::decay_t<U>, Double4>)
    NCA_HD constexpr Double4(const U& other)
      : x(static_cast<double>(other.x))
      , y(static_cast<double>(other.y))
    {
      if constexpr (Vector3DType<U>) {
        z = static_cast<double>(other.z);
      }
      if constexpr (Vector4DType<U>) {
        w = static_cast<double>(other.w);
      }
    }

    DEFINE_VECTOR_OPS(Double4, BINOPS4)
    DEFINE_SCALAR_OPS(Double4, SCALAROPS4)
    DEFINE_VECTOR_COMPARISONS(Double4, COMPOPS4)
  };
#pragma pack(pop)

#undef BINOPS4
#undef BINOPS3
#undef BINOPS2
#undef SCALAROPS4
#undef SCALAROPS3
#undef SCALAROPS2
#undef COMPOPS4
#undef COMPOPS3
#undef COMPOPS2
#undef DEFINE_SCALAR_OPS
#undef DEFINE_VECTOR_OPS
#undef DEFINE_VECTOR_COMPARISONS

  template <Vector2DType T>
  std::ostream& operator<<(std::ostream& oss, const T& vec) {
    oss << "{" << vec.x << ", " << vec.y;
    if constexpr (Vector3DType<T>) {
      oss << ", " << vec.z;
    }
    if constexpr (Vector4DType<T>) {
      oss << ", " << vec.w;
    }
    oss << "}";
    return oss;
  }

  template <Vector2DType T>
  NCA_HD bool isfinite(const T& v) {
    if constexpr (Vector4DType<T>) {
      return
        std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w);
    } else if constexpr (Vector3DType<T>) {
      return
        std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    } else {
      return std::isfinite(v.x) && std::isfinite(v.y);
    }
  }

} // namespace ncarray

#endif // NCARRAY_CUSTOM_TYPES_HH
