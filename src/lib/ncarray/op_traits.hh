/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_OP_TRAITS_HH
#define NCARRAY_OP_TRAITS_HH

#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"

#ifdef __CUDACC_RTC__
typedef long long ssize_t;
#else
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif
#endif

#ifdef __CUDACC_RTC__
#include <cuda/std/cmath>
#include <cuda/std/complex>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

using cuda::std::isfinite;

using cuda::std::complex;
using cuda::std::is_same_v;
using cuda::std::numeric_limits;

using cuda::std::int8_t;
using cuda::std::int16_t;
using cuda::std::int32_t;
using cuda::std::int64_t;

using cuda::std::uint8_t;
using cuda::std::uint16_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;

#else
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

using std::isfinite;

using std::complex;
using std::is_same_v;
using std::numeric_limits;

using std::int8_t;
using std::int16_t;
using std::int32_t;
using std::int64_t;

using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
#endif

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
  /**
   * Numeric traits for specifying types during arithmetic as well as providing
   * operations for types that don't have them, or overriding their default
   * behaviour.
   *
   * Examples include type promotion for small integers (e.g. std::uint8_t) so
   * accumulation operations don't overflow, or providing comparisons for complex
   * number types.
   */
  template <typename T>
  struct BaseOpTraits {
    using value_type = T;
    using sum_type = T;
    using diff_type = T;
    using truediv_type = double;

    // Comparisons and identities -- needed especially for specializations below
    // on things like complex
    NCA_HD static bool greater(const T& a, const T& b) { return a > b; }
    NCA_HD static bool ge(const T& a, const T& b) { return a >= b; }
    NCA_HD static bool less(const T& a, const T& b) { return a < b; }
    NCA_HD static bool le(const T& a, const T& b) { return a <= b; }
    NCA_HD static T lowest() { return numeric_limits<T>::lowest(); }
    NCA_HD static T max() { return numeric_limits<T>::max(); }

    template <typename To>
    NCA_HD static To cast(const T& val) {
      if constexpr (is_same_v<To, T>) {
        return val;
      } else if constexpr (Vector2DType<T> && !Vector2DType<To>) {
        // For vectors just return the first value during a cast to scalar
        return static_cast<To>(val.x);
      } else if constexpr (!Vector2DType<T> && Vector2DType<To>) {
        // For scalar to vector, broadcast.
        To res;
        res.x = static_cast<decltype(To::x)>(val);
        res.y = static_cast<decltype(To::y)>(val);
        if constexpr (Vector3DType<To>) {
          res.z = static_cast<decltype(To::z)>(val);
        }
        if constexpr (Vector4DType<To>) {
          res.w = static_cast<decltype(To::w)>(val);
        }
        return res;
      } else if constexpr (requires { To().real(); }) {
        // Scalar cast to complex
        return To(static_cast<typename To::value_type>(val), 0);
      } else {
        // Scalar cast to scalar
        return static_cast<To>(val);
      }
    }

    NCA_HD static bool land(const T& a, const T& b) {
      if constexpr (requires { a.real(); }) {
        // Complex
        if (a.real() && b.real()) {
          return a.imag() && b.imag();
        }
        return false;
      } else if constexpr (Vector4DType<T>) {
        return (a.x && b.x) && (a.y && b.y) && (a.z && b.z) && (a.w && b.w);
      } else if constexpr (Vector3DType<T>) {
        return (a.x && b.x) && (a.y && b.y) && (a.z && b.z);
      } else if constexpr (Vector2DType<T>) {
        return (a.x && b.x) && (a.y && b.y);
      } else {
        return a && b;
      }
    }
    NCA_HD static bool lor(const T& a, const T& b) {
      if constexpr (requires { a.real(); }) {
        // Complex
        if (a.real() || b.real()) {
          return true;
        }
        return a.imag() || b.imag();
      } else if constexpr (Vector4DType<T>) {
        return (a.x || b.x) || (a.y || b.y) || (a.z || b.z) || (a.w || b.w);
      } else if constexpr (Vector3DType<T>) {
        return (a.x || b.x) || (a.y || b.y) || (a.z || b.z);
      } else if constexpr (Vector2DType<T>) {
        return (a.x || b.x) || (a.y || b.y);
      } else {
        return a || b;
      }
    }

    NCA_HD static bool isfinite(const T& v) {
      if constexpr (Vector4DType<T>) {
        return ::isfinite(v.x) && ::isfinite(v.y) && ::isfinite(v.z) && ::isfinite(v.w);
      } else if constexpr (Vector3DType<T>) {
        return ::isfinite(v.x) && ::isfinite(v.y) && ::isfinite(v.z);
      } else if constexpr (Vector2DType<T>) {
        return ::isfinite(v.x) && ::isfinite(v.y);
      } else {
        return ::isfinite(v);
      }
    }
  };

  template <typename T>
  struct op_traits : BaseOpTraits<T> {};

  // Complex need comparison operations
  template <typename T>
  struct op_traits<complex<T>> : BaseOpTraits<T> {
    using value_type = T;
    using sum_type = complex<T>;
    using diff_type = complex<T>;
    using truediv_type = complex<double>;

    NCA_HD static bool greater(const complex<T>& a, const complex<T>& b) {
      if (a.real() != b.real()) {
        return a.real() > b.real();
      }
      return a.imag() > b.imag();
    }

    NCA_HD static bool ge(const complex<T>& a, const complex<T>& b) {
      if (a.real() != b.real()) {
        return a.real() > b.real();
      }
      return a.imag() >= b.imag();
    }

    NCA_HD static bool less(const complex<T>& a, const complex<T>& b) {
      if (a.real() != b.real()) {
        return a.real() < b.real();
      }
      return a.imag() < b.imag();
    }

    NCA_HD static bool le(const complex<T>& a, const complex<T>& b) {
      if (a.real() != b.real()) {
        return a.real() < b.real();
      }
      return a.imag() <= b.imag();
    }

    NCA_HD static complex<T> lowest() {
      return { numeric_limits<T>::lowest(), numeric_limits<T>::lowest() };
    }

    NCA_HD static complex<T> max() {
      return { numeric_limits<T>::max(), numeric_limits<T>::max() };
    }

    template <typename To>
    NCA_HD static To cast(const complex<T>& val) {
      if constexpr (is_same_v<To, complex<T>>) {
        return val;
      } else if constexpr (is_same_v<To, bool>) {
        return val.real() != 0 || val.imag() != 0;
      } else if constexpr (Vector2DType<To>) {
        // For vectors broadcast into the first values.
        To res;
        res.x = static_cast<decltype(To::x)>(val.real());
        res.y = static_cast<decltype(To::y)>(val.imag());
        if constexpr (Vector3DType<To>) {
          res.z = static_cast<decltype(To::z)>(0);
        }
        if constexpr (Vector4DType<To>) {
          res.w = static_cast<decltype(To::w)>(0);
        }
        return res;
      } else if constexpr (requires { To().real(); }) {
        // Complex to complex cast
        return To(static_cast<typename To::value_type>(val.real()),
                  static_cast<typename To::value_type>(val.imag()));
      } else {
        // Complex to scalar cast (narrowed to real)
        return static_cast<To>(val.real());
      }
    }

    NCA_HD static bool land(const complex<T>& a, const complex<T>& b) {
      if (a.real() && b.real()) {
        return a.imag() && b.imag();
      }
      return false;
    }
    NCA_HD static bool lor(const complex<T>& a, const complex<T>& b) {
      if (a.real() || b.real()) {
        return true;
      }
      return a.imag() || b.imag();
    }

    NCA_HD static bool isfinite(const complex<T>& v) {
      return ::isfinite(v.real()) && ::isfinite(v.imag());
    }
  };

  template <>
  struct op_traits<int8_t> : BaseOpTraits<int8_t> {
    using sum_type = int64_t;
    using diff_type = int64_t;
  };

  template <>
  struct op_traits<uint8_t> : BaseOpTraits<uint8_t> {
    using sum_type = uint64_t;
    // NOTE: Unsigned types promot to SIGNED for subtraction!
    using diff_type = int64_t;
  };

  template <>
  struct op_traits<int16_t> : BaseOpTraits<int16_t> {
    using sum_type = int64_t;
    using diff_type = int64_t;
  };

  template <>
  struct op_traits<uint16_t> : BaseOpTraits<uint16_t> {
    using sum_type = uint64_t;
    // NOTE: Unsigned types promot to SIGNED for subtraction!
    using diff_type = int64_t;
  };

  template <>
  struct op_traits<bool> : BaseOpTraits<bool> {
    using sum_type = uint64_t;
    using diff_type = int64_t;
  };

  template <>
  struct op_traits<Float2> : BaseOpTraits<Float2> {
    using value_type = float;
    using truediv_type = Double2;

    // Comparisons and identities -- needed especially for specializations below
    // on things like complex
    NCA_HD static bool greater(const Float2& a, const Float2& b) {
      return a.x > b.x && a.y > b.y;
    }
    NCA_HD static bool less(const Float2& a, const Float2& b) {
      return a.x < b.x && a.y < b.y;
    }
    NCA_HD static Float2 lowest() {
      return {
        numeric_limits<float>::lowest(),
        numeric_limits<float>::lowest()
      };
    }
    NCA_HD static Float2 max() {
      return {
        numeric_limits<float>::max(),
        numeric_limits<float>::max()
      };
    }
  };

  template <>
  struct op_traits<Float3> : BaseOpTraits<Float3> {
    using value_type = float;
    using truediv_type = Double3;

    NCA_HD static bool greater(const Float3& a, const Float3& b) {
      return a.x > b.x && a.y > b.y && a.z > b.z;
    }
    NCA_HD static bool less(const Float3& a, const Float3& b) {
      return a.x < b.x && a.y < b.y && a.z < b.z;
    }
    NCA_HD static Float3 lowest() {
      return {
        numeric_limits<float>::lowest(),
        numeric_limits<float>::lowest(),
        numeric_limits<float>::lowest()
      };
    }
    NCA_HD static Float3 max() {
      return {
        numeric_limits<float>::max(),
        numeric_limits<float>::max(),
        numeric_limits<float>::max()
      };
    }
  };

  template <>
  struct op_traits<Float4> : BaseOpTraits<Float4> {
    using value_type = float;
    using truediv_type = Double4;

    NCA_HD static bool greater(const Float4& a, const Float4& b) {
      return a.x > b.x && a.y > b.y && a.z > b.z && a.w > b.w;
    }
    NCA_HD static bool less(const Float4& a, const Float4& b) {
      return a.x < b.x && a.y < b.y && a.z < b.z && a.w < b.w;
    }
    NCA_HD static Float4 lowest() {
      return {
        numeric_limits<float>::lowest(),
        numeric_limits<float>::lowest(),
        numeric_limits<float>::lowest(),
        numeric_limits<float>::lowest()
      };
    }
    NCA_HD static Float4 max() {
      return {
        numeric_limits<float>::max(),
        numeric_limits<float>::max(),
        numeric_limits<float>::max(),
        numeric_limits<float>::max()
      };
    }
  };

  template <>
  struct op_traits<Double2> : BaseOpTraits<Double2> {
    using value_type = double;
    using truediv_type = Double2;

    // Comparisons and identities -- needed especially for specializations below
    // on things like complex
    NCA_HD static bool greater(const Double2& a, const Double2& b) {
      return a.x > b.x && a.y > b.y;
    }
    NCA_HD static bool less(const Double2& a, const Double2& b) {
      return a.x < b.x && a.y < b.y;
    }
    NCA_HD static Double2 lowest() {
      return {
        numeric_limits<double>::lowest(),
        numeric_limits<double>::lowest()
      };
    }
    NCA_HD static Double2 max() {
      return {
        numeric_limits<double>::max(),
        numeric_limits<double>::max()
      };
    }
  };

  template <>
  struct op_traits<Double3> : BaseOpTraits<Double3> {
    using value_type = double;
    using truediv_type = Double3;

    NCA_HD static bool greater(const Double3& a, const Double3& b) {
      return a.x > b.x && a.y > b.y && a.z > b.z;
    }
    NCA_HD static bool less(const Double3& a, const Double3& b) {
      return a.x < b.x && a.y < b.y && a.z < b.z;
    }
    NCA_HD static Double3 lowest() {
      return {
        numeric_limits<double>::lowest(),
        numeric_limits<double>::lowest(),
        numeric_limits<double>::lowest()
      };
    }
    NCA_HD static Double3 max() {
      return {
        numeric_limits<double>::max(),
        numeric_limits<double>::max(),
        numeric_limits<double>::max()
      };
    }
  };

  template <>
  struct op_traits<Double4> : BaseOpTraits<Double4> {
    using value_type = double;
    using truediv_type = Double4;

    NCA_HD static bool greater(const Double4& a, const Double4& b) {
      return a.x > b.x && a.y > b.y && a.z > b.z && a.w > b.w;
    }
    NCA_HD static bool less(const Double4& a, const Double4& b) {
      return a.x < b.x && a.y < b.y && a.z < b.z && a.w < b.w;
    }
    NCA_HD static Double4 lowest() {
      return {
        numeric_limits<double>::lowest(),
        numeric_limits<double>::lowest(),
        numeric_limits<double>::lowest(),
        numeric_limits<double>::lowest()
      };
    }
    NCA_HD static Double4 max() {
      return {
        numeric_limits<double>::max(),
        numeric_limits<double>::max(),
        numeric_limits<double>::max(),
        numeric_limits<double>::max()
      };
    }
  };


} // namespace ncarray

#endif // NCARRAY_OP_TRAITS_HH
