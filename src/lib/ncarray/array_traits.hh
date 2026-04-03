/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ARRAY_TRAITS_HH
#define NCARRAY_ARRAY_TRAITS_HH

#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"
#include "ncarray/storage.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <complex>
#include <concepts>
#include <cstdint>
#include <limits>
#include <vector>

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
  namespace impl {
    template <typename T = void> struct default_owner;
  } // namespace impl
  /**
   * The following concepts enforce the interface for the generic algorithms.
   */
  template <typename T>
  concept Shaped = requires(const T arr) {
    { arr.ndim() } -> std::convertible_to<ssize_t>;
    { arr.shape() } -> std::convertible_to<const ssize_t*>;
  };

  template <typename T>
  concept Strided = requires(const T arr) {
    { arr.strides() } -> std::convertible_to<const ssize_t*>;
  };

  template <typename T>
  concept HasData = requires(const T arr) {
    { arr.data() } -> std::convertible_to<const void*>;
  };

  template <typename T>
  concept HasDType = requires(const T arr) {
    { arr.dtype() } -> std::same_as<ncarray::DType>;
  };

  template <typename T>
  concept ArrayLike = Shaped<T> && Strided<T> && HasData<T> && HasDType<T>;

  // Mutable/writable arrays can get data that is not just const void*
  template <typename T>
  concept MutableArrayLike = ArrayLike<T> && requires(T arr) {
    { arr.data() } -> std::same_as<void*>;
  };

  template <class T>
  concept ViewArrayLike = ArrayLike<T> &&
    std::is_base_of_v<ViewTag, typename std::remove_cvref_t<T>::StoragePolicy>;

  // ArrayLikes that own the data should be constructable from just the shape and type
  // This indicates they can control data buffer
  template <typename T>
  concept OwningArrayLike = ArrayLike<T> && requires(std::vector<ssize_t> shape, DType dtype) {
    T(shape, dtype);
  };

  // NCArray* arrays have pointer axes (suboffsets may too) - so additional
  // optional contract for that
  template <typename T>
  constexpr bool get_is_pointer_axis(const T& arr, ssize_t axis) {
    if constexpr (requires { arr.is_pointer_axis(axis); }) {
      return arr.is_pointer_axis(axis);
    } else {
      // Non-NCArray* or suboffsets arrays never have pointer axis
      return false;
    }
  }

  // Likewise check if there are offsets
  template <typename T>
  constexpr const ssize_t* if_has_get_offsets(const T& arr) {
    if constexpr (requires { arr.offsets(); } &&
                  std::is_convertible_v<decltype(arr.offsets()), const ssize_t*>) {
      return arr.offsets();
    } else {
      return nullptr;
    }
  }

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
    NCA_HD static bool le(const T& a, const T& b) { return a >= b; }
    NCA_HD static T lowest() { return std::numeric_limits<T>::lowest(); }
    NCA_HD static T max() { return std::numeric_limits<T>::max(); }

    template <typename To>
    NCA_HD static To cast(const T& val) {
      if constexpr (std::is_same_v<To, T>) {
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
  };

  template <typename T> struct op_traits : BaseOpTraits<T> {};

  // Complex need comparison operations
  template <typename T>
  struct op_traits<std::complex<T>> : BaseOpTraits<T> {
    using value_type = T;
    using sum_type = std::complex<T>;
    using diff_type = std::complex<T>;
    using truediv_type = std::complex<double>;

    NCA_HD static bool greater(const std::complex<T>& a, const std::complex<T>& b) {
      if (a.real() != b.real()) {
        return a.real() > b.real();
      }
      return a.imag() > b.imag();
    }

    NCA_HD static bool ge(const std::complex<T>& a, const std::complex<T>& b) {
      if (a.real() != b.real()) {
        return a.real() > b.real();
      }
      return a.imag() >= b.imag();
    }

    NCA_HD static bool less(const std::complex<T>& a, const std::complex<T>& b) {
      if (a.real() != b.real()) {
        return a.real() < b.real();
      }
      return a.imag() < b.imag();
    }

    NCA_HD static bool le(const std::complex<T>& a, const std::complex<T>& b) {
      if (a.real() != b.real()) {
        return a.real() < b.real();
      }
      return a.imag() <= b.imag();
    }

    NCA_HD static std::complex<T> lowest() {
      return {std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest()};
    }

    NCA_HD static std::complex<T> max() {
      return {std::numeric_limits<T>::max(), std::numeric_limits<T>::max()};
    }

    template <typename To>
    NCA_HD static To cast(const std::complex<T>& val) {
      if constexpr (std::is_same_v<To, std::complex<T>>) {
        return val;
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
  };

  template <>
  struct op_traits<std::int8_t> : BaseOpTraits<std::int8_t> {
    using sum_type = std::int64_t;
    using diff_type = std::int64_t;
  };

  template <>
  struct op_traits<std::uint8_t> : BaseOpTraits<std::uint8_t> {
    using sum_type = std::uint64_t;
    // NOTE: Unsigned types promot to SIGNED for subtraction!
    using diff_type = std::int64_t;
  };

  template <>
  struct op_traits<std::int16_t> : BaseOpTraits<std::int16_t> {
    using sum_type = int64_t;
    using diff_type = int64_t;
  };

  template <>
  struct op_traits<std::uint16_t> : BaseOpTraits<std::uint16_t> {
    using sum_type = std::uint64_t;
    // NOTE: Unsigned types promot to SIGNED for subtraction!
    using diff_type = std::int64_t;
  };

  template <>
  struct op_traits<bool> : BaseOpTraits<bool> {
    using sum_type = std::uint64_t;
    using diff_type = std::int64_t;
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
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
      };
    }
    NCA_HD static Float2 max() {
      return {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
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
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
      };
    }
    NCA_HD static Float3 max() {
      return {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
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
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
      };
    }
    NCA_HD static Float4 max() {
      return {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
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
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest()
      };
    }
    NCA_HD static Double2 max() {
      return {
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()
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
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest()
      };
    }
    NCA_HD static Double3 max() {
      return {
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()
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
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest()
      };
    }
    NCA_HD static Double4 max() {
      return {
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()
      };
    }
  };


} // namespace ncarray

#endif // NCARRAY_ARRAY_TRAITS_HH
