/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_CUSTOM_TYPES_HH
#define NCARRAY_CUSTOM_TYPES_HH

#ifdef __CUDACC_RTC__
#include <cuda/std/cmath>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

// Need to `using` just this, since its global in CUDA >= 13, library <13.
#if __CUDACC_VER_MAJOR__ < 13
using cuda::std::sqrt;
#endif

typedef long long ssize_t;

namespace hd_std = cuda::std;

#else
#include <algorithm>
#include <cmath>
#include <complex> // For std::sqrt in the nca_sqrt helper
#include <concepts>
#include <ostream>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

// Need to `using` just this, since its global in CUDA >= 13, library <13.
using std::sqrt;

namespace hd_std = std;

#endif

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

#ifndef NCA_D
#ifdef __CUDACC__
#define NCA_D __device__
#else
#define NCA_D
#endif
#endif

#ifndef NCARRAY_MAX_NDIM
#define NCARRAY_MAX_NDIM 10
#endif

namespace ncarray {
  /**
   * A small struct for holding a key and value.
   *
   * Used, for instance, in GPU-based key/value reductions.
   *
   * @tparam KeyT The underlying type of the key used for lookup.
   * @tparam ValT The underlying type of the value being stored.
   */
  template <typename KeyT, typename ValT>
  struct KeyValPair {
    KeyT key;
    ValT val;

    KeyValPair() = default;
    NCA_HD KeyValPair(KeyT _key, ValT _val)
      : key(_key)
      , val(_val)
    {}

    NCA_HD inline bool operator==(const KeyValPair& other) const {
      return key == other.key && val == other.val;
    }

    NCA_HD inline bool operator!=(const KeyValPair& other) const {
      return !(*this == other);
    }
  };

  /**
   * A struct to define mapping requirements when performing axis-aware reductions.
   *
   * The fields of this struct are used to simplify traversal of input arrays and their
   * mapping (after applying the reducing operation) to the output array when performing
   * reductions along specific axes. This is not used for full reductions to scalars.
   */
  struct ReductionParams {
    ssize_t shape[NCARRAY_MAX_NDIM];
    ssize_t strides[NCARRAY_MAX_NDIM];
    ssize_t in_strides[NCARRAY_MAX_NDIM];
    int shifts[NCARRAY_MAX_NDIM];
    ssize_t masks[NCARRAY_MAX_NDIM];
    ssize_t ndim;
    double ddof { 0.0 }; ///< Delta degrees of freedom, used only for variance and stdev
  };

#ifndef __CUDACC_RTC__
  /**
   * Setup a small struct with strides, offsets and masks for reducing axes.
   *
   * @note This function is NOT compatible with device code.
   *       This is a host only function.
   *
   * @param[in] axes The user-requested axes that willbe reduced.
   * @param[in] arr_ndim The number of dimensions (axes) in the array currently.
   * @param[in] in_shape The shape of the axes in the array.
   * @param[in] in_strides The strides of the axes in the array in ELEMENTS (not bytes).
   * @param[out] new_shape The new shape for the array following reduction.
   * @param[out] new_ndim The number of dimensions after reduction.
   * @param[in] itemsize The size in bytes of the array elements data type.
   * @param[in] ddof The delta degrees of freedom -- only passed through to var/std reductions.
   * @returns params The reduction parameter table.
   */
  inline ReductionParams build_reduction_params(const std::vector<ssize_t>& axes,
                                                ssize_t arr_ndim,
                                                const ssize_t* in_shape,
                                                const ssize_t* in_strides,
                                                ssize_t (&new_shape)[NCARRAY_MAX_NDIM],
                                                ssize_t& new_ndim,
                                                ssize_t itemsize,
                                                double ddof = 0.0) {
    ReductionParams params;
    params.ddof = ddof;
    params.ndim = arr_ndim;

    // Setup a table of axes to reduce (fast in kernel)
    bool is_reduced[NCARRAY_MAX_NDIM] { false };
    for (ssize_t axis : axes) {
      if (axis >= 0 && axis < arr_ndim) {
        is_reduced[axis] = true;
      }
    }

    ssize_t current_out_stride { 1 };
    ssize_t cummulative_in_stride { 1 };

    for (ssize_t d = arr_ndim - 1; d >= 0; --d) {
      params.shape[d] = in_shape[d];
      params.in_strides[d] = in_strides[d] / itemsize;

      params.masks[d] = params.shape[d] - 1;
      params.shifts[d] = static_cast<int>(std::log2(cummulative_in_stride));

      cummulative_in_stride *= params.shape[d];

      if (is_reduced[d]) {
        params.strides[d] = 0;
      } else {
        params.strides[d] = current_out_stride;
        // Record this dimension for the result array
        new_shape[new_ndim++] = in_shape[d]; // This needs to be reversed later
        current_out_stride *= in_shape[d];
      }
    }
    std::reverse(new_shape, new_shape + new_ndim);

    return params;
  }
#endif

  /**
   * An accumulator used for the calculation of variance and standard deviation.
   */
  template <typename VarT>
  struct VarAccumulator {
    double count;
    VarT mean;
    VarT m2;

    NCA_D inline bool operator==(const VarAccumulator& other) const {
      return count == other.count && mean == other.mean && m2 == other.m2;
    }

    NCA_D inline bool operator!=(const VarAccumulator& other) const {
      return !(*this == other);
    }

    NCA_D static VarAccumulator merge(const VarAccumulator& a,
                                      const VarAccumulator& b) {
      if (a.count == 0) {
        return b;
      }
      if (b.count == 0) {
        return a;
      }

      double n { a.count + b.count };
      VarT delta { b.mean - a.mean };
      VarT m { a.mean + delta * (b.count / n) };
      VarT s { a.m2 + b.m2 + (delta * delta) * (a.count * b.count / n) };

      return { n, m, s };
    }
  };

/**
 * @def DEFINE_VECTOR_PRE_UNARY_OPS(VECDTYPE, ...)
 * @brief A helper for creating prefix unary op function definitions for vector data types.
 */
#define DEFINE_VECTOR_PRE_UNARY_OPS(VECDTYPE, ...)                   \
  NCA_HD constexpr VECDTYPE operator+() const {                      \
    return { __VA_ARGS__(+) };                                       \
  }                                                                  \
  NCA_HD constexpr VECDTYPE operator-() const {                      \
    return { __VA_ARGS__(-) };                                       \
  }

/**
 * @def DEFINE_VECTOR_INC_DEC_OPS(VECDTYPE, ...)
 * @brief A helper for creating increment/decrement function definitions for vector data types.
 */
#define DEFINE_VECTOR_INC_DEC_OPS(VECDTYPE, ...)                     \
  NCA_HD constexpr VECDTYPE& operator++() {                          \
    __VA_ARGS__(++)                                                  \
    return *this;                                                    \
  }                                                                  \
  NCA_HD constexpr VECDTYPE& operator--() {                          \
    __VA_ARGS__(--)                                                  \
    return *this;                                                    \
  }                                                                  \
  NCA_HD constexpr VECDTYPE operator++(int) {                        \
    VECDTYPE tmp = *this;                                            \
    ++(*this);                                                       \
    return tmp;                                                      \
  }                                                                  \
  NCA_HD constexpr VECDTYPE operator--(int) {                        \
    VECDTYPE tmp = *this;                                            \
    --(*this);                                                       \
    return tmp;                                                      \
  }

/**
 * @def DEFINE_VECTOR_OPS(VECDTYPE, ...)
 * @brief A helper for function definitions for inplace and binary operations between vector data types.
 */
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

  /**
   * An aliased concept for the the is_arithmetic type trait value.
   */
  template <typename T>
  concept Numeric = hd_std::is_arithmetic<T>::value;

/**
 * @def DEFINE_SCALAR_OPS(VECDTYPE, ...)
 * @brief A helper for function definitions for inplace and binary operations between
 *        vector data types and scalars (handling broadcasting as needed).
 */
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

/**
 * @def DEFINE_VECTOR_OPS(VECDTYPE, ...)
 * @brief A helper for function definitions for comparison operations
 *        between vector data types.
 */
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
  friend NCA_HD bool operator<=(const VECDTYPE& a, const VECDTYPE& b) { \
    return __VA_ARGS__(a, b, <=);                                       \
  }                                                                     \
  friend NCA_HD bool operator>(const VECDTYPE& a, const VECDTYPE& b) {  \
    return __VA_ARGS__(a, b, >);                                        \
  }                                                                     \
  friend NCA_HD bool operator>=(const VECDTYPE& a, const VECDTYPE& b) { \
    return __VA_ARGS__(a, b, >=);                                       \
  }

/**
 * @def PRUNARYOPS2(OP)
 * @brief Create the prefix unary ops for a 2-scalar vector type.
 */
#define PREUNARYOPS2(OP) OP x, OP y
/**
 * @def PRUNARYOPS3(OP)
 * @brief Create the prefix unary ops for a 3-scalar vector type.
 */
#define PREUNARYOPS3(OP) OP x, OP y, OP z
/**
 * @def PRUNARYOPS4(OP)
 * @brief Create the prefix unary ops for a 4-scalar vector type.
 */
#define PREUNARYOPS4(OP) OP x, OP y, OP z, OP w

/**
 * @def INCDECOPS2(OP)
 * @brief Create the increment/decrement ops for a 2-scalar vector type.
 */
#define INCDECOPS2(OP) OP x; OP y;
/**
 * @def INCDECOPS3(OP)
 * @brief Create the increment/decrement ops for a 3-scalar vector type.
 */
#define INCDECOPS3(OP) OP x; OP y; OP z;
/**
 * @def INCDECOPS4(OP)
 * @brief Create the increment/decrement ops for a 4-scalar vector type.
 */
#define INCDECOPS4(OP) OP x; OP y; OP z; OP w;

/**
 * @def BINOPS2(OP)
 * @brief Create the binary ops for a 2-scalar vector type.
 */
#define BINOPS2(OP) x OP other.x, y OP other.y
/**
 * @def BINOPS3(OP)
 * @brief Create the binary ops for a 3-scalar vector type.
 */
#define BINOPS3(OP) x OP other.x, y OP other.y, z OP other.z
/**
 * @def BINOPS4(OP)
 * @brief Create the binary ops for a 4-scalar vector type.
 */
#define BINOPS4(OP) x OP other.x, y OP other.y, z OP other.z, w OP other.w

/**
 * @def SCALAROPS2(OP)
 * @brief Create the op definitions for a 2-scalar vector type and a scalar.
 */
#define SCALAROPS2(v, s, OP) static_cast<decltype(v.x)>(v.x OP s), static_cast<decltype(v.y)>(v.y OP s)
/**
 * @def SCALAROPS3(OP)
 * @brief Create the op definitions for a 3-scalar vector type and a scalar.
 */
#define SCALAROPS3(v, s, OP)            \
  static_cast<decltype(v.x)>(v.x OP s), \
  static_cast<decltype(v.y)>(v.y OP s), \
  static_cast<decltype(v.z)>(v.z OP s)
/**
 * @def SCALAROPS4(OP)
 * @brief Create the op definitions for a 4-scalar vector type and a scalar.
 */
#define SCALAROPS4(v, s, OP)            \
  static_cast<decltype(v.x)>(v.x OP s), \
  static_cast<decltype(v.y)>(v.y OP s), \
  static_cast<decltype(v.z)>(v.z OP s), \
  static_cast<decltype(v.w)>(v.w OP s)

/**
 * @def COMPOPS2(OP)
 * @brief Create the comparison ops for a 2-scalar vector type.
 */
#define COMPOPS2(a, b, OP) a.x OP b.x && a.y OP b.y
/**
 * @def COMPOPS3(OP)
 * @brief Create the comparison ops for a 3-scalar vector type.
 */
#define COMPOPS3(a, b, OP) a.x OP b.x && a.y OP b.y && a.z OP b.z
/**
 * @def COMPOPS4(OP)
 * @brief Create the comparison ops for a 4-scalar vector type.
 */
#define COMPOPS4(a, b, OP) a.x OP b.x && a.y OP b.y && a.z OP b.z && a.w OP b.w

  /**
   * Determines an object that has two scalar values x and y.
   */
  template <typename T>
  concept Vector2DType = requires(T v) {
    v.x;
    v.y;
  };
  /**
   * Determines an object that has three scalar values x, y, and z.
   */
  template <typename T>
  concept Vector3DType = Vector2DType<T> && requires(T v) {
    v.z;
  };
  /**
   * Determines an object that has four scalar values x, y, z, and w.
   */
  template <typename T>
  concept Vector4DType = Vector3DType<T> && requires(T v) {
    v.w;
  };

#pragma pack(push, 8)
  /**
   * A 2-float (single precision) vector data type.
   *
   * @note This is analogous to, and layout compatible with, CUDA's float2.
   */
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
    requires (!hd_std::is_same_v<hd_std::decay_t<U>, Float2>)
    NCA_HD constexpr Float2(const U& other)
      : x(static_cast<float>(other.x))
      , y(static_cast<float>(other.y))
    {}

    NCA_HD explicit inline operator bool() const {
      return x || y;
    }

    DEFINE_VECTOR_PRE_UNARY_OPS(Float2, PREUNARYOPS2)
    DEFINE_VECTOR_INC_DEC_OPS(Float2, INCDECOPS2)
    DEFINE_VECTOR_OPS(Float2, BINOPS2)
    DEFINE_SCALAR_OPS(Float2, SCALAROPS2)
    DEFINE_VECTOR_COMPARISONS(Float2, COMPOPS2)
  };
#pragma pack(pop)

#pragma pack(push, 4)
  /**
   * A 3-float (single precision) vector data type.
   *
   * @note This is analogous to, and layout compatible with, CUDA's float3.
   */
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
    requires (!hd_std::is_same_v<hd_std::decay_t<U>, Float3>)
    NCA_HD constexpr Float3(const U& other)
      : x(static_cast<float>(other.x))
      , y(static_cast<float>(other.y))
    {
      if constexpr (Vector3DType<U>) {
        z = static_cast<float>(other.z);
      }
    }

    NCA_HD explicit inline operator bool() const {
      return x || y || z;
    }

    DEFINE_VECTOR_PRE_UNARY_OPS(Float3, PREUNARYOPS3)
    DEFINE_VECTOR_INC_DEC_OPS(Float3, INCDECOPS3)
    DEFINE_VECTOR_OPS(Float3, BINOPS3)
    DEFINE_SCALAR_OPS(Float3, SCALAROPS3)
    DEFINE_VECTOR_COMPARISONS(Float3, COMPOPS3)
  };
#pragma pack(pop)

#pragma pack(push, 8)
  /**
   * A 4-float (single precision) vector data type.
   *
   * @note This is analogous to, and layout compatible with, CUDA's float4.
   */
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
    requires (!hd_std::is_same_v<hd_std::decay_t<U>, Float4>)
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

    NCA_HD explicit inline operator bool() const {
      return x || y || z || w;
    }

    DEFINE_VECTOR_PRE_UNARY_OPS(Float4, PREUNARYOPS4)
    DEFINE_VECTOR_INC_DEC_OPS(Float4, INCDECOPS4)
    DEFINE_VECTOR_OPS(Float4, BINOPS4)
    DEFINE_SCALAR_OPS(Float4, SCALAROPS4)
    DEFINE_VECTOR_COMPARISONS(Float4, COMPOPS4)
  };
#pragma pack(pop)

#pragma pack(push, 16)
  /**
   * A 2-double (double precision) vector data type.
   *
   * @note This is analogous to, and layout compatible with, CUDA's double2.
   */
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
    requires (!hd_std::is_same_v<hd_std::decay_t<U>, Double2>)
    NCA_HD constexpr Double2(const U& other)
      : x(static_cast<double>(other.x))
      , y(static_cast<double>(other.y))
    {}

    NCA_HD explicit inline operator bool() const {
      return x || y;
    }

    DEFINE_VECTOR_PRE_UNARY_OPS(Double2, PREUNARYOPS2)
    DEFINE_VECTOR_INC_DEC_OPS(Double2, INCDECOPS2)
    DEFINE_VECTOR_OPS(Double2, BINOPS2)
    DEFINE_SCALAR_OPS(Double2, SCALAROPS2)
    DEFINE_VECTOR_COMPARISONS(Double2, COMPOPS2)
  };
#pragma pack(pop)

#pragma pack(push, 8)
  /**
   * A 3-double (double precision) vector data type.
   *
   * @note This is analogous to, and layout compatible with, CUDA's double3.
   */
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
    requires (!hd_std::is_same_v<hd_std::decay_t<U>, Double3>)
    NCA_HD constexpr Double3(const U& other)
      : x(static_cast<double>(other.x))
      , y(static_cast<double>(other.y))
    {
      if constexpr (Vector3DType<U>) {
        z = static_cast<double>(other.z);
      }
    }

    NCA_HD explicit inline operator bool() const {
      return x || y || z;
    }

    DEFINE_VECTOR_PRE_UNARY_OPS(Double3, PREUNARYOPS3)
    DEFINE_VECTOR_INC_DEC_OPS(Double3, INCDECOPS3)
    DEFINE_VECTOR_OPS(Double3, BINOPS3)
    DEFINE_SCALAR_OPS(Double3, SCALAROPS3)
    DEFINE_VECTOR_COMPARISONS(Double3, COMPOPS3)
  };
#pragma pack(pop)

#pragma pack(push, 16)
  /**
   * A 4-double (double precision) vector data type.
   *
   * @note This is analogous to, and layout compatible with, CUDA's double4.
   * @note The double4 is deprected in CUDA 13+. This type is not however.
   */
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
    requires (!hd_std::is_same_v<hd_std::decay_t<U>, Double4>)
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

    NCA_HD explicit inline operator bool() const {
      return x || y || z || w;
    }

    DEFINE_VECTOR_PRE_UNARY_OPS(Double4, PREUNARYOPS4)
    DEFINE_VECTOR_INC_DEC_OPS(Double4, INCDECOPS4)
    DEFINE_VECTOR_OPS(Double4, BINOPS4)
    DEFINE_SCALAR_OPS(Double4, SCALAROPS4)
    DEFINE_VECTOR_COMPARISONS(Double4, COMPOPS4)
  };
#pragma pack(pop)

#undef PREUNARYOPS2
#undef PREUNARYOPS3
#undef PREUNARYOPS4
#undef INCDECOPS2
#undef INCDECOPS3
#undef INCDECOPS4
#undef BINOPS4
#undef BINOPS3
#undef BINOPS2
#undef SCALAROPS4
#undef SCALAROPS3
#undef SCALAROPS2
#undef COMPOPS4
#undef COMPOPS3
#undef COMPOPS2
#undef DEFINE_VECTOR_PRE_UNARY_OPS
#undef DEFINE_VECTOR_INC_DEC_OPS
#undef DEFINE_SCALAR_OPS
#undef DEFINE_VECTOR_OPS
#undef DEFINE_VECTOR_COMPARISONS

#ifndef __CUDACC_RTC__
  /**
   * A formatting helper for printing vector datatypes.
   *
   * @note This function is NOT compatible with device code.
   *       This is a host only function.
   *
   * @tparam T The input datatype, requires at least Vector2DType.
   * @param[out] oss The stream to write the contents to.
   * @param[in] vec The vector data type to print to the stream.
   */
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
#endif

  /**
   * A helper for setting up square roots on various types.
   *
   * @note The warning suppression is to silence otherwise extremely verbose output.
   *
   * @tparam T The datatype of the value to take the square root of.
   * @param[in] val The value to take the square root of.
   */
  template <typename T>
#ifdef __CUDACC__
  #pragma nv_diag_suppress 20011, 20014
#endif
  NCA_HD inline T nca_sqrt(const T& val) {
    return sqrt(val);
  }
#ifdef __CUDACC__
  #pragma nv_diag_default 20011, 20014
#endif

  /**
   * The square root overload for handling the elementwise square-root of vector types.
   *
   * @tparam T The datatype of the value to take the square root of, requires at least Vector2DType.
   * @param[in] val The vector value to take the elementwise square root of.
   */
  template <Vector2DType T>
  NCA_HD inline T nca_sqrt(const T& v) {
    if constexpr (Vector4DType<T>) {
      return T { sqrt(v.x), sqrt(v.y), sqrt(v.z), sqrt(v.w) };
    } else if constexpr (Vector3DType<T>) {
      return T { sqrt(v.x), sqrt(v.y), sqrt(v.z) };
    } else {
      return T { sqrt(v.x), sqrt(v.y) };
    }
  }

} // namespace ncarray
#endif // NCARRAY_CUSTOM_TYPES_HH
