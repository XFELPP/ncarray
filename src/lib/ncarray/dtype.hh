/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_DTYPE_HH
#define NCARRAY_DTYPE_HH

#include "ncarray/custom_types.hh"

#ifdef __CUDACC_RTC__
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

using cuda::std::complex;

using cuda::std::int8_t;
using cuda::std::int16_t;
using cuda::std::int32_t;
using cuda::std::int64_t;

using cuda::std::uint8_t;
using cuda::std::uint16_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;

using cuda::std::is_same;
using cuda::std::disjunction;
using cuda::std::is_void_v;

using cuda::std::forward;

#else
#include <cassert>
#include <complex>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

using std::complex;

using std::int8_t;
using std::int16_t;
using std::int32_t;
using std::int64_t;

using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;

using std::is_same;
using std::disjunction;
using std::is_void_v;

using std::forward;

#endif

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
#ifndef __CUDACC_RTC__
  class type_error : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
  };
#endif

  /**
   * The main identifier for the ncarray type system. The DType determines array element type.
   */
  enum class DType {
    bool_,      ///< Boolean type
    char_,      ///< `char` type (signed 8-bit)
    //uchar, // Stuck cause of uint8_t
    uint8,      ///< 8-bit unsigned integer
    uint16,     ///< 16-bit unsigned integer
    uint32,     ///< 32-bit unsigned integer
    uint64,     ///< 64-bit unsigned integer
    int8,       ///< 8-bit signed integer
    int16,      ///< 16-bit signed integer
    int32,      ///< 32-bit signed integer
    int64,      ///< 64-bit signed integer
    float32,    ///< single precision float
    float64,    ///< double precision float
    float128,   ///< equivalent to long double
    complex64,  ///< complex<float>
    complex128, ///< complex<double>
    complex256, ///< complex<long double>
    vfloat2,    ///< 2-float vector type (Float2)
    vfloat3,    ///< 3-float vector type (Float3)
    vfloat4,    ///< 4-float vector type (Float4)
    vdouble2,   ///< 2-double vector type (Double2)
    vdouble3,   ///< 3-double vector tyep (Double3)
    vdouble4    ///< 4-double vector type (Double4)
  };

  /**
   * A small struct that maps a type to the DType enumerator via the `value` member.
   */
  template <typename T> struct dtype_traits;

  /**
   * @def REGISTER_NCARRAY_DTYPE(TYPE, ENUM_VAL)
   * @brief Macro to create dtype_traits specializations.
   */
#define REGISTER_NCARRAY_DTYPE(TYPE, ENUM_VAL)      \
  template <> struct dtype_traits<TYPE> {           \
    static constexpr DType value = DType::ENUM_VAL; \
    using type = TYPE;                              \
  };

  REGISTER_NCARRAY_DTYPE(bool, bool_)
  REGISTER_NCARRAY_DTYPE(char, char_)
  //REGISTER_NCARRAY_DTYPE(unsigned char, uchar) // Stuck cause of uint8_t
  REGISTER_NCARRAY_DTYPE(uint8_t, uint8)
  REGISTER_NCARRAY_DTYPE(uint16_t, uint16)
  REGISTER_NCARRAY_DTYPE(uint32_t, uint32)
  REGISTER_NCARRAY_DTYPE(uint64_t, uint64)

  REGISTER_NCARRAY_DTYPE(int8_t, int8)
  REGISTER_NCARRAY_DTYPE(int16_t, int16)
  REGISTER_NCARRAY_DTYPE(int32_t, int32)
  REGISTER_NCARRAY_DTYPE(int64_t, int64)

  REGISTER_NCARRAY_DTYPE(float, float32)
  REGISTER_NCARRAY_DTYPE(double, float64)
  REGISTER_NCARRAY_DTYPE(long double, float128)

  REGISTER_NCARRAY_DTYPE(complex<float>, complex64)
  REGISTER_NCARRAY_DTYPE(complex<double>, complex128)
  REGISTER_NCARRAY_DTYPE(complex<long double>, complex256)

  REGISTER_NCARRAY_DTYPE(Float2, vfloat2)
  REGISTER_NCARRAY_DTYPE(Float3, vfloat3)
  REGISTER_NCARRAY_DTYPE(Float4, vfloat4)

  REGISTER_NCARRAY_DTYPE(Double2, vdouble2)
  REGISTER_NCARRAY_DTYPE(Double3, vdouble3)
  REGISTER_NCARRAY_DTYPE(Double4, vdouble4)
#undef REGISTER_NCARRAY_DTYPE

  /**
   * Returns the size in bytes of the provided DType.
   *
   * @note In the case of NVRTC JIT compiled code, quad precision is converted to
   *       double as it is not-supported. This exceptional case is preprocessor protected.
   *
   * @todo Formally enforce these guarantees. This *should* be correct in all currently
   *       supported use cases -- but its not guaranteed. The fixed sizes listed here
   *       need to be handled (via whatever types required) in the event they are not
   *       satisfied in a particular case/on a particular platform.
   *
   * @param[in] type The input DType.
   * @returns The size in bytes.
   */
  NCA_HD inline size_t itemsize(DType type) {
    switch (type) {
    case DType::bool_:
    case DType::char_:
    case DType::uint8:
    case DType::int8:
      return 1;
    case DType::uint16:
    case DType::int16:
      return 2;
    case DType::uint32:
    case DType::int32:
    case DType::float32:
      return 4;
    case DType::uint64:
    case DType::int64:
    case DType::float64:
    case DType::complex64:
      return 8;
    case DType::float128: {
#ifdef __CUDACC_RTC__
      // For NVRTC must use double (double), not quad precision (long double)
      // This should hopefully have aready been mapped to DType::float64
      // but in case it made it through, make sure to return correct size
      return 8;
#else
      return 16;
#endif
    }
    case DType::complex128:
      return 16;
    case DType::complex256: {
#ifdef __CUDACC_RTC__
      // For NVRTC must use double (complex<double>), not quad precision (complex<long double>)
      // This should hopefully have aready been mapped to DType::complex128
      // but in case it made it through, make sure to return correct size
      return 16;
#else
      return 32;
#endif
    }
    case DType::vfloat2:
      return 8;
    case DType::vfloat3:
      return 12;
    case DType::vfloat4:
      return 16;
    case DType::vdouble2:
      return 16;
    case DType::vdouble3:
      return 24;
    case DType::vdouble4:
      return 32;
    default:
      return 0;
    }
  }

#ifndef __CUDACC_RTC__
  /**
   * Returns the string representation of the DType for printing.
   *
   * @note This function is NOT compatible with device code.
   *       This is a host only function.
   *
   * @param[in] type The input DType.
   * @returns The string representation of the data type.
   */
  inline std::string to_string(DType type) {
    switch (type) {
    case DType::bool_:
      return "bool";
    case DType::char_:
      return "char";
    case DType::int8:
      return "int8";
    case DType::int16:
      return "int16";
    case DType::int32:
      return "int32";
    case DType::int64:
      return "int64";
    case DType::uint8:
      return "uint8";
    case DType::uint16:
      return "uint16";
    case DType::uint32:
      return "uint32";
    case DType::uint64:
      return "uint64";
    case DType::float32:
      return "float32";
    case DType::float64:
      return "float64";
    case DType::float128:
      return "float128";
    case DType::complex64:
      return "complex64";
    case DType::complex128:
      return "complex128";
    case DType::complex256:
      return "complex256";
    case DType::vfloat2:
      return "vfloat2";
    case DType::vfloat3:
      return "vfloat3";
    case DType::vfloat4:
      return "vfloat4";
    case DType::vdouble2:
      return "vdouble2";
    case DType::vdouble3:
      return "vdouble3";
    case DType::vdouble4:
      return "vdouble4";
    default:
      throw type_error("Unkonwn type!");
    }
  }

  /**
   * The set of scalar types used by ncarray - one for each DType.
   * @note This function is NOT compatible with device code.
   *       This is a host only function.
   */
  using Scalar = std::variant<
    // Cannot use unsigned char, identical to std::uint8_t
    bool, char,
    uint8_t, uint16_t, uint32_t, uint64_t,
    int8_t, int16_t, int32_t, int64_t,
    float, double, long double,
    complex<float>, complex<double>, complex<long double>,
    Float2, Float3, Float4, Double2, Double3, Double4
  >;

  /**
   * Convert a pointer to an array element to a Scalar.
   *
   * @note This function is NOT compatible with device code.
   *       This is a host only function.
   *
   * @tparam T The underlying type of the array element.
   * @param[in] ptr The pointer to the array element.
   * @returns The Scalar wrapper.
   */
  template <typename T>
  Scalar to_scalar(const void* ptr) {
    return Scalar(*reinterpret_cast<const T*>(ptr));
  }
#endif

  /**
   * A restricted dispatcher for integer types. This is useful when using arrays
   * to hold indices for other arrays.
   */
  template <typename Visitor>
  NCA_HD inline auto dispatch_integers(DType type, Visitor&& visitor) {
    switch (type) {
    case DType::uint32: {
      return visitor.template operator()<uint32_t>();
    }
    case DType::uint64: {
      return visitor.template operator()<uint64_t>();
    }
    case DType::int32: {
      return visitor.template operator()<int32_t>();
    }
    case DType::int64: {
      return visitor.template operator()<int64_t>();
    }
    default: {
      assert(false && "Invalid DType for operation! Only integers accepted!");
      return visitor.template operator()<uint32_t>();
    }
    }
  }

  /**
   * Basic type list.
   */
  template <typename... Ts>
  struct type_list {};

  /**
   * Struct for testing presence of a type in a type_list.
   *
   * @tparam T The object under test.
   * @tparam List The list to check for the inclusion of T.
   */
  template <typename T, typename List>
  struct is_in_list;

  template <typename T, typename... Ts>
  struct is_in_list<T, type_list<Ts...>> : disjunction<is_same<T, Ts>...> {};

  /**
   * Whether a requested type is in the specified type_list.
   *
   * @tparam T The object under test.
   * @tparam List The list to check for the inclusion of T.
   */
  template <typename T, typename List>
  constexpr bool is_in_type_list_v = is_in_list<T, List>::value;

  /**
   * All supported base datatypes in ncarray.
   *
   * This includes basic C++ types, complex numbers and vector types.
   *
   * @note Due to quad precision not being supported, long double is not included
   *       when compiling with NVRTC, and neither is complex<long double>. nvcc
   *       seemingly transparently converts these to double; however, NVRTC does not
   *       and this results in compilation errors, or hard to track down crashes if
   *       not explicitly removed.
   */
#ifdef __CUDACC_RTC__
  // For NVRTC (and really device code, generally) long double does not work
  // nvcc generally maps this automatically to double, but must exclude both
  // long double, and complex<long double> from NVRTC code
  using base_types = type_list<
    bool, char,
    uint8_t, uint16_t, uint32_t, uint64_t,
    int8_t, int16_t, int32_t, int64_t,
    float, double, // No long double
    complex<float>, complex<double>, // No complex<long double>
    Float2, Float3,  Float4, Double2, Double3, Double4>;
#else
  using base_types = type_list<
    bool, char,
    uint8_t, uint16_t, uint32_t, uint64_t,
    int8_t, int16_t, int32_t, int64_t,
    float, double, long double,
    complex<float>, complex<double>, complex<long double>,
    Float2, Float3, Float4, Double2, Double3, Double4>;
#endif

  /**
   * A struct for concatenating type_lists together.
   *
   * @tparam Lists The type lists to concatenate.
   */
  template <typename... Lists>
  struct concat;

  template <typename... Ts, typename... Us, typename... Rest>
  struct concat<type_list<Ts...>, type_list<Us...>, Rest...> {
    using type = typename concat<type_list<Ts..., Us...>, Rest...>::type;
  };

  template <typename List>
  struct concat<List> {
    using type = List;
  };

  /**
   * The full type of a set of type_lists concatenated with concat.
   */
  template <typename... Lists>
  using concat_t = typename concat<Lists...>::type;

  /**
   * A helper struct to wrap a templated class.
   *
   * This takes the templated class and a type_list creating a list of the class
   * having been specialized by all types in the list.
   */
  template <template <typename> typename Wrapper, typename List>
  struct wrap_list;

  template <template <typename> typename Wrapper, typename... Ts>
  struct wrap_list<Wrapper, type_list<Ts...>> {
    using type = type_list<Wrapper<Ts>...>;
  };

  using accumulator_types = typename wrap_list<VarAccumulator, base_types>::type;

  /**
   * The KeyValPair type used for reductions in ncarray.
   */
  template <typename T>
  using IndexedKVP = KeyValPair<ssize_t, T>;

  using keyval_types = typename wrap_list<IndexedKVP, base_types>::type;

  /**
   * Provide supporting types for the ArrayElementProxy restricted conversion operators.
   *
   * Due to differences in definitions of fixed-width ints on host and device (std,
   * vs cuda::std), the ArrayElementProxy concept on the operator T& may delete needed
   * operators inside kernels. This type_list adds back the requisite types.
   */
  using device_int_compat_types =
    type_list<long, unsigned long, long long, unsigned long long>;

  /**
   * The set of all types in ncarray, including accumulators and the base dtypes.
   */
  using all_supported_types =
    concat_t<base_types, accumulator_types, keyval_types, device_int_compat_types>;

  /**
   * A dispatch engine for matching a DType to the underlying type and running a function.
   */
  template <typename List>
  struct list_dispatcher;

  // Silence very verbose warning about use of unqualified std::forward below
#if defined(__clang__)
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunqualified-std-cast-call"
#endif

#ifdef __CUDACC__
  // Silence warning: Calling __host__ __device__ from host code (dispatch called in GPUEngine)
  #pragma nv_diag_suppress 20014
#endif

  template <typename T, typename... Ts>
  struct list_dispatcher<type_list<T, Ts...>> {
    template <typename Visitor>
    NCA_HD static auto dispatch(DType type, Visitor&& visitor) {
      if (dtype_traits<T>::value == type) {
        return visitor.template operator()<T>();
      }
      if constexpr (sizeof...(Ts) > 0) {
        return list_dispatcher<type_list<Ts...>>::dispatch(type, forward<Visitor>(visitor));
      } else {
        using ReturnT = decltype(visitor.template operator()<T>());
        if constexpr (!is_void_v<ReturnT>) {
          return ReturnT{};
        }
      }
    }
  };
  template <>
  struct list_dispatcher<type_list<>> {
    template <typename Visitor>
    NCA_HD static auto dispatch(DType, Visitor&&) {}
  };

  /**
   * Dispatch a visitor/lambda based on DType.
   *
   * This function dispatches only on the base_types (this controls combinatorial
   * explosion to some degree). In certain places, broader dispatching may be
   * desired. In that case, using list_dispatcher directly with specialized
   * type lists can be done instead.
   *
   * @note In device code, long double is not supported. nvcc generally maps this to
   *       double automatically. NVRTC does not. We explicitly map long double to double
   *       and complex<long double> to complex<double> for NVRTC compiled code.
   *
   * @tparam Visitor The type of the visitor to be dispatched based on DType.
   * @param[in] type The DType to use for dispatching.
   * @param[in] visitor The lambda visitor.
   * @returns Whatever is returned by the visitor.
   */
  template <typename Visitor>
  NCA_HD inline auto dispatch(DType type, Visitor&& visitor) {
#ifdef __CUDACC_RTC__
    if (type == DType::float128) {
      return list_dispatcher<base_types>::dispatch(DType::float64,
                                                   forward<Visitor>(visitor));
    }
    if (type == DType::complex256) {
      return list_dispatcher<base_types>::dispatch(DType::complex128,
                                                   forward<Visitor>(visitor));
    }
#else
    return list_dispatcher<base_types>::dispatch(type, forward<Visitor>(visitor));
#endif
  }

#ifdef __CUDACC__
  #pragma nv_diag_default 20014
#endif

#if defined(__clang__)
  #pragma clang diagnostic pop
#endif

} // namespace ncarray

#endif // NCARRAY_DTYPE_HH
