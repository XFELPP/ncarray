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

using cuda::std::complex;

using cuda::std::int8_t;
using cuda::std::int16_t;
using cuda::std::int32_t;
using cuda::std::int64_t;

using cuda::std::uint8_t;
using cuda::std::uint16_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;

#else
#include <cassert>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
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

  enum class DType {
    bool_,
    char_,
    //uchar, // Stuck cause of uint8_t
    uint8,
    uint16,
    uint32,
    uint64,
    int8,
    int16,
    int32,
    int64,
    float32,
    float64,
    float128,
    complex64,
    complex128,
    complex256,
    vfloat2,
    vfloat3,
    vfloat4,
    vdouble2,
    vdouble3,
    vdouble4
  };

  template <typename T> struct dtype_traits;
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
    case DType::float128:
    case DType::complex128:
      return 16;
    case DType::complex256:
      return 32;
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

  using Scalar = std::variant<
    // Cannot use unsigned char, identical to std::uint8_t
    bool, char,
    uint8_t, uint16_t, uint32_t, uint64_t,
    int8_t, int16_t, int32_t, int64_t,
    float, double, long double,
    complex<float>, complex<double>, complex<long double>,
    Float2, Float3, Float4, Double2, Double3, Double4
  >;

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

  template <typename Visitor>
  NCA_HD inline auto dispatch(DType type, Visitor&& visitor) {
    switch (type) {
    case DType::bool_: {
      return visitor.template operator()<bool>();
    }
    case DType::char_: {
      return visitor.template operator()<char>();
    }
    // Stuck becuase of std::uint8_t
    // case DType::uchar: {
    //  return visitor.template operator()<unsigned char>();
    //}
    case DType::uint8: {
      return visitor.template operator()<uint8_t>();
    }
    case DType::uint16: {
      return visitor.template operator()<uint16_t>();
    }
    case DType::uint32: {
      return visitor.template operator()<uint32_t>();
    }
    case DType::uint64: {
      return visitor.template operator()<uint64_t>();
    }
    case DType::int8: {
      return visitor.template operator()<int8_t>();
    }
    case DType::int16: {
      return visitor.template operator()<int16_t>();
    }
    case DType::int32: {
      return visitor.template operator()<int32_t>();
    }
    case DType::int64: {
      return visitor.template operator()<int64_t>();
    }
    case DType::float32: {
      return visitor.template operator()<float>();
    }
    case DType::float64: {
      return visitor.template operator()<double>();
    }
    case DType::float128: {
      return visitor.template operator()<long double>();
    }
    case DType::complex64: {
      return visitor.template operator()<complex<float>>();
    }
    case DType::complex128: {
      return visitor.template operator()<complex<double>>();
    }
    case DType::complex256: {
      return visitor.template operator()<complex<long double>>();
    }
    case DType::vfloat2: {
      return visitor.template operator()<Float2>();
    }
    case DType::vfloat3: {
      return visitor.template operator()<Float3>();
    }
    case DType::vfloat4: {
      return visitor.template operator()<Float4>();
    }
    case DType::vdouble2: {
      return visitor.template operator()<Double2>();
    }
    case DType::vdouble3: {
      return visitor.template operator()<Double3>();
    }
    case DType::vdouble4: {
      return visitor.template operator()<Double4>();
    }
    default:
      assert(false && "Invalid DType for operation!");
      return visitor.template operator()<float>();
    }
  }
} // namespace ncarray

#endif // NCARRAY_DTYPE_HH
