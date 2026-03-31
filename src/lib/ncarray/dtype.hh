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

#include <complex>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <variant>

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
  class type_error : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
  };

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
  //REGISTER_NCARRAY_DTYPE(unsigned char, uchar) // Stuck cause of uint8_t
  REGISTER_NCARRAY_DTYPE(std::uint8_t, uint8)
  REGISTER_NCARRAY_DTYPE(std::uint16_t, uint16)
  REGISTER_NCARRAY_DTYPE(std::uint32_t, uint32)
  REGISTER_NCARRAY_DTYPE(std::uint64_t, uint64)
  REGISTER_NCARRAY_DTYPE(char, char_)
  REGISTER_NCARRAY_DTYPE(std::int8_t, int8)
  REGISTER_NCARRAY_DTYPE(std::int16_t, int16)
  REGISTER_NCARRAY_DTYPE(std::int32_t, int32)
  REGISTER_NCARRAY_DTYPE(std::int64_t, int64)
  REGISTER_NCARRAY_DTYPE(float, float32)
  REGISTER_NCARRAY_DTYPE(double, float64)

  REGISTER_NCARRAY_DTYPE(std::complex<float>, complex64)
  REGISTER_NCARRAY_DTYPE(std::complex<double>, complex128)
  REGISTER_NCARRAY_DTYPE(std::complex<long double>, complex256)

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
    std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t,
    std::int8_t, std::int16_t, std::int32_t, std::int64_t,
    float, double, std::complex<float>, std::complex<double>, std::complex<long double>,
    Float2, Float3, Float4, Double2, Double3, Double4
  >;

  template <typename T> Scalar
  to_scalar(const void* ptr) {
    return Scalar(*reinterpret_cast<const T*>(ptr));
  }
} // namespace ncarray

#endif // NCARRAY_DTYPE_HH
