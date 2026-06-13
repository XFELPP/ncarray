#ifndef NCARRAY_DEVICE_CASTS_CUH
#define NCARRAY_DEVICE_CASTS_CUH

#include "ncarray/custom_types.hh"
#include "ncarray/op_traits.hh"

#ifdef __CUDACC_RTC__
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
#include <complex>
#include <cstdint>

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

namespace ncarray {
  namespace device {


    template <typename DestT, typename SrcT>
    __device__ inline DestT vm_cast_logic(const void* ptr) {
        return op_traits<SrcT>::template cast<DestT>(*static_cast<const SrcT*>(ptr));
    }

    template <typename DestT>
    __device__ inline DestT device_cast(int src_idx, const void* ptr) {
      switch (static_cast<DType>(src_idx)) {
      case DType::bool_: {
        return vm_cast_logic<DestT, bool>(ptr);
      }
      case DType::char_: {
        return vm_cast_logic<DestT, char>(ptr);
      }

      case DType::uint8: {
        return vm_cast_logic<DestT, uint8_t>(ptr);
      }
      case DType::uint16: {
        return vm_cast_logic<DestT, uint16_t>(ptr);
      }
      case DType::uint32: {
        return vm_cast_logic<DestT, uint32_t>(ptr);
      }
      case DType::uint64: {
        return vm_cast_logic<DestT, uint64_t>(ptr);
      }

      case DType::int8: {
        return vm_cast_logic<DestT, int8_t>(ptr);
      }
      case DType::int16: {
        return vm_cast_logic<DestT, int16_t>(ptr);
      }
      case DType::int32: {
        return vm_cast_logic<DestT, int32_t>(ptr);
      }
      case DType::int64: {
        return vm_cast_logic<DestT, int64_t>(ptr);
      }

      case DType::float32: {
        return vm_cast_logic<DestT, float>(ptr);
      }
      case DType::float64: {
        return vm_cast_logic<DestT, double>(ptr);
      }
      case DType::float128: {
        return vm_cast_logic<DestT, long double>(ptr);
      }

      case DType::complex64: {
        return vm_cast_logic<DestT, complex<float>>(ptr);
      }
      case DType::complex128: {
        return vm_cast_logic<DestT, complex<double>>(ptr);
      }
      case DType::complex256: {
        return vm_cast_logic<DestT, complex<long double>>(ptr);
      }

      case DType::vfloat2: {
        return vm_cast_logic<DestT, Float2>(ptr);
      }
      case DType::vfloat3: {
        return vm_cast_logic<DestT, Float3>(ptr);
      }
      case DType::vfloat4: {
        return vm_cast_logic<DestT, Float4>(ptr);
      }

      case DType::vdouble2: {
        return vm_cast_logic<DestT, Double2>(ptr);
      }
      case DType::vdouble3: {
        return vm_cast_logic<DestT, Double3>(ptr);
      }
      case DType::vdouble4: {
        return vm_cast_logic<DestT, Double4>(ptr);
      }
      default: {
        return DestT{};
      }
      }
    }
  } // namespace device
}
#endif // NCARRAY_DEVICE_CASTS_CUH
