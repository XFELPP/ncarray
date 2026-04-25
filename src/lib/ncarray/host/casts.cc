#include "ncarray/host/casts.hh"

#include "ncarray/custom_types.hh"
#include "ncarray/op_traits.hh"

#include <cstdint>
#include <complex>

namespace ncarray {
  namespace host {
    template <typename DestT, typename SrcT>
    DestT vm_cast_logic(const void* ptr) {
      return op_traits<SrcT>::template cast<DestT>(*static_cast<const SrcT*>(ptr));
    }

    template <typename T>
    const typename VM_Cast_Table<T>::Fn VM_Cast_Table<T>::tbl[22] = {
      vm_cast_logic<T, bool>,
      vm_cast_logic<T, char>,

      vm_cast_logic<T, std::uint8_t>,
      vm_cast_logic<T, std::uint16_t>,
      vm_cast_logic<T, std::uint32_t>,
      vm_cast_logic<T, std::uint64_t>,

      vm_cast_logic<T, std::int8_t>,
      vm_cast_logic<T, std::int16_t>,
      vm_cast_logic<T, std::int32_t>,
      vm_cast_logic<T, std::int64_t>,

      vm_cast_logic<T, float>,
      vm_cast_logic<T, double>,
      vm_cast_logic<T, long double>,

      vm_cast_logic<T, std::complex<float>>,
      vm_cast_logic<T, std::complex<double>>,
      vm_cast_logic<T, std::complex<long double>>,

      vm_cast_logic<T, Float2>,
      vm_cast_logic<T, Float3>,
      vm_cast_logic<T, Float4>,

      vm_cast_logic<T, Double2>,
      vm_cast_logic<T, Double3>,
      vm_cast_logic<T, Double4>
    };

    #define DEFINE_VM_CAST_TABLE(T)     \
      template struct VM_Cast_Table<T>;

    DEFINE_VM_CAST_TABLE(bool)
    DEFINE_VM_CAST_TABLE(char)

    DEFINE_VM_CAST_TABLE(std::uint8_t)
    DEFINE_VM_CAST_TABLE(std::uint16_t)
    DEFINE_VM_CAST_TABLE(std::uint32_t)
    DEFINE_VM_CAST_TABLE(std::uint64_t)

    DEFINE_VM_CAST_TABLE(std::int8_t)
    DEFINE_VM_CAST_TABLE(std::int16_t)
    DEFINE_VM_CAST_TABLE(std::int32_t)
    DEFINE_VM_CAST_TABLE(std::int64_t)

    DEFINE_VM_CAST_TABLE(float)
    DEFINE_VM_CAST_TABLE(double)
    DEFINE_VM_CAST_TABLE(long double)

    DEFINE_VM_CAST_TABLE(std::complex<float>)
    DEFINE_VM_CAST_TABLE(std::complex<double>)
    DEFINE_VM_CAST_TABLE(std::complex<long double>)

    DEFINE_VM_CAST_TABLE(Float2)
    DEFINE_VM_CAST_TABLE(Float3)
    DEFINE_VM_CAST_TABLE(Float4)

    DEFINE_VM_CAST_TABLE(Double2)
    DEFINE_VM_CAST_TABLE(Double3)
    DEFINE_VM_CAST_TABLE(Double4)

    #undef DEFINE_VM_CAST_TABLE
  } // namespace host
} // namespace ncarray
