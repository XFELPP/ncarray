#ifndef NCARRAY_HOST_CASTS_HH
#define NCARRAY_HOST_CASTS_HH

#include "ncarray/custom_types.hh"

#include <complex>
#include <cstdint>

#ifdef _WIN32
// On Windows need to export symbols for DLLs
#ifdef NCA_BUILD_NCARRAY_API
#define NCA_CAST_API __declspec(dllexport)
#define NCA_EXTERN_CAST_API
#else
#define NCA_CAST_API __declspec(dllimport)
#define NCA_EXTERN_CAST_API __declspec(dllimport)
#endif
#else
#define NCA_CAST_API
#define NCA_EXTERN_CAST_API
#endif

namespace ncarray {
  namespace host {
    template <typename T>
    struct VM_Cast_Table {
      using Fn = T (*)(const void*);

      static const Fn tbl[22];

      inline Fn operator[](int idx) const { return tbl[idx]; }
    };

    template <typename T>
    inline constexpr VM_Cast_Table<T> vm_cast_table {};

    #define EXTERN_VM_CAST_TABLE(T)                                \
      extern template struct NCA_EXTERN_CAST_API VM_Cast_Table<T>;

    EXTERN_VM_CAST_TABLE(bool)
    EXTERN_VM_CAST_TABLE(char)

    EXTERN_VM_CAST_TABLE(std::uint8_t)
    EXTERN_VM_CAST_TABLE(std::uint16_t)
    EXTERN_VM_CAST_TABLE(std::uint32_t)
    EXTERN_VM_CAST_TABLE(std::uint64_t)

    EXTERN_VM_CAST_TABLE(std::int8_t)
    EXTERN_VM_CAST_TABLE(std::int16_t)
    EXTERN_VM_CAST_TABLE(std::int32_t)
    EXTERN_VM_CAST_TABLE(std::int64_t)

    EXTERN_VM_CAST_TABLE(float)
    EXTERN_VM_CAST_TABLE(double)
    EXTERN_VM_CAST_TABLE(long double)

    EXTERN_VM_CAST_TABLE(std::complex<float>)
    EXTERN_VM_CAST_TABLE(std::complex<double>)
    EXTERN_VM_CAST_TABLE(std::complex<long double>)

    EXTERN_VM_CAST_TABLE(Float2)
    EXTERN_VM_CAST_TABLE(Float3)
    EXTERN_VM_CAST_TABLE(Float4)

    EXTERN_VM_CAST_TABLE(Double2)
    EXTERN_VM_CAST_TABLE(Double3)
    EXTERN_VM_CAST_TABLE(Double4)

    #undef EXTERN_VM_CAST_TABLE
  }
}
#endif // NCARRAY_HOST_CASTS_HH
