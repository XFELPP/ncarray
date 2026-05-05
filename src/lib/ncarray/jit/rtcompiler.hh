/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_JIT_RTCOMPILER_HH
#define NCARRAY_JIT_RTCOMPILER_HH

// nvrtcGetTypeName is not available by default.
// Define macro to non-zero to make it available
#define NVRTC_GET_TYPE_NAME 1

#include "ncarray/array_impl.hh"
#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"
#ifdef __CUDACC__
#include "ncarray/device/casts.cuh"
#endif
#include "ncarray/dtype.hh"
#include "ncarray/host/casts.hh"
#include "ncarray/op_code.hh"
#include "ncarray/op_traits.hh"
#include "ncarray/mvnode.hh"

#include <cuda.h>
#include <nvrtc.h>

#ifdef _WIN32
#include <BaseTsd.h>
#include <windows.h>
typedef SSIZE_T ssize_t;
#else
#include <dlfcn.h>
#include <sys/types.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

#define NVRTC_SAFE_CALL(x)                               \
  do {                                                   \
    nvrtcResult result = x;                              \
    if (result != NVRTC_SUCCESS) {                       \
      std::cerr << "\nerror: " #x " failed with error "  \
                << nvrtcGetErrorString(result) << '\n';  \
      exit(1);                                           \
    }                                                    \
  } while (0)

#define CUDA_SAFE_CALL(x)                                \
do {                                                     \
   CUresult result = x;                                  \
   if (result != CUDA_SUCCESS) {                         \
      const char* msg;                                   \
      cuGetErrorName(result, &msg);                      \
      std::cerr << "\nerror: " #x " failed with error "  \
                << msg << '\n';                          \
      exit(1);                                           \
   }                                                     \
} while(0)

namespace fs = std::filesystem;

namespace ncarray {

  using PreCompiledTypes = std::tuple<
    //float,
    double
    //std::uint16_t,
    //std::uint32_t,
    //std::int32_t,
    //std::int64_t,
    //bool
  >;

  template <typename T, typename TypeList>
  struct is_in_type_list;

  template <typename T, typename... Types>
  struct is_in_type_list<T, std::tuple<Types...>> {
    static constexpr bool value = (std::is_same_v<T, Types> || ... );
  };

  template <typename T, typename... Types>
  struct is_in_type_list<T, std::variant<Types...>> {
    static constexpr bool value = (std::is_same_v<T, Types> || ...);
  };

  template <typename T, typename... Types>
  static constexpr bool is_in_type_list_v = is_in_type_list<T, Types...>::value;

  template <typename T, typename... Types>
  constexpr bool is_any_of_v = (std::is_same_v<T, Types> || ...);

  class RuntimeCompiler {
  public:
    static RuntimeCompiler& instance();

    CUfunction get_expr_kernel(DType dest_t,
                               DType src_t,
                               DType work_t,
                               int n_views,
                               int n_scalars,
                               const std::vector<Instruction>& instrs,
                               bool expr_is_soarr = false);

    // --- Reduction Kernels --- //


    CUfunction get_fill_kernel(DType dest_t, bool dest_is_so = false);
    CUfunction get_copy_kernel(DType dest_t, DType src_t, bool src_is_so = false);
    CUfunction get_copy_view_into_view_kernel(DType dest_t,
                                              DType src_t,
                                              bool dest_is_so = false,
                                              bool src_is_so = false);

  private:
    RuntimeCompiler();

    template <typename DestT>
    std::string get_name_for_type() {
      std::string type_name;
      NVRTC_SAFE_CALL(nvrtcGetTypeName<DestT>(&type_name));
      return type_name;
    }

    CUfunction ptx_to_sass(std::string ptx, const char* func_name);

    std::string get_expression_kernel_str(DType dest_t,
                                          DType src_t,
                                          DType work_t,
                                          int n_views,
                                          int n_scalars,
                                          const std::vector<Instruction>& instrs,
                                          bool expr_is_soarr = false);

    std::string get_fill_kernel_str(DType dest_t, bool dest_is_so = false);
    std::string get_copy_kernel_str(DType dest_t, DType src_t, bool src_is_so = false);
    std::string get_copy_view_into_view_kernel_str(DType dest_t,
                                                   DType src_t,
                                                   bool dest_is_so = false,
                                                   bool src_is_so = false);

    std::string compile_kernel(std::string kernel_str, std::string k_id, const char* func_name);

    std::unordered_map<std::string, CUfunction> m_kernel_cache;
  };
} // namespace ncarray

#endif // NCARRAY_JIT_RTCOMPILER_HH
