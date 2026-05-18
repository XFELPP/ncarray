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
#include "ncarray/indexing.hh"
#include "ncarray/jit/path_utils.hh"
#include "ncarray/jit/jit_utils.hh"
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
  class RuntimeCompiler {
  public:
    static RuntimeCompiler& instance();

    CUfunction get_expr_kernel(DType dest_t,
                               DType src_t,
                               DType work_t,
                               int n_views,
                               int n_scalars,
                               ssize_t ndim,
                               const ssize_t* final_shape,
                               const std::vector<Instruction>& instrs,
                               bool expr_is_soarr = false);

    template <int NDIM>
    CUfunction get_stencil_expr_kernel(DType dest_t,
                                       DType src_t,
                                       DType work_t,
                                       const std::vector<StaticCoords<NDIM>>& offsets,
                                       const std::vector<Instruction>& instrs,
                                       const std::vector<Scalar>& scalars,
                                       const std::vector<std::uint8_t>& is_pointer_axis,
                                       bool expr_is_soarr = false) {
      std::string arch_opt = get_arch_opt();
      std::string kernel_str = get_stencil_expr_kernel_str(dest_t,
                                                           src_t,
                                                           work_t,
                                                           offsets,
                                                           instrs,
                                                           scalars,
                                                           is_pointer_axis,
                                                           expr_is_soarr);
      std::string k_id = hash_to_hex(kernel_str + arch_opt);

      if (m_kernel_cache.count(k_id)) {
        return m_kernel_cache[k_id];
      }

      fs::path cache_path = get_cache_dir() / (k_id + ".ptx");
      std::string ptx;

      if (fs::exists(cache_path)) {
        ptx = read_file(cache_path);

        CUfunction k_func = ptx_to_sass(ptx, "jit_stencil_expr_kernel");
        m_kernel_cache[k_id] = k_func;
      } else {
        ptx = compile_kernel(kernel_str, k_id, "jit_stencil_expr_kernel");

        write_file(cache_path, ptx);
      }

      return m_kernel_cache[k_id];
    }

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
                                          ssize_t ndim,
                                          const ssize_t* final_shape,
                                          const std::vector<Instruction>& instrs,
                                          bool expr_is_soarr = false);

    template <int NDIM>
    std::string get_stencil_expr_kernel_str(DType dest_t,
                                            DType src_t,
                                            DType work_t,
                                            const std::vector<StaticCoords<NDIM>>& offsets,
                                            const std::vector<Instruction>& instrs,
                                            const std::vector<Scalar>& scalars,
                                            const std::vector<std::uint8_t>& is_pointer_axis,
                                            bool expr_is_soarr = false) {

      auto get_dtype_name = [&] <typename T> () { return get_name_for_type<T>(); };
      std::string dest_t_name = dispatch(dest_t, get_dtype_name);
      std::string src_t_name = dispatch(src_t, get_dtype_name);
      std::string work_t_name = dispatch(work_t, get_dtype_name);

      std::string view_t_name = get_name_for_type<ArrayImpl<NCOffsetsPolicy, DevViewPolicy>>();
      if (expr_is_soarr) {
        view_t_name = get_name_for_type<ArrayImpl<SOArrayPolicy, DevViewPolicy>>();
      }

      std::vector<std::string> stack;
      int v_ptr { 0 };
      int n_scalars { 0 };
      for (const auto& instr : instrs) {
        OpCode op = get_op(instr);
        int idx = get_index(instr);
        if (op == OpCode::LOAD_NCARR) {
          stack.push_back("v" + std::to_string(v_ptr++));
        } else if (op == OpCode::LOAD_CONST) {
          stack.push_back("s" + std::to_string(idx));
          n_scalars = std::max(n_scalars, idx + 1);
        } else {
          std::string r { stack.back() };
          stack.pop_back();
          std::string l { stack.back() };
          stack.pop_back();
          if (op == OpCode::ADD) {
            stack.push_back("(" + l + " + " + r + ")");
          } else if (op == OpCode::MUL) {
            stack.push_back("(" + l + " * " + r + ")");
          } else if (op == OpCode::SUB) {
            stack.push_back("(" + l + " - " + r + ")");
          } else if (op == OpCode::DIV) {
            stack.push_back("(" + l + " / " + r + ")");
          } else if (op == OpCode::GT) {
            stack.push_back("(" + l + " > " + r + ")");
          }
        }
      }
      std::string math_result = stack.back();

      std::string s_params;
      for (int i = 0; i < n_scalars; ++i) {
        s_params += ", " + work_t_name + " s" + std::to_string(i);
      }

      // Setup prefix groups based on known pointer axes elimnating as many calls to
      // `advance` as possible. These will then be replaced with constant byte offsets.

      int max_ptr_axis { -1 };
      for (int dim = 0; dim < NDIM; ++dim) {
        if (is_pointer_axis[dim]) {
          max_ptr_axis = dim;
        }
      }

      std::vector<std::vector<std::size_t>> prefix_groups;
      std::vector<std::vector<ssize_t>> unique_prefixes;

      for (std::size_t i = 0; i < offsets.size(); ++i) {
        std::vector<ssize_t> prefix;

        for (int dim = 0; dim <= max_ptr_axis; ++dim) {
          prefix.push_back(offsets[i][dim]);
        }

        bool found { false };
        for (std::size_t grp = 0; grp < unique_prefixes.size(); ++grp) {
          if (unique_prefixes[grp] == prefix) {
            prefix_groups[grp].push_back(i);
            found = true;
            break;
          }
        }
        if (!found) {
          unique_prefixes.push_back(prefix);
          prefix_groups.push_back({i});
        }
      }

      std::string load_logic;
      // First, resolve the pointer indirections for each unique prefix group
      for (std::size_t grp = 0; grp < unique_prefixes.size(); ++grp) {
        std::string p_name { "ptr_prefix_" + std::to_string(grp) };
        load_logic += "  const uint8_t* " + p_name + " = reinterpret_cast<const uint8_t*>(src_data);\n";
        for (int dim = 0; dim <= max_ptr_axis; ++dim) {
          std::string off_val { std::to_string(unique_prefixes[grp][dim]) };

          if (expr_is_soarr) {
            load_logic +=
              "  " + p_name + " += (c_src[" + std::to_string(dim) + "] + " + off_val +
              ") * src_l.stride(" + std::to_string(dim) + ");\n";
            if (is_pointer_axis[dim]) {
              load_logic +=
                "  " + p_name + " = *reinterpret_cast<const uint8_t* const*>(" + p_name
                + ") + src_l.suboffset(" + std::to_string(dim) + ");\n";
            }
          } else {
            if (is_pointer_axis[dim]) {
              load_logic +=
                "  " + p_name + " = reinterpret_cast<const uint8_t*>(reinterpret_cast<const void* const*>(" +
                p_name + ")[c_src[" + std::to_string(dim) + "] + " + off_val +
                " + src_l.offset(" + std::to_string(dim) + ")]);\n";
            } else {
              load_logic +=
                "  " + p_name + " += (c_src[" + std::to_string(dim) + "] + " + off_val + ") * src_l.stride(" +
                std::to_string(dim) + ") + src_l.offset(" + std::to_string(dim) + ");\n";
            }
          }
        }
      }

      // Now, the loads can just be done with straight byte offsets.
      for (std::size_t grp = 0; grp < prefix_groups.size(); ++grp) {
        std::string p_name { "ptr_prefix_" + std::to_string(grp) };

        for (std::size_t idx : prefix_groups[grp]) {
          std::string vid { std::to_string(idx) };

          load_logic += "  const uint8_t* byte_ptr_" + vid + " = " + p_name + ";\n";

          for (int dim = max_ptr_axis + 1; dim < NDIM; ++dim) {
            std::string off_val { std::to_string(offsets[idx][dim]) };

            if (expr_is_soarr) {
              load_logic +=
                "  byte_ptr_" + vid + " += (c_src[" + std::to_string(dim) + "] + " + off_val +
                ") * src_l.stride(" + std::to_string(dim) + ");\n";
            } else {
              load_logic +=
                "  byte_ptr_" + vid + " += (c_src[" + std::to_string(dim) + "] + " + off_val +
                ") * src_l.stride(" + std::to_string(dim) + ") + src_l.offset(" + std::to_string(dim) + ");\n";
            }
          }

          load_logic +=
            "  " + work_t_name + " v" + vid + " = static_cast<" + work_t_name + ">" +
            "(*reinterpret_cast<const " + src_t_name + "*>(byte_ptr_" + vid + "));\n";
        }
      }

      std::string source = R"a(
#include "ncarray/device/kernels.cuh"

extern "C" __global__ void jit_stencil_expr_kernel(const void* src_data,
                                                   )a" + view_t_name + R"a( src_l,
                                                   void* dest_data,
                                                   )a" + view_t_name + " dest_l" +
                                                   s_params + R"a( ) {
  unsigned b_idx { blockIdx.x * blockDim.x + threadIdx.x };
  if (b_idx >= dest_l.size()) {
    return;
  }

  ncarray::StaticCoords<NDIM_VAL, unsigned> c_out;
  unsigned tmp_idx { b_idx };
  #pragma unroll
  for (int dim = NDIM_VAL - 1; dim >= 0; --dim) {
    c_out[dim] = tmp_idx % dest_l.shape(dim);
    tmp_idx /= dest_l.shape(dim);
  }

  ncarray::StaticCoords<NDIM_VAL, unsigned> c_src;
  #pragma unroll
  for (int dim = 0; dim < NDIM_VAL; ++dim) {
    unsigned step = src_l.shape(dim) / dest_l.shape(dim);
    c_src[dim] = c_out[dim] * step;
  }

)a" + load_logic + R"a(
  *reinterpret_cast<)a" + dest_t_name + R"a(*>(dest_l.advance(dest_data, c_out)) =
    static_cast<)a" + dest_t_name + R"a(>( )a" + math_result + R"a( );
}
)a";

      std::size_t pos { 0 };
      while ((pos = source.find("NDIM_VAL", pos)) != std::string::npos) {
        source.replace(pos, 8, std::to_string(NDIM));
        pos += std::to_string(NDIM).length();
      }

      return source;
    }

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
