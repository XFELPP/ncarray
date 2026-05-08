/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/jit/rtcompiler.hh"

#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"
#include "ncarray/jit/path_utils.hh"
#include "ncarray/op_code.hh"
#include "rtcompiler.hh"

#include <cuda.h>
#include <nvrtc.h>

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

    namespace fs = std::filesystem;

namespace {
  std::string hash_to_hex(const std::string& input) {
    std::hash<std::string> hasher;
    size_t hash = hasher(input);
    std::stringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << hash;
    return ss.str();
  }

  std::string get_arch_opt() {
    int major_v;
    int minor_v;
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    cuDeviceGetAttribute(&major_v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor_v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

    std::string arch_opt {
      "-arch=compute_" + std::to_string(major_v) + std::to_string(minor_v)
    };

    return arch_opt;
  }
}

namespace ncarray {

  RuntimeCompiler::RuntimeCompiler() {
    cuInit(0);
  }

  RuntimeCompiler& RuntimeCompiler::instance() {
    static RuntimeCompiler inst;
    return inst;
  }

  CUfunction RuntimeCompiler::get_expr_kernel(DType dest_t,
                                              DType src_t,
                                              DType work_t,
                                              int n_views,
                                              int n_scalars,
                                              const std::vector<Instruction>& instrs,
                                              bool expr_is_soarr) {
    std::string arch_opt = get_arch_opt();
    std::string kernel_str = get_expression_kernel_str(dest_t,
                                                       src_t,
                                                       work_t,
                                                       n_views,
                                                       n_scalars,
                                                       instrs,
                                                       expr_is_soarr);
    std::string k_id = hash_to_hex(kernel_str + arch_opt);

    if (m_kernel_cache.count(k_id)) {
      return m_kernel_cache[k_id];
    }

    fs::path cache_path = get_cache_dir() / (k_id + ".ptx");
    std::string ptx;

    if (fs::exists(cache_path)) {
      ptx = read_file(cache_path);

      CUfunction k_func = ptx_to_sass(ptx, "jit_expression_kernel");
      m_kernel_cache[k_id] = k_func;
    } else {
      ptx = compile_kernel(kernel_str, k_id, "jit_expression_kernel");

      write_file(cache_path, ptx);
    }

    return m_kernel_cache[k_id];
  }

  CUfunction RuntimeCompiler::get_fill_kernel(DType dest_t, bool dest_is_so) {
    std::string arch_opt = get_arch_opt();
    std::string kernel_str = get_fill_kernel_str(dest_t, dest_is_so);
    std::string k_id = hash_to_hex(kernel_str + arch_opt);

    if (m_kernel_cache.count(k_id)) {
      return m_kernel_cache[k_id];
    }

    fs::path cache_path = get_cache_dir() / (k_id + ".ptx");
    std::string ptx;

    if (fs::exists(cache_path)) {
      ptx = read_file(cache_path);

      CUfunction k_func = ptx_to_sass(ptx, "jit_fill_kernel");
      m_kernel_cache[k_id] = k_func;
    } else {
      ptx = compile_kernel(kernel_str, k_id, "jit_fill_kernel");

      write_file(cache_path, ptx);
    }

    return m_kernel_cache[k_id];
  }

  CUfunction RuntimeCompiler::get_copy_kernel(DType dest_t, DType src_t, bool src_is_so) {
    std::string arch_opt = get_arch_opt();
    std::string kernel_str = get_copy_kernel_str(dest_t, src_t, src_is_so);
    std::string k_id = hash_to_hex(kernel_str + arch_opt);

    if (m_kernel_cache.count(k_id)) {
      return m_kernel_cache[k_id];
    }

    fs::path cache_path = get_cache_dir() / (k_id + ".ptx");
    std::string ptx;

    if (fs::exists(cache_path)) {
      ptx = read_file(cache_path);

      CUfunction k_func = ptx_to_sass(ptx, "jit_copy_kernel");
      m_kernel_cache[k_id] = k_func;
    } else {
      ptx = compile_kernel(kernel_str, k_id, "jit_copy_kernel");

      write_file(cache_path, ptx);
    }

    return m_kernel_cache[k_id];
  }

  CUfunction RuntimeCompiler::get_copy_view_into_view_kernel(DType dest_t,
                                                             DType src_t,
                                                             bool dest_is_so,
                                                             bool src_is_so) {
    std::string arch_opt = get_arch_opt();
    std::string kernel_str = get_copy_view_into_view_kernel_str(dest_t,
                                                                src_t,
                                                                dest_is_so,
                                                                src_is_so);
    std::string k_id = hash_to_hex(kernel_str + arch_opt);

    if (m_kernel_cache.count(k_id)) {
      return m_kernel_cache[k_id];
    }

    fs::path cache_path = get_cache_dir() / (k_id + ".ptx");
    std::string ptx;

    if (fs::exists(cache_path)) {
      ptx = read_file(cache_path);

      CUfunction k_func = ptx_to_sass(ptx, "jit_copy_view_into_view_kernel");
      m_kernel_cache[k_id] = k_func;
    } else {
      ptx = compile_kernel(kernel_str, k_id, "jit_copy_view_into_view_kernel");

      write_file(cache_path, ptx);
    }

    return m_kernel_cache[k_id];
  }

  CUfunction RuntimeCompiler::ptx_to_sass(std::string ptx, const char* func_name) {
    CUmodule cu_mod;
    CUjit_option ptx_jit_opts[] = { CU_JIT_OPTIMIZATION_LEVEL };
    void* ptx_jit_vals[] = { reinterpret_cast<void*>(3) };
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&cu_mod, ptx.data(), 1, ptx_jit_opts, ptx_jit_vals));
    CUfunction k_func;
    CUDA_SAFE_CALL(cuModuleGetFunction(&k_func, cu_mod, func_name));

    return k_func;
  }

  std::string RuntimeCompiler::get_expression_kernel_str(DType dest_t,
                                                         DType src_t,
                                                         DType work_t, // Intermediate evals and Scalars
                                                         int n_views,
                                                         int n_scalars,
                                                         const std::vector<Instruction>& instrs,
                                                         bool expr_is_soarr) {
    auto get_dtype_name = [&] <typename T> () {
      return get_name_for_type<T>();
    };

    std::string dest_t_name = dispatch(dest_t, get_dtype_name);
    std::string src_t_name = dispatch(src_t, get_dtype_name);
    std::string work_t_name = dispatch(work_t, get_dtype_name);

    std::string result_type;
    std::string layout_t;
    if (expr_is_soarr) {
      result_type = get_name_for_type<ArrayImpl<SOArrayPolicy, DevViewPolicy>>();
      layout_t = "ncarray::SOArrayPolicy";
    } else {
      result_type = get_name_for_type<ArrayImpl<NCOffsetsPolicy, DevViewPolicy>>();
      layout_t = "ncarray::NCOffsetsPolicy";
    }

    std::string d_params;
    std::string packing_logic;

    for (int i = 0; i < n_views; ++i) {
      auto s_idx = std::to_string(i);
      if (i > 0) {
        d_params += ", ";
      }
      d_params += "const void* d" + s_idx + ", " + layout_t + " l" + s_idx;
      packing_logic += "  mvnode.data["    + s_idx + "] = d" + s_idx + ";\n";
      packing_logic += "  mvnode.layouts[" + s_idx + "] = l" + s_idx + ";\n";
    }

    for (int i = 0; i < n_scalars; ++i) {
      d_params += ", " + work_t_name + " s" + std::to_string(i);
      packing_logic += "  mvnode.scalars[" + std::to_string(i) + "] = s" + std::to_string(i) + ";\n";
    }

    int operand_ptr { 0 };
    int op_ptr { 0 };
    int n_indices { 0 };
    for (const auto& instr : instrs) {
      int idx = get_index(instr);
      OpCode op = get_op(instr);

      if (op == OpCode::LOAD_NCARR || op == OpCode::LOAD_SOARR) {
        packing_logic += "  mvnode.op_map[" + std::to_string(operand_ptr++) + "] = " + std::to_string(idx) + ";\n";
      } else if (op == OpCode::IDX) {
        // Negative values are sentinals for IDX "virtual loads"
        packing_logic += "  mvnode.op_map[" + std::to_string(operand_ptr++) + "] = -1;\n";
        n_indices++; // We encode the indices now into hte template var as a "View"
      } else if (op == OpCode::LOAD_CONST) {
        packing_logic += "  mvnode.op_map[" + std::to_string(operand_ptr++) + "] = " + std::to_string(n_views + idx) + ";\n";
      } else {
        auto opi_str = std::to_string(static_cast<int>(op));
        packing_logic += "  mvnode.ops[" + std::to_string(op_ptr++) + "] = " + "static_cast<ncarray::OpCode>(" + opi_str + ");\n";
      }
    }

    std::string jit_expression_k_sig =
      "extern \"C\" __global__ void jit_expression_kernel(" + d_params + ", " + result_type + " result) {\n";


    // NViews actually encodes "true" arrays AND the virtual arrays from IDX operations
    // StaticExprMVNode<NViews, NScalars, NInstrs, ArrT, ScalarT, MemTag, Layout>
    std::string StaticExprMVNode_n = "ncarray::StaticExprMVNode<" +
      std::to_string(n_views + n_indices) + ", " + // Arrays AND IDX loads
      std::to_string(n_scalars)           + ", " +
      std::to_string(instrs.size())       + ", " +
      src_t_name                          + ", " + // Array datatype ArrT
      work_t_name                         + ", " + // We will pre-cast all scalars to WorkT
      "ncarray::DevTag"                   + ", " +
      layout_t + "> mvnode;\n";

    std::string wrapper =
      jit_expression_k_sig +
      "  " + StaticExprMVNode_n +
         packing_logic + "  \n" +
      "  ncarray::execute_expression_d<" + dest_t_name + ">(mvnode, result);\n" +
      "}\n";

    return wrapper;
  }

  std::string RuntimeCompiler::get_fill_kernel_str(DType dest_t, bool dest_is_so) {
    auto get_dtype_name = [&] <typename T> () {
      return get_name_for_type<T>();
    };

    std::string dest_t_name = dispatch(dest_t, get_dtype_name);

    std::string dest_view_type = get_name_for_type<ArrayImpl<NCOffsetsPolicy, DevViewPolicy>>();

    if (dest_is_so) {
      dest_view_type = get_name_for_type<ArrayImpl<SOArrayPolicy, DevViewPolicy>>();
    }

    std::string wrapper =
      "extern \"C\" __global__ void jit_fill_kernel(" + dest_view_type + " out, " + dest_t_name + " val) {\n"
        "  ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };\n"
        "  if (idx < out.size()) {\n"
        "    ncarray::fill_d(out, val);\n"
        "  }\n"
        "}\n";

    return wrapper;
  }

  std::string RuntimeCompiler::get_copy_kernel_str(DType dest_t,
                                                   DType src_t,
                                                   bool src_is_so) {
    auto get_dtype_name = [&] <typename T> () {
      return get_name_for_type<T>();
    };

    std::string dest_t_name = dispatch(dest_t, get_dtype_name);
    std::string src_t_name = dispatch(src_t, get_dtype_name);
    std::string src_view_type = get_name_for_type<ArrayImpl<NCOffsetsPolicy, DevViewPolicy>>();

    if (src_is_so) {
      src_view_type = get_name_for_type<ArrayImpl<SOArrayPolicy, DevViewPolicy>>();
    }

    return
      "extern \"C\" __global__ void jit_copy_kernel(" + dest_t_name + "* dest, const " + src_view_type + " src) {\n"
        "  ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };\n"
        "  if (idx < src.size()) {\n"
        "    ncarray::copy_into_d<" + dest_t_name + ", " + src_t_name + ">(dest, src);\n"
        "  }\n"
        "}\n";
  }

  std::string RuntimeCompiler::get_copy_view_into_view_kernel_str(DType dest_t,
                                                                  DType src_t,
                                                                  bool dest_is_so,
                                                                  bool src_is_so) {
    auto get_dtype_name = [&] <typename T> () {
      return get_name_for_type<T>();
    };

    std::string dest_t_name = dispatch(dest_t, get_dtype_name);
    std::string src_t_name = dispatch(src_t, get_dtype_name);
    std::string dest_view_type = get_name_for_type<ArrayImpl<NCOffsetsPolicy, DevViewPolicy>>();
    std::string src_view_type = get_name_for_type<ArrayImpl<NCOffsetsPolicy, DevViewPolicy>>();

    if (dest_is_so) {
      dest_view_type = get_name_for_type<ArrayImpl<SOArrayPolicy, DevViewPolicy>>();
    }

    if (src_is_so) {
      src_view_type = get_name_for_type<ArrayImpl<SOArrayPolicy, DevViewPolicy>>();
    }
    return
      "extern \"C\" __global__ void jit_copy_view_into_view_kernel(" + dest_view_type + " dest, const " + src_view_type + " src) {\n"
      "  ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };\n"
      "  if (idx < src.size()) {\n"
      "    ncarray::copy_view_into_view_d<" + dest_t_name + ", " + src_t_name + ">(dest, src);\n"
      "  }\n"
      "}\n";
  }

  std::string RuntimeCompiler::compile_kernel(std::string kernel_str,
                                              std::string k_id,
                                              const char* func_name) {
    int major_v;
    int minor_v;
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    cuDeviceGetAttribute(&major_v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor_v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

    std::string arch_opt {
      "-arch=compute_" + std::to_string(major_v) + std::to_string(minor_v)
    };

    std::string nca_inc_dir = get_install_include_path();
    std::string jit_inc_dir;
    if (const char* env_inc = std::getenv("NCARRAY_JIT_INCLUDE_DIR")) {
      jit_inc_dir = std::string(env_inc);
    } else {
      jit_inc_dir = (fs::path(nca_inc_dir) / "jit").string();
    }

    std::string nca_opt { "-I" + nca_inc_dir };
    std::string jit_opt { "-I" + jit_inc_dir };

    std::vector<const char*> opts = {
      "--std=c++20",
      arch_opt.c_str(),
      nca_opt.c_str(), // Normal ncarray headers
      jit_opt.c_str()  // Bundled CCCl headers for JIT
    };

    std::string source =
      "#include \"ncarray/array_impl.hh\"\n"
      "#include \"ncarray/device/kernels.cuh\"\n"
      "#include \"ncarray/mvnode.hh\"\n\n" +
      kernel_str;
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, source.c_str(), "kernels.cu", 0, NULL, NULL));

    nvrtcResult res = nvrtcCompileProgram(prog, opts.size(), opts.data());

    if (res != NVRTC_SUCCESS) {
      std::size_t log_size { 0 };
      nvrtcGetProgramLogSize(prog, &log_size);
      std::vector<char> log(log_size);
      nvrtcGetProgramLog(prog, log.data());
      throw std::runtime_error("JIT Compilation Failed!\n" + std::string(log.data()));
    }

    // Get PTX
    std::size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::vector<char> ptx(ptx_size);
    nvrtcGetPTX(prog, ptx.data());

    // Load PTX into a CUDA module
    CUfunction k_func = ptx_to_sass(std::string(ptx.data()), func_name);

    m_kernel_cache[k_id] = k_func;

    nvrtcDestroyProgram(&prog);
    return std::string(ptx.data(), ptx.size());
  }
} // namespace ncarray
