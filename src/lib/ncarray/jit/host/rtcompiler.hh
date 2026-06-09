/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_JIT_HOST_RTCOMPILER_HH
#define NCARRAY_JIT_HOST_RTCOMPILER_HH

#include "ncarray/dtype.hh"
#include "ncarray/layout.hh"
#include "ncarray/op_code.hh"
#include "ncarray/op_traits.hh"

#include <asmjit/core.h>
#include <asmjit/x86.h>

#ifdef _WIN32
typedef SSIZE_T ssize_t;

// On Windows need to export symbols for DLLs
#ifdef NCA_BUILD_JIT_API
#define NCA_JIT_API __declspec(dllexport)
#else
#define NCA_JIT_API __declspec(dllimport)
#endif
#else
#include <dlfcn.h>
#include <sys/types.h>

#define NCA_JIT_API
#endif

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

namespace ncarray {
  namespace host {
    /**
     * The signature for evaluation of lineriazable arrays of single datatypes/layouts.
     */
    using ExprKernelFunc = void (*)(const void** src_ptrs, void* dest_data);

    class NCA_JIT_API RuntimeCompiler {
    public:
      static RuntimeCompiler& instance();

      /**
       * Convert the ncarray type system to corresponding asmjit type system.
       *
       * This function only handles scalar datatypes. E.g. complex and vector types
       * must be constructed using combinations of the returned type for their unit
       * types (a complex<float> is two TypeId::kFloat32 for example).
       *
       * A `char_` is converted to signed integer kInt8. A `bool_` is converted to
       * unsigned integer kUInt8, as these do not have corresponding types in asmjit.
       *
       * @param[in] dtype The ncarray datatype to convert.
       * @returns typeid The asmjit TypeId that corresponds to the ncarray DType.
       */
      asmjit::TypeId dtype_to_typeid(DType dtype);

      /**
       * Get the compiled funcion/kernel for evaluation of the lineriazable expression
       * created by the layouts and instructions passed in.
       *
       * If the function is in memory, it will be returned from the in-memory cache,
       * otherwise, if it has previously been compiled, it will be loaded from disk,
       * or, compiled and written to disk, before being returned.
       *
       * @param[in] dest_t The final output datatype.
       * @param[in] src_t The input datatype of the arrays.
       * @param[in] work_t The datatype to use for intermediate calculations. This may
       *            or may not be equivalent to dest_t. E.g. a long expression may use
       *            float/int64 while performing intermediate operations, while the
       *            final result of the expression is a comparison returning bool.
       * @param[in] dest_layout The destination/result Layout to write into.
       * @param[in] instrs The instructions (packed index + OpCode) for the expression.
       * @param[in] layouts The layouts of the arrays involved.
       * @param[in] scalars Any scalar constants involved.
       * @param[in] expr_is_soarr Whether the arrays are SOArray* or NCArray*.
       * @returns k_func The function to evaluate the expression.
       */
      ExprKernelFunc get_expr_kernel(DType dest_t,
                                     DType src_t,
                                     DType work_t,
                                     const SOArrayPolicy& dest_layout,
                                     const std::vector<Instruction>& instrs,
                                     const std::vector<SOArrayPolicy>& layouts,
                                     const std::vector<Scalar>& scalars,
                                     bool expr_is_soarr = false);

    private:
      RuntimeCompiler();

      /**
       * Compile the function to evaluate an expression, for the x86 architecture.
       *
       * After compiling, the function will be written to disk.
       *
       * @param[in] cache_path The path to write the compiled function to, for on-disk
       *            storage.
       * @param[in] dest_t The final output datatype.
       * @param[in] src_t The input datatype of the arrays.
       * @param[in] work_t The datatype to use for intermediate calculations. This may
       *            or may not be equivalent to dest_t. E.g. a long expression may use
       *            float/int64 while performing intermediate operations, while the
       *            final result of the expression is a comparison returning bool.
       * @param[in] dest_layout The destination/result Layout to write into.
       * @param[in] instrs The instructions (packed index + OpCode) for the expression.
       * @param[in] layouts The layouts of the arrays involved.
       * @param[in] scalars Any scalar constants involved.
       * @param[in] expr_is_soarr Whether the arrays are SOArray* or NCArray*.
       */
      void compile_x86_expr_kernel(fs::path cache_path,
                                   DType dest_t,
                                   DType src_t,
                                   DType work_t,
                                   const SOArrayPolicy& dest_layout,
                                   const std::vector<Instruction>& instrs,
                                   const std::vector<SOArrayPolicy>& layouts,
                                   const std::vector<Scalar>& scalars,
                                   bool expr_is_soarr = false);

      /**
       * In memory cache of compiled functions for expression evaluation.
       */
      std::unordered_map<std::string, ExprKernelFunc> m_kernel_cache;
    };
  } // namespace host
} // namespace ncarray

#endif // NCARRAY_JIT_HOST_RTCOMPILER_HH
