/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_EXPRESSION_STATICMVNODE_HH
#define NCARRAY_EXPRESSION_STATICMVNODE_HH

#include "ncarray/custom_types.hh"
#ifdef __CUDACC__
#ifndef __CUDACC_RTC__
#include "ncarray/device/casts.cuh"
#endif
#endif
#include "ncarray/dtype.hh"
#include "ncarray/expression/interface.hh"
#include "ncarray/expression/mvnode.hh"
#ifndef __CUDACC_RTC__
#include "ncarray/host/casts.hh"
#endif
#include "ncarray/layout.hh"
#include "ncarray/op_code.hh"
#include "ncarray/op_traits.hh"

#ifdef __CUDACC_RTC__
typedef long long ssize_t;

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/complex>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

namespace hd_std = cuda::std;

#else

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace hd_std = std;

#endif

#ifndef __CUDACC_RTC__
#include <vector>
#endif

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
  template <
    int NViews,
    int NScalars,
    int NInstrs,
    typename ArrT = float,
    typename ScalarT = float,
    class MemTag = DevTag,
    class Layout = NCOffsetsPolicy
  >
  struct StaticExprMVNode : public ExpressionTag, public ExprOpInterface {
    static constexpr int NOperands = NViews + NScalars;
    static constexpr int NOps = NInstrs - NOperands; // Ignores unary for now

    Layout layouts[NViews];
    const void* data[NViews];
    ScalarT scalars[NScalars > 1 ? NScalars : 1];
    OpCode ops[NOps > 1 ? NOps : 1];
    short op_map[NOperands > 1 ? NOperands : 1];

    StaticExprMVNode() = default;

#ifndef __CUDACC_RTC__
    StaticExprMVNode(const ExprMVNode<MemTag>& node) {
      final_shape.set(node.shape(), node.ndim());
      final_size = node.size();

      for (int i = 0; i < NViews; ++i) {
        // IDX generators are counted as virtual views.
        // Need to guard to avoid reading past the actual arrays that are present
        if (i < static_cast<int>(node.layouts.size())) {
          layouts[i] = node.layouts[i];
          data[i] = node.data[i];
        } else {
          data[i] = nullptr;
        }
      }

      auto cast_op = [&](auto&& val) {
        using SrcT = hd_std::decay_t<decltype(val)>;
        return op_traits<SrcT>::template cast<ScalarT>(val);
      };

      int scalar_i { 0 };
      for (const auto& scalar : node.scalars) {
        scalars[scalar_i++] = std::visit(cast_op, scalar);
      }

      int operand_ptr { 0 };
      int op_ptr { 0 };
      // Start virtual arrays (for IDX/IOTA) after real ones
      int idx_counter { static_cast<int>(node.layouts.size()) };
      for (const auto& instr : node.instrs) {
        int idx = get_index(instr);
        OpCode op = get_op(instr);

        if (op == OpCode::LOAD_NCARR || op == OpCode::LOAD_SOARR) {
          op_map[operand_ptr++] = idx;
        } else if (op == OpCode::IDX) {
          // The IDX/IOTA (APL style index generator) is treated as a virtual load
          op_map[operand_ptr++] = idx_counter++;
        } else if (op == OpCode::LOAD_CONST) {
          op_map[operand_ptr++] = NViews + idx;
        } else {
          ops[op_ptr++] = op;
        }
      }
    }
#endif

    template <typename DestT, typename Coords>
    NCA_HD inline DestT eval(const Coords& coords) const {
      ScalarT values[NOperands];

      #pragma unroll NViews
      for (int i = 0; i < NViews; ++i) {
        if (data[i] == nullptr) {
          // Null data is considered a virtual load, e.g. for IDX/IOTA
          values[i] = this->md_to_lin<ScalarT>(coords);
        } else {
          auto& layout = layouts[i];
          const void* leaf_ptr = layout.advance(data[i], coords);

          const ArrT item = *reinterpret_cast<const ArrT*>(leaf_ptr);
          values[i] = op_traits<ArrT>::template cast<ScalarT>(item);
        }
      }

      #pragma unroll NScalars
      for (int i = 0; i < NScalars; ++i) {
        values[NViews + i] = scalars[i];
      }

      ScalarT res { values[op_map[0]] };
      int operand_cur { 1 }; // Cursor starts at the second, since we pulled the first

      #pragma unroll NOps
      for (int i = 0; i < NOps; ++i) {
        // Check for Binary/Unary. Binary ops begin with OpCode::ADD
        OpCode op = ops[i];
        if (static_cast<int>(op) >= static_cast<int>(OpCode::ADD)) {
          ScalarT next_v { values[op_map[operand_cur++]] };
          res = this->apply_op(res, next_v, op);
        } else {
          // Unary ops only work on the res value
          res = this->apply_op(res, ScalarT(0), op);
        }
      }
      return op_traits<ArrT>::template cast<DestT>(res);
    }

    NCA_HD inline DType dtype() const { return dtype_traits<ArrT>::value; }
  };

#ifndef __CUDACC_RTC__

  template <class MemTag>
  bool can_linearize(const ExprMVNode<MemTag>& node) {
    if (node.layouts.size() > 16 || node.instrs.size() > 32) {
      // Bow out early if there are many instructions/arrays to pass by value on stack
      return false;
    }
    int stack_depth = 0;
    bool seen_arr { false };
    bool expr_soarr { false };
    DType ref_dtype;
    for (const auto& instr : node.instrs) {
      OpCode op = get_op(instr);
      if (op == OpCode::LOAD_NCARR || op == OpCode::LOAD_SOARR) {
        stack_depth++;

        if (!seen_arr) {
          seen_arr = true;
          ref_dtype = node.dtypes[get_index(instr)];
          if (op == OpCode::LOAD_SOARR) {
            expr_soarr = true;
          }
        } else if (op == OpCode::LOAD_SOARR && !expr_soarr) {
          // Have a different layout
          return false;
        }
        int idx = get_index(instr);
        if (node.dtypes[idx] != ref_dtype) {
          // Arrays of different dtypes
          return false;
        }
      } else if (op == OpCode::LOAD_CONST || op == OpCode::IDX) {
        // IDX, the index generator, is like a virtual load.
        stack_depth++;
      } else if (static_cast<int>(op) >= static_cast<int>(OpCode::ADD)) {
        // Binary operations begin at ADD. The stack must be 2 deep
        if (stack_depth < 2) {
          return false;
        }
        stack_depth--;
      } else {
        // Unary ops require a stack depth of 1
        if (stack_depth < 1) {
          return false;
        }
      }

      if (stack_depth > 2) {
        return false;
      }
    }
    return stack_depth == 1;
  }

#endif
} // namespace ncarray

#endif // NCARRAY_EXPRESSION_STATICMVNODE_HH
