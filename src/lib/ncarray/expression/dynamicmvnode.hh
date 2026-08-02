/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_EXPRESSION_DYNAMICMVNODE_HH
#define NCARRAY_EXPRESSION_DYNAMICMVNODE_HH

#include "ncarray/custom_types.hh"
#ifdef __CUDACC__
#include "ncarray/device/casts.cuh"
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
  template <class MemTag = DevTag, int MaxStackSize = 24>
  struct DynamicExprMVNode : public ExpressionTag, public ExprOpInterface {
    const Instruction* instrs { nullptr };         ///< Instruction stack

    const SOArrayPolicy* layouts { nullptr };      ///< Array layout operands
    const void* const * data { nullptr };          ///< Array data
    const DType* arr_dtypes { nullptr };           ///< Datatypes for arrays

    // Scalars
    const DType* constants_dtypes { nullptr };     ///< Scalar data types
    const hd_std::uint16_t* constants_offsets { nullptr }; ///< Scalar offset in byte stream
    // Largest supported scalar type is 32 bytes.
    const hd_std::uint8_t* constants_buf { nullptr };      ///< Byte stream of scalars

    DType expr_dtype;     ///< The DType for the final evaluated expression
    DType work_dtype;     ///< The DType for intermediate sub-expression evaluations

    hd_std::uint8_t n_layouts { 0 };
    hd_std::uint8_t n_scalars { 0 };
    hd_std::uint8_t n_instrs { 0 };

    size_t arr_alignment { 16 };

#ifndef __CUDACC_RTC__
    DynamicExprMVNode(const ExprMVNode<MemTag>& node,
                      const hd_std::uint8_t* buf,
                      hd_std::uint8_t n_instrs_,
                      hd_std::uint8_t n_arrays_,
                      hd_std::uint8_t n_scalars_,
                      size_t alignment = 16)
      : n_layouts(n_arrays_)
      , n_scalars(n_scalars_)
      , n_instrs(n_instrs_)
      , arr_alignment(alignment)
    {
      expr_dtype = node.expr_dtype;
      work_dtype = node.work_dtype;
      final_size = node.size();
      final_shape.set(node.shape(), node.ndim());

      auto align = [&](size_t off) { return (off + (alignment - 1)) & ~(alignment - 1); };

      hd_std::uint16_t offset { 0 };
      instrs = reinterpret_cast<const Instruction*>(buf);
      offset += n_instrs * sizeof(Instruction);
      offset = align(offset);

      size_t aligned_layout_size { align(sizeof(SOArrayPolicy)) };
      layouts = reinterpret_cast<const SOArrayPolicy*>(buf + offset);
      offset += n_layouts * aligned_layout_size;

      data = reinterpret_cast<const void* const *>(buf + offset);
      offset += n_layouts * sizeof(void**);
      offset = align(offset);

      arr_dtypes = reinterpret_cast<const DType*>(buf + offset);
      offset += n_layouts * sizeof(DType);
      offset = align(offset);

      constants_dtypes = reinterpret_cast<const DType*>(buf + offset);
      offset += n_scalars * sizeof(DType);
      offset = align(offset);

      constants_offsets = reinterpret_cast<const hd_std::uint16_t*>(buf + offset);
      offset += n_scalars * 2;
      offset = align(offset);

      constants_buf = buf + offset;
    }
#endif

    NCA_HD inline DType dtype() const { return expr_dtype; }

    NCA_HD inline ssize_t itemsize() const { return ncarray::itemsize(this->dtype()); }

    template <typename DestT, typename Coords>
    NCA_HD inline DestT eval(const Coords& coords) const {
      auto eval_op = [&] <typename WorkT> () {
        WorkT stack[MaxStackSize] { 0 };
        int top { -1 };

        for (int i = 0; i < n_instrs; ++i) {
          Instruction instr = instrs[i];
          OpCode op = get_op(instr);

          switch (op) {
          // --- Loads and generators --- //
          case OpCode::IDX: {
            stack[++top] = this->md_to_lin<WorkT>(coords);
            break;
          }
          case OpCode::LOAD_NCARR:
          case OpCode::LOAD_SOARR: {
            int arr_idx = get_index(instr);
            const hd_std::uint8_t* layout_ptr = reinterpret_cast<const hd_std::uint8_t*>(layouts);
            size_t aligned_offset =
              arr_idx * ((sizeof(SOArrayPolicy) + (arr_alignment - 1)) & ~(arr_alignment - 1));
            const auto& view = *reinterpret_cast<const SOArrayPolicy*>(layout_ptr + aligned_offset);

            DType src_dtype { arr_dtypes[arr_idx] };
            const void* ptr { nullptr };
            if (op == OpCode::LOAD_NCARR) {
              ptr = reinterpret_cast<const NCOffsetsPolicy&>(view).advance(data[arr_idx], coords);
            } else {
              ptr = view.advance(data[arr_idx], coords);
            }

            int src_idx { static_cast<int>(src_dtype) };

            if constexpr (hd_std::is_same_v<MemTag, HostTag>) {
  #ifndef __CUDA_ARCH__
              stack[++top] = host::vm_cast_table<WorkT>[src_idx](ptr);
  #endif
            } else {
  #ifdef __CUDACC__
              stack[++top] = device::device_cast<WorkT>(src_idx, ptr);
  #endif
            }
            break;
          }
          case OpCode::LOAD_CONST: {
            int c_idx = get_index(instr);
            const void* ptr = constants_buf + constants_offsets[c_idx];
            int src_idx { static_cast<int>(constants_dtypes[c_idx]) };

            if constexpr (hd_std::is_same_v<MemTag, HostTag>) {
  #ifndef __CUDA_ARCH__
              stack[++top] = host::vm_cast_table<WorkT>[src_idx](ptr);
  #endif
            } else {
  #ifdef __CUDACC__
              stack[++top] = device::device_cast<WorkT>(src_idx, ptr);
  #endif
            }
            break;
          }
          // --- Unary ops --- //
          case OpCode::NEG:
          case OpCode::INC:
          case OpCode::DEC: {
            // The unary ops transform the top in place. Second operand ignored.
            stack[top] = this->apply_op(stack[top], WorkT(0), op);
            break;
          }
          // Skipping SZOF, ADDR, INDR, CAST, LNOT, BNOT
          // --- Binary ops --- //
          // Arithmetic
          case OpCode::ADD:
          case OpCode::SUB:
          case OpCode::MUL:
          case OpCode::DIV:
          case OpCode::MOD:
          // Comparisons
          case OpCode::EQ:
          case OpCode::NE:
          case OpCode::LT:
          case OpCode::LE:
          case OpCode::GT:
          case OpCode::GE:
          // Binary Logical
          case OpCode::LAND:
          case OpCode::LOR: {
            WorkT right = stack[top--];
            WorkT left = stack[top--];
            stack[++top] = this->apply_op(left, right, op);
            break;
          }
          // Skipping BAND, BOR, XOR, LSHFT, RSHFT
          default:
            break;
          }
        }
        return op_traits<WorkT>::template cast<DestT>(stack[0]);
      };

      if constexpr (hd_std::is_same_v<DestT, bool>) {
        return dispatch(this->work_dtype, eval_op);
      } else {
        return dispatch(dtype_traits<DestT>::value, eval_op);
      }
    }
  };

#ifndef __CUDACC_RTC__

  template <typename MemTag>
  inline size_t bytes_for_dynamic_vm(const ExprMVNode<MemTag>& node, size_t alignment = 16) {
    auto align = [&](size_t off) { return (off + (alignment - 1)) & ~(alignment - 1); };
    auto n_instrs = node.instrs.size();
    auto n_arrays = node.layouts.size();
    auto n_scalars = node.scalars.size();

    size_t aligned_layout_size { align(sizeof(SOArrayPolicy)) };
    size_t total_bytes =
      align(n_instrs  * sizeof(Instruction)) + // Packed instructions
      n_arrays  * aligned_layout_size        + // Array layout structs
      align(n_arrays  * sizeof(void*))       + // Array data
      align(n_arrays  * sizeof(DType))       + // Array data types
      align(n_scalars * sizeof(DType))       + // Scalar data types
      align(n_scalars * 2)                   + // 2 bytes for each scalar offset
      align(n_scalars * 32);                   // Largest supported scalar type is 32 bytes

    return (total_bytes + (alignment - 1)) & ~(alignment - 1);
  }

  template <typename MemTag>
  inline DynamicExprMVNode<MemTag> get_dynamic_mv_node(const ExprMVNode<MemTag>& node,
                                                       hd_std::uint8_t* h_ptr,
                                                       hd_std::uint8_t* d_ptr = nullptr,
                                                       size_t alignment = 16) {
    auto align = [&](size_t off) { return (off + (alignment - 1)) & ~(alignment - 1); };

    auto n_instrs = node.instrs.size();
    auto n_arrays = node.layouts.size();
    auto n_scalars = node.scalars.size();

    size_t offset { 0 };
    std::copy(node.instrs.begin(),
              node.instrs.end(),
              reinterpret_cast<Instruction*>(h_ptr));
    offset += node.instrs.size() * sizeof(Instruction);
    offset = align(offset);

    size_t aligned_layout_size { align(sizeof(SOArrayPolicy)) };
    for (size_t i = 0; i < n_arrays; ++i) {
      auto* layout_ptr = reinterpret_cast<SOArrayPolicy*>(h_ptr + offset + i * aligned_layout_size);
      *layout_ptr = node.layouts[i];
    }
    offset += aligned_layout_size * n_arrays;

    std::copy(node.data.begin(),
              node.data.end(),
              reinterpret_cast<void**>(h_ptr + offset));
    offset += node.data.size() * sizeof(void**);
    offset = align(offset);

    std::copy(node.dtypes.begin(),
              node.dtypes.end(),
              reinterpret_cast<DType*>(h_ptr + offset));
    offset += node.dtypes.size() * sizeof(DType);
    offset = align(offset);

    auto scalar_dtype_bytes { sizeof(DType) * node.scalars.size() };
    auto scalar_off_bytes { 2 * node.scalars.size() };

    unsigned scalar_cnt { 0 };
    hd_std::uint16_t scalar_off { 0 };

    auto scalar_cast = [&](auto&& val) {
      using SrcT = hd_std::decay_t<decltype(val)>;
      auto* bytes = reinterpret_cast<const hd_std::uint8_t*>(&val);
      // Data types begin at memory: (offset + sizeof(DType) * scalar_cnt)
      size_t dtypes_off = align(offset);
      auto* dtype_buf = reinterpret_cast<DType*>(h_ptr + dtypes_off);
      dtype_buf[scalar_cnt] = dtype_traits<SrcT>::value;

      // Scalar offsets begin at:
      // (offset + node.scalars.size() * sizeof(DType) + scalar_cnt * 2)
      size_t offsets_off = align(offset + scalar_dtype_bytes);
      auto* offsets_buf = reinterpret_cast<hd_std::uint16_t*>(h_ptr + offsets_off);
      hd_std::uint16_t off = scalar_off;
      offsets_buf[scalar_cnt] = off;
      scalar_off += sizeof(val);

      // Actual scalars buffer begin at:
      // (offset + scalar_dtype_bytes + scalar_off_bytes)
      size_t scalars_buf_off = align(align(offset + scalar_dtype_bytes) + scalar_off_bytes);
      hd_std::uint8_t* scalars_buf = h_ptr + scalars_buf_off;
      for (unsigned i = 0; i < sizeof(val); ++i) {
        scalars_buf[off + i] = bytes[i];
      }
      scalar_cnt++;
    };

    for (const auto& scalar : node.scalars) {
      std::visit(scalar_cast, scalar);
    }

    if constexpr (hd_std::is_same_v<MemTag, HostTag>) {
      return DynamicExprMVNode<MemTag>(node, h_ptr, n_instrs, n_arrays, n_scalars, alignment);
    } else {
      return DynamicExprMVNode<MemTag>(node, d_ptr, n_instrs, n_arrays, n_scalars, alignment);
    }
  }
#endif

} // namespace ncarray

#endif // NCARRAY_EXPRESSION_DYNAMICMVNODE_HH
