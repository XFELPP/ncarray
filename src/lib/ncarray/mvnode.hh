/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_MVNODE_HH
#define NCARRAY_MVNODE_HH

#include "ncarray/custom_types.hh"
#ifdef __CUDACC__
#ifndef __CUDACC_RTC__
#include "ncarray/device/casts.cuh"
#endif
#endif
#include "ncarray/dtype.hh"
#include "ncarray/expression.hh"
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

using cuda::std::false_type;
using cuda::std::is_same_v;
using cuda::std::isfinite;
using cuda::std::true_type;
using cuda::std::uint8_t;
using cuda::std::uint16_t;

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

using std::false_type;
using std::is_same_v;
using std::isfinite;
using std::nan;
using std::true_type;
using std::uint8_t;
using std::uint16_t;

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
#ifndef __CUDACC_RTC__

  /**
   * Test for supported types of objects when creating expressions for evaluation.
   *
   * This includes anything that descends from the ExpressionTag, the Scalar variant
   * and true C++ types if they reside in the many type list from dtype.hh.
   */
  template <typename T>
  concept AnyExpressionOrScalar =
    AnyExpression<T>                               ||
    std::is_same_v<std::decay_t<T>, Scalar>        ||
    is_in_type_list_v<std::decay_t<T>, base_types>;

  /**
   * The enum DTypes are already ordered. For expressions' working dtype, promote
   * to the numerically larger. The final value is determined using op_traits.
   */
  inline DType promote_expr_types(DType a, DType b) {
    if (a == b) {
      return a;
    }

    if (static_cast<int>(a) > static_cast<int>(b)) {
      return a;
    }
    return b;
  }

  template <class MemTag = DevTag>
  struct ExprMVNode : public ExpressionTag {
    std::vector<SOArrayPolicy> layouts; // Same mem layout for NC and SO layouts
    std::vector<DType> dtypes;
    std::vector<void*> data;
    std::vector<Instruction> instrs;
    std::vector<Scalar> scalars;
    bool soarray { true };

    DType expr_dtype; ///< The DType for the final evaluated expression
    DType work_dtype; ///< The DType for intermediate sub-expression evaluations
    bool has_expr_dtype { false };

    /**
     * Default constructor provided as the converting one below exists.
     */
    ExprMVNode() = default;

    /**
     * A converting constructor is provided for ergonomics.
     *
     * This mostly allows inplace operators to work without end-users needing to think
     * or know about the underlying expression tape/tree/node/etc....
     */
    template <AnyExpressionOrScalar Node>
    ExprMVNode(const Node& node) {
      this->build_node(node);
    }

    template <typename DestT, typename Coords>
    inline DestT eval(const Coords& coords) const {
      // NOTE: This eval call can only run on the host due to the containers used.
      auto eval_op = [&] <typename WorkT> () {
        WorkT res { 0 };

        for (std::size_t i = 0; i < layouts.size(); ++i) {
          auto& layout = layouts[i];
          const void* leaf_ptr = layout.advance(data[i], coords);

          int src_idx { static_cast<int>(dtypes[i]) };

          WorkT leaf_res;
          if constexpr (std::is_same_v<MemTag, HostTag>) {
  #ifndef __CUDA_ARCH__
            leaf_res = host::vm_cast_table<WorkT>[src_idx](leaf_ptr);
  #endif
          } else {
  #ifdef __CUDACC__
            leaf_res = device::device_cast<WorkT>(src_idx, leaf_ptr);
  #endif
          }

          if (i == 0) {
            res = leaf_res;
          } else {
            OpCode op = get_op(instrs[i]);
            res = this->template apply_op<WorkT>(res, leaf_res, op);
          }
        }

        return static_cast<DestT>(res);
      };

      if constexpr (std::is_same_v<DestT, bool>) {
        // In the case of bool as the final result (e.g. for comparisons) we will do
        // the rest of the evaluation in the working datatype to avoid precision loss.
        return dispatch(this->work_dtype, eval_op);
      } else {
        return dispatch(dtype_traits<DestT>::value, eval_op);
      }
    }

    template <typename DestT>
    inline DestT apply_op(DestT res, DestT leaf, OpCode op) const {
      if (op == OpCode::ADD) {
        return res + leaf;
      } else if (op == OpCode::SUB) {
        return res - leaf;
      } else if (op == OpCode::MUL) {
        if constexpr (is_same_v<DestT, bool>) {
          return res && leaf;
        } else {
          return res * leaf;
        }
      } else if (op == OpCode::DIV) {
        bool is_finite { op_traits<DestT>::isfinite(res) };

        if (leaf == DestT(0)) {
          return is_finite ? nan("") : res;
        }
        return res / leaf;
        // Comparisons
      }
      return DestT{};
    }

    template <AnyExpressionOrScalar Node>
    inline auto build_node(const Node& node) {
      if (!has_expr_dtype) {
        // If we have never seen an operand before, set the total expr dtype
        // to the dtype of the operand
        // Otherwise, it gets updated for each operation (see, e.g., operator+ below)
        if constexpr (ArrayLike<Node> || std::is_base_of_v<ExprMVNode, Node>) {
          expr_dtype = node.dtype();
          work_dtype = node.dtype();
        } else {
          // Ensure both true types and the variant get processed
          Scalar tmp = node;
          auto get_dtype = [](auto&& val) {
            using T = std::decay_t<decltype(val)>;
            return dtype_traits<T>::value;
          };
          expr_dtype = std::visit(get_dtype, tmp);
          work_dtype = expr_dtype;
        }
        has_expr_dtype = true;
      }
      if constexpr (ArrayLike<Node>) {
        OpCode code = OpCode::LOAD_NCARR;
        if constexpr (std::is_base_of_v<SOArrayPolicy, Node>) {
          code = OpCode::LOAD_SOARR;
          this->layouts.push_back(static_cast<const SOArrayPolicy&>(node));
        } else {
          // If an NCOffsets array, the memory layout is the same so this is safe.
          // The OpCode will tell use how to load it.
          this->layouts.push_back(reinterpret_cast<const SOArrayPolicy&>(node));
          this->soarray = false;
        }
        this->dtypes.push_back(node.dtype());
        this->data.push_back(node.data());
        this->instrs.push_back(pack_instruction(code,
                                                static_cast<int>(this->layouts.size()) - 1));
      } else if constexpr(std::is_base_of_v<ExprMVNode, Node>){
        // When inserting new arrays and scalars, the instruction indices must
        // be shifted. They were calculated relative to the start of the node
        // so now need to get shifted by this' current size of each vector
        size_t arr_idx_shift { this->layouts.size() };
        size_t scalar_idx_shift { this->scalars.size() };

        this->layouts.insert(this->layouts.end(),
                             node.layouts.begin(),
                             node.layouts.end());
        this->dtypes.insert(this->dtypes.end(),
                            node.dtypes.begin(),
                            node.dtypes.end());
        this->data.insert(this->data.end(), node.data.begin(), node.data.end());

        for (const auto& instr : node.instrs) {
          OpCode op = get_op(instr);
          int idx = get_index(instr);

          if (op == OpCode::LOAD_NCARR || op == OpCode::LOAD_SOARR) {
            this->instrs.push_back(pack_instruction(op, idx + arr_idx_shift));
          } else if (op == OpCode::LOAD_CONST) {
            this->instrs.push_back(pack_instruction(op, idx + scalar_idx_shift));
          } else {
            // For non-load ops, the index is irrelevant
            this->instrs.push_back(instr);
          }
        }

        this->scalars.insert(this->scalars.end(),
                             node.scalars.begin(),
                             node.scalars.end());
      } else {
        // Scalars
        this->scalars.push_back(node);
        this->instrs.push_back(pack_instruction(OpCode::LOAD_CONST,
                                                static_cast<int>(this->scalars.size()) - 1));
      }
    }

    // NOTE: It is extraordinarily unlikely that these functions will be called before
    //       a layout is inserted. However, it is theoretically possible as an ExprMVNode
    //       can be constructed (using the converting constructor) from a scalar only.
    //       It should in theory be used immediately to combine with an array; however,
    //       we'll guard the empty layout case just in case.
    inline const ssize_t* shape() const {
      return this->layouts.empty() ? nullptr : this->layouts[0].shape();
    }
    inline ssize_t size() const {
      return this->layouts.empty() ? 1 : this->layouts[0].size();
    }
    inline ssize_t ndim() const {
      return this->layouts.empty() ? 0 : this->layouts[0].ndim();
    }
    inline DType dtype() const { return this->expr_dtype; }

    // Arithmetic

    template <AnyExpressionOrScalar RightNode>
    inline auto operator+(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::ADD, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::ADD, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    inline auto operator-(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::SUB, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::SUB, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    inline auto operator*(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::MUL, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::MUL, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    inline auto operator/(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::DIV, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::DIV, 0));
      return *this;
    }

    // Comparisons

    template <AnyExpressionOrScalar RightNode>
    inline auto operator==(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::EQ, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::EQ, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    inline auto operator!=(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::NE, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::NE, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    inline auto operator<(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::LT, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::LT, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    inline auto operator<=(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::LE, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::LE, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    inline auto operator>(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::GT, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::GT, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    inline auto operator>=(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::GE, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::GE, 0));
      return *this;
    }

    // Logical

    template <AnyExpressionOrScalar RightNode>
    inline auto operator&&(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::LAND, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::LAND, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    inline auto operator||(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::LOR, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::LOR, 0));
      return *this;
    }

    inline auto operator!() {
      this->expr_dtype = determine_dtype_for_op(OpCode::LNOT, this->expr_dtype);
      this->instrs.push_back(pack_instruction(OpCode::LNOT, 0));

      return *this;
    }

  private:
    template <class Node>
    inline DType node_dtype(const Node& node) const {
      if constexpr (ArrayLike<Node> || std::is_base_of_v<ExprMVNode, Node>) {
        return node.dtype();
      } else {
        Scalar tmp = node;

        auto dtype_op = [](auto&& val) {
          using ScalarT = std::decay_t<decltype(val)>;
          return dtype_traits<ScalarT>::value;
        };

        return std::visit(dtype_op, tmp);
      }
    }
  };
#endif // nvrtc guard

  template <
    int NViews,
    int NScalars,
    int NInstrs,
    typename ArrT = float,
    typename ScalarT = float,
    class MemTag = DevTag,
    class Layout = NCOffsetsPolicy
  >
  struct StaticExprMVNode : public ExpressionTag {
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
      for (int i = 0; i < NViews; ++i) {
        layouts[i] = node.layouts[i];
        data[i] = node.data[i];
      }

      auto cast_op = [&](auto&& val) {
        using SrcT = std::decay_t<decltype(val)>;
        return op_traits<SrcT>::template cast<ScalarT>(val);
      };

      int scalar_i { 0 };
      for (const auto& scalar : node.scalars) {
        scalars[scalar_i++] = std::visit(cast_op, scalar);
      }

      int operand_ptr { 0 };
      int op_ptr { 0 };
      for (const auto& instr : node.instrs) {
        int idx = get_index(instr);
        OpCode op = get_op(instr);

        if (op == OpCode::LOAD_NCARR || op == OpCode::LOAD_SOARR) {
          op_map[operand_ptr++] = idx;
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
        auto& layout = layouts[i];
        const void* leaf_ptr = layout.advance(data[i], coords);

        const ArrT item = *reinterpret_cast<const ArrT*>(leaf_ptr);
        values[i] = op_traits<ArrT>::template cast<ScalarT>(item);
      }

      #pragma unroll NScalars
      for (int i = 0; i < NScalars; ++i) {
        values[NViews + i] = scalars[i];
      }

      ScalarT res { values[op_map[0]] };

      #pragma unroll NOps
      for (int i = 0; i < NOps; ++i) {
        ScalarT next_v { values[op_map[i + 1]] };
        res = this->apply_op(res, next_v, ops[i]);
      }
      return op_traits<ArrT>::template cast<DestT>(res);
    }
    template <typename OpT>
    NCA_HD inline OpT apply_op(OpT res, OpT leaf, OpCode op) const {
      if (op == OpCode::ADD) {
        return static_cast<OpT>(res + leaf);
      } else if (op == OpCode::SUB) {
        return static_cast<OpT>(res - leaf);
      } else if (op == OpCode::MUL) {
        if constexpr (is_same_v<OpT, bool>) {
          return static_cast<OpT>(res && leaf);
        } else {
          return static_cast<OpT>(res * leaf);
        }
      } else if (op == OpCode::DIV) {
        bool is_finite { op_traits<OpT>::isfinite(res) };

        if (leaf == OpT(0)) {
          return is_finite ? static_cast<OpT>(nan("")) : res;
        }
        return static_cast<OpT>(res / leaf);
        // Comparisons
      } else if (op == OpCode::EQ) {
        return static_cast<OpT>(res == leaf);
      } else if (op == OpCode::NE) {
        return static_cast<OpT>(res != leaf);
      } else if (op == OpCode::LT) {
        return static_cast<OpT>(op_traits<OpT>::less(res, leaf));
      } else if (op == OpCode::LE) {
        return static_cast<OpT>(op_traits<OpT>::le(res, leaf));
      } else if (op == OpCode::GT) {
        return static_cast<OpT>(op_traits<OpT>::greater(res, leaf));
      } else if (op == OpCode::GE) {
        return static_cast<OpT>(op_traits<OpT>::ge(res, leaf));
      } else if (op == OpCode::LAND) {
        return static_cast<OpT>(op_traits<OpT>::land(res, leaf));
      } else if (op == OpCode::LOR) {
        return static_cast<OpT>(op_traits<OpT>::lor(res, leaf));
      }
      return OpT{};
    }

    NCA_HD inline const ssize_t* shape() const { return layouts[0].shape(); }
    NCA_HD inline ssize_t size() const { return layouts[0].size(); }
    NCA_HD inline ssize_t ndim() const { return layouts[0].ndim(); }
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
      } else if (op == OpCode::LOAD_CONST) {
        stack_depth++;
      } else {
        if (stack_depth < 2) {
          return false;
        }
        stack_depth--;
      }

      if (stack_depth > 2) {
        return false;
      }
    }
    return stack_depth == 1;
  }

#endif

  template <class MemTag = DevTag, int MaxStackSize = 24>
  struct DynamicExprMVNode : public ExpressionTag {
    const Instruction* instrs { nullptr };         ///< Instruction stack

    const SOArrayPolicy* layouts { nullptr };      ///< Array layout operands
    const void* const * data { nullptr };          ///< Array data
    const DType* arr_dtypes { nullptr };           ///< Datatypes for arrays

    // Scalars
    const DType* constants_dtypes { nullptr };     ///< Scalar data types
    const uint16_t* constants_offsets { nullptr }; ///< Scalar offset in byte stream
    // Largest supported scalar type is 32 bytes.
    const uint8_t* constants_buf { nullptr };      ///< Byte stream of scalars

    Metadata final_shape; ///< The result shape
    ssize_t final_size;   ///< The result number of elements
    DType expr_dtype;     ///< The DType for the final evaluated expression
    DType work_dtype;     ///< The DType for intermediate sub-expression evaluations

    uint8_t n_layouts { 0 };
    uint8_t n_scalars { 0 };
    uint8_t n_instrs { 0 };

    size_t arr_alignment { 16 };

#ifndef __CUDACC_RTC__
    DynamicExprMVNode(const ExprMVNode<MemTag>& node,
                      const uint8_t* buf,
                      uint8_t n_instrs_,
                      uint8_t n_arrays_,
                      uint8_t n_scalars_,
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

      uint16_t offset { 0 };
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

      constants_offsets = reinterpret_cast<const uint16_t*>(buf + offset);
      offset += n_scalars * 2;
      offset = align(offset);

      constants_buf = buf + offset;
    }
#endif

    NCA_HD inline ssize_t ndim() const { return final_shape.ndim; }

    NCA_HD inline const ssize_t* shape() const { return final_shape.data; }

    NCA_HD inline DType dtype() const { return expr_dtype; }

    NCA_HD inline ssize_t size() const { return final_size; }

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
          case OpCode::LOAD_NCARR:
          case OpCode::LOAD_SOARR: {
            int arr_idx = get_index(instr);
            const uint8_t* layout_ptr = reinterpret_cast<const uint8_t*>(layouts);
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

            if constexpr (is_same_v<MemTag, HostTag>) {
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

            if constexpr (is_same_v<MemTag, HostTag>) {
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
          // Arithmetic
          case OpCode::ADD:
          case OpCode::SUB:
          case OpCode::MUL:
          case OpCode::DIV:
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
          default:
            break;
          }
        }
        return op_traits<WorkT>::template cast<DestT>(stack[0]);
      };

      if constexpr (std::is_same_v<DestT, bool>) {
        return dispatch(this->work_dtype, eval_op);
      } else {
        return dispatch(dtype_traits<DestT>::value, eval_op);
      }
    }

    template <typename OpT>
    NCA_HD inline OpT apply_op(OpT res, OpT leaf, OpCode op) const {
      if (op == OpCode::ADD) {
        return static_cast<OpT>(res + leaf);
      } else if (op == OpCode::SUB) {
        return static_cast<OpT>(res - leaf);
      } else if (op == OpCode::MUL) {
        if constexpr (is_same_v<OpT, bool>) {
          return static_cast<OpT>(res && leaf);
        } else {
          return static_cast<OpT>(res * leaf);
        }
      } else if (op == OpCode::DIV) {
        bool is_finite { op_traits<OpT>::isfinite(res) };

        if (leaf == OpT(0)) {
          return is_finite ? static_cast<OpT>(nan("")) : res;
        }
        return static_cast<OpT>(res / leaf);
        // Comparisons
      } else if (op == OpCode::EQ) {
        return static_cast<OpT>(res == leaf);
      } else if (op == OpCode::NE) {
        return static_cast<OpT>(res != leaf);
      } else if (op == OpCode::LT) {
        return static_cast<OpT>(op_traits<OpT>::less(res, leaf));
      } else if (op == OpCode::LE) {
        return static_cast<OpT>(op_traits<OpT>::le(res, leaf));
      } else if (op == OpCode::GT) {
        return static_cast<OpT>(op_traits<OpT>::greater(res, leaf));
      } else if (op == OpCode::GE) {
        return static_cast<OpT>(op_traits<OpT>::ge(res, leaf));
      } else if (op == OpCode::LAND) {
        return static_cast<OpT>(op_traits<OpT>::land(res, leaf));
      } else if (op == OpCode::LOR) {
        return static_cast<OpT>(op_traits<OpT>::lor(res, leaf));
      }
      return OpT{};
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
                                                       uint8_t* h_ptr,
                                                       uint8_t* d_ptr = nullptr,
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
    uint16_t scalar_off { 0 };

    auto scalar_cast = [&](auto&& val) {
      using SrcT = std::decay_t<decltype(val)>;
      auto* bytes = reinterpret_cast<const uint8_t*>(&val);
      // Data types begin at memory: (offset + sizeof(DType) * scalar_cnt)
      size_t dtypes_off = align(offset);
      auto* dtype_buf = reinterpret_cast<DType*>(h_ptr + dtypes_off);
      dtype_buf[scalar_cnt] = dtype_traits<SrcT>::value;

      // Scalar offsets begin at:
      // (offset + node.scalars.size() * sizeof(DType) + scalar_cnt * 2)
      size_t offsets_off = align(offset + scalar_dtype_bytes);
      auto* offsets_buf = reinterpret_cast<uint16_t*>(h_ptr + offsets_off);
      uint16_t off = scalar_off;
      offsets_buf[scalar_cnt] = off;
      scalar_off += sizeof(val);

      // Actual scalars buffer begin at:
      // (offset + scalar_dtype_bytes + scalar_off_bytes)
      size_t scalars_buf_off = align(align(offset + scalar_dtype_bytes) + scalar_off_bytes);
      uint8_t* scalars_buf = h_ptr + scalars_buf_off;
      for (unsigned i = 0; i < sizeof(val); ++i) {
        scalars_buf[off + i] = bytes[i];
      }
      scalar_cnt++;
    };

    for (const auto& scalar : node.scalars) {
      std::visit(scalar_cast, scalar);
    }

    if constexpr (is_same_v<MemTag, HostTag>) {
      return DynamicExprMVNode<MemTag>(node, h_ptr, n_instrs, n_arrays, n_scalars, alignment);
    } else {
      return DynamicExprMVNode<MemTag>(node, d_ptr, n_instrs, n_arrays, n_scalars, alignment);
    }
  }

  // --- Additional Traits and Concepts --- //

  template <class Expr>
  struct is_exprmv_node : false_type {};

  template <typename MemTag>
  struct is_exprmv_node<ExprMVNode<MemTag>> : true_type {};

  template <class Expr>
  constexpr bool is_exprmv_node_v = is_exprmv_node<Expr>::value;

#endif

} // namespace ncarray

#endif // NCARRAY_MVNODE_HH
