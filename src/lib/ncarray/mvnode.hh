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

  template <typename T>
  concept AnyExpressionOrScalar = AnyExpression<T> || std::is_same_v<T, Scalar>;

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
    NCA_HD inline auto build_node(const Node& node) {
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
        this->layouts.insert(this->layouts.end(),
                             node.layouts.begin(),
                             node.layouts.end());
        this->dtypes.insert(this->dtypes.end(),
                            node.dtypes.begin(),
                            node.dtypes.end());
        this->data.insert(this->data.end(), node.data.begin(), node.data.end());
        this->instrs.insert(this->instrs.end(),
                            node.instrs.begin(),
                            node.instrs.end());
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

    NCA_HD inline const ssize_t* shape() const { return this->layouts[0].shape(); }
    NCA_HD inline ssize_t size() const { return this->layouts[0].size(); }
    NCA_HD inline ssize_t ndim() const { return this->layouts[0].ndim(); }
    NCA_HD inline DType dtype() const { return this->expr_dtype; }

    // Arithmetic

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator+(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::ADD, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::ADD, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator-(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::SUB, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::SUB, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator*(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::MUL, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::MUL, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator/(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::DIV, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::DIV, 0));
      return *this;
    }

    // Comparisons

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator==(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::EQ, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::EQ, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator!=(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::NE, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::NE, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator<(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::LT, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::LT, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator<=(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::LE, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::LE, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator>(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::GT, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::GT, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator>=(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::GE, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::GE, 0));
      return *this;
    }

    // Logical

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator&&(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::LAND, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::LAND, 0));
      return *this;
    }

    template <AnyExpressionOrScalar RightNode>
    NCA_HD inline auto operator||(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::LOR, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::LOR, 0));
      return *this;
    }

    NCA_HD inline auto operator!() {
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
          return is_finite ? nan("") : res;
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

  template <class MemTag = DevTag, int NLayouts = 16, int NScalars = 8, int NInstrs = 24>
  struct DynamicExprMVNode : public ExpressionTag {
    Instruction instrs[NInstrs > 1 ? NInstrs : 1];                 ///< Instruction stack

    SOArrayPolicy layouts[NLayouts > 1 ? NLayouts : 1];            ///< Array layout operands
    const void* data[NLayouts > 1 ? NLayouts : 1];                 ///< Array data
    DType arr_dtypes[NLayouts > 1 ? NLayouts : 1];                 ///< Datatypes for arrays

    // Largest supported scalar type is 32 bytes.
    uint8_t constants_buf[NScalars > 1 ? NScalars * 32 : 1] { 0 }; ///< Scalars as byte stream
    uint16_t constants_offsets[NScalars > 1 ? NScalars : 1] { 0 }; ///< Offset in stream
    DType constants_dtypes[NScalars > 1 ? NScalars : 1];           ///< Scalar datatypes

    Metadata m_shape;
    ssize_t m_size;
    DType m_dtype;

    uint8_t n_layouts { 0 };
    uint8_t n_scalars { 0 };
    uint8_t n_instrs { 0 };

#ifndef __CUDACC_RTC__
    DynamicExprMVNode(const ExprMVNode<MemTag>& node) {
      uint16_t c_off { 0 };

      auto scalar_cast = [&] (auto&& val) {
        using SrcT = std::decay_t<decltype(val)>;
        auto* bytes = reinterpret_cast<const uint8_t*>(&val);
        constants_dtypes[n_scalars] = dtype_traits<SrcT>::value;
        uint16_t off = c_off;
        constants_offsets[n_scalars] = off;
        c_off += sizeof(val);
        for (unsigned i = 0; i < sizeof(val); ++i) {
          constants_buf[off + i] = bytes[i];
        }
        n_scalars++;
      };
      n_layouts = node.layouts.size();
      for (unsigned i = 0; i < n_layouts; ++i) {
        layouts[i] = node.layouts[i];
        data[i] = node.data[i];
        arr_dtypes[i] = node.dtypes[i];
      }

      n_instrs = node.instrs.size();
      for (int i = 0; i < n_instrs; ++i) {
        instrs[i] = node.instrs[i];
      }

      for (const auto& scalar : node.scalars) {
        std::visit(scalar_cast, scalar);
      }
    }
#endif

    NCA_HD inline ssize_t ndim() const { return m_shape.ndim; }

    NCA_HD inline const ssize_t* shape() const { return m_shape.data; }

    NCA_HD inline DType dtype() const { return m_dtype; }

    NCA_HD inline ssize_t size() const { return m_size; }

    NCA_HD inline ssize_t itemsize() const { return ncarray::itemsize(this->dtype()); }

    template <typename DestT, typename Coords>
    NCA_HD inline DestT eval(const Coords& coords) const {
      auto eval_op = [&] <typename WorkT> () {
        WorkT stack[NInstrs] { 0 };
        int top { -1 };

        for (int i = 0; i < n_instrs; ++i) {
          Instruction instr = instrs[i];
          OpCode op = get_op(instr);

          switch (op) {
          case OpCode::LOAD_NCARR:
          case OpCode::LOAD_SOARR: {
            int arr_idx = get_index(instr);
            const auto& view = layouts[arr_idx];

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
          /*
          // Arithmetic
          case OpCode::ADD: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            stack[++top] = left + right;
            break;
          }
          case OpCode::SUB: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            stack[++top] = left - right;
            break;
          }
          case OpCode::MUL: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            if constexpr (is_same_v<DestT, bool>) {
              stack[++top] = left && right;
            } else {
              stack[++top] = left * right;
            }
            break;
          }
          case OpCode::DIV: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            stack[++top] = left / right;
            break;
          }
          // Comparisons
          case OpCode::EQ: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            stack[++top] = left == right;
            break;
          }
          case OpCode::NE: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            stack[++top] = left != right;
            break;
          }
          case OpCode::LT: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            stack[++top] = op_traits<DestT>::less(left, right);
            break;
          }
          case OpCode::LE: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            stack[++top] = op_traits<DestT>::le(left, right);
            break;
          }
          case OpCode::GT: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            stack[++top] = op_traits<DestT>::greater(left, right);
            break;
          }
          case OpCode::GE: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            stack[++top] = op_traits<DestT>::ge(left, right);
            break;
          }
            */
          // Logical
            /*
          case OpCode::LAND: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            stack[++top] = left && right;
          }
          case OpCode::LOR: {
            DestT right = stack[top--];
            DestT left = stack[top--];
            stack[++top] = left || right;
          }
            */
          default:
            break;
          }
        }
        return op_traits<WorkT>::template cast<DestT>(stack[0]);
      };

      if constexpr (std::is_same_v<DestT, bool>) {
        return dispatch(this->arr_dtypes[0], eval_op);
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
          return is_finite ? nan("") : res;
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
