/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_EXPRESSION_MVNODE_HH
#define NCARRAY_EXPRESSION_MVNODE_HH

#include "ncarray/custom_types.hh"
#ifdef __CUDACC__
#ifndef __CUDACC_RTC__
#include "ncarray/device/casts.cuh"
#endif
#endif
#include "ncarray/dtype.hh"
#include "ncarray/expression/interface.hh"
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
#ifndef __CUDACC_RTC__
  // The ExprMVNode is a host-only expression builder. Guard NVRTC from seeing it

  /**
   * @brief A multi-view array expression builder.
   *
   * ExprMVNode is a host-only expression builder. It has no restrictions on the
   * number, or type of arrays (e.g. operands including views from multiple arrays
   * are supported).
   */
  template <class MemTag = DevTag>
  struct ExprMVNode : public ExpressionTag {
    std::vector<SOArrayPolicy> layouts; // Same mem layout for NC and SO layouts
    std::vector<DType> dtypes;
    std::vector<void*> data;
    std::vector<Instruction> instrs;
    std::vector<Scalar> scalars;
    bool soarray { true };

    Metadata final_shape;
    ssize_t final_size { 0 };
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
          if constexpr (hd_std::is_same_v<MemTag, HostTag>) {
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

      if constexpr (hd_std::is_same_v<DestT, bool>) {
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
        if constexpr (hd_std::is_same_v<DestT, bool>) {
          return res && leaf;
        } else {
          return res * leaf;
        }
      } else if (op == OpCode::DIV) {
        bool is_finite { op_traits<DestT>::isfinite(res) };

        if (leaf == DestT(0)) {
          return is_finite ? hd_std::nan("") : res;
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
          final_shape.set(node.shape(), node.ndim());
          final_size = node.size();
        } else {
          // Ensure both true types and the variant get processed
          Scalar tmp = node;
          auto get_dtype = [](auto&& val) {
            using T = hd_std::decay_t<decltype(val)>;
            return dtype_traits<T>::value;
          };
          expr_dtype = std::visit(get_dtype, tmp);
          work_dtype = expr_dtype;
          final_shape.data[0] = 0;
          final_shape.ndim = 0;
          final_size = 0;
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
      return this->layouts.empty() ? final_shape.data : this->layouts[0].shape();
    }
    inline ssize_t size() const {
      return this->layouts.empty() ? final_size : this->layouts[0].size();
    }
    inline ssize_t ndim() const {
      return this->layouts.empty() ? final_shape.ndim : this->layouts[0].ndim();
    }
    inline DType dtype() const { return this->expr_dtype; }

    // -- Unary ops --- //

    inline auto operator-() {
      this->expr_dtype = determine_dtype_for_op(OpCode::NEG, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::NEG, 0));

      return *this;
    }

    inline auto operator++() {
      this->expr_dtype = determine_dtype_for_op(OpCode::INC, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::INC, 0));

      return *this;
    }

    inline auto operator--() {
      this->expr_dtype = determine_dtype_for_op(OpCode::DEC, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::DEC, 0));

      return *this;
    }

    inline auto operator!() {
      this->expr_dtype = determine_dtype_for_op(OpCode::LNOT, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::LNOT, 0));

      return *this;
    }

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

    template <AnyExpressionOrScalar RightNode>
    inline auto operator%(const RightNode& right) {
      build_node(right);
      DType r_dtype { node_dtype(right) };
      this->work_dtype = promote_expr_types(this->work_dtype, r_dtype);
      this->expr_dtype = determine_dtype_for_op(OpCode::MOD, this->work_dtype);
      this->instrs.push_back(pack_instruction(OpCode::MOD, 0));
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

  private:
    template <class Node>
    inline DType node_dtype(const Node& node) const {
      if constexpr (ArrayLike<Node> || std::is_base_of_v<ExprMVNode, Node>) {
        return node.dtype();
      } else {
        Scalar tmp = node;

        auto dtype_op = [](auto&& val) {
          using ScalarT = hd_std::decay_t<decltype(val)>;
          return dtype_traits<ScalarT>::value;
        };

        return std::visit(dtype_op, tmp);
      }
    }
  };

  template <class Expr>
  struct is_exprmv_node : hd_std::false_type {};

  template <typename MemTag>
  struct is_exprmv_node<ExprMVNode<MemTag>> : hd_std::true_type {};

  template <class Expr>
  constexpr bool is_exprmv_node_v = is_exprmv_node<Expr>::value;
#endif // nvrtc guard
} // namespace ncarray

#endif // NCARRAY_EXPRESSION_MVNODE_HH
