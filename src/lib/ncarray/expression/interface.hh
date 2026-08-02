/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_EXPRESSION_INTERFACE_HH
#define NCARRAY_EXPRESSION_INTERFACE_HH

#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"
#include "ncarray/layout.hh"
#include "ncarray/op_code.hh"
#include "ncarray/op_traits.hh"

#ifdef __CUDACC_RTC__
typedef long long ssize_t;

#include <cuda/std/type_traits>

namespace hd_std = cuda::std;

#else

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <concepts>
#include <type_traits>

namespace hd_std = std;

#endif // nvrtc check

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
  struct ExpressionTag {};

  // Helper to check if a type is an expression
  template <typename T>
  concept Expression = hd_std::is_base_of_v<ExpressionTag, T>;

  /**
   * Defines array-like expressions.
   */
  template <typename T>
  concept ArrayExpression = Expression<T> || ArrayLike<T>;

  // Numeric comes from custom_types.hh (basically std::is_arithmetic_v)
  /**
   * Defines all supported types for use in expressions.
   *
   * This provides support for building expression trees beyond just a
   * a simple binary operation.
   */
  template <typename T>
  concept AnyExpression =
    Expression<T> || ArrayLike<T> || Numeric<T> || Vector2DType<T>;

#ifndef __CUDACC_RTC__ // Because it has the variant, cannot be including in NVRTC
  /**
   * Test for supported types of objects when creating expressions for evaluation.
   *
   * This includes anything that descends from the ExpressionTag, the Scalar variant
   * and true C++ types if they reside in the many type list from dtype.hh.
   */
  template <typename T>
  concept AnyExpressionOrScalar =
    AnyExpression<T>                                   ||
    hd_std::is_same_v<hd_std::decay_t<T>, Scalar>      ||
    is_in_type_list_v<hd_std::decay_t<T>, base_types>;

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

#endif // nvrtc guard

  struct ExprOpInterface {
    Metadata final_shape; ///< The result shape
    ssize_t final_size;   ///< The result number of elements

    NCA_HD inline ssize_t ndim() const { return final_shape.ndim; }

    NCA_HD inline const ssize_t* shape() const { return final_shape.data; }

    NCA_HD inline ssize_t size() const { return final_size; }

    template <typename OpT, typename Coords>
    NCA_HD inline OpT md_to_lin(Coords coords) const {
      using IdxT = hd_std::decay_t<decltype(coords[0])>;
      IdxT lin_idx { 0 };
      IdxT cum_stride { 1 };

      for (int dim = static_cast<int>(coords.size()) - 1; dim >= 0; --dim) {
        lin_idx += coords[dim] * cum_stride;
        cum_stride *= this->shape()[dim];
      }

      return static_cast<OpT>(lin_idx);
    }

    template <typename OpT>
    NCA_HD inline OpT apply_op(OpT res, OpT leaf, OpCode op) const {
      // --- Unary ops --- //
      if (op == OpCode::NEG) {
        return static_cast<OpT>(op_traits<OpT>::neg(res));
      } else if (op == OpCode::INC) {
        return static_cast<OpT>(op_traits<OpT>::inc(res));
      } else if (op == OpCode::DEC) {
        return static_cast<OpT>(op_traits<OpT>::dec(res));
      // Skipping SZOF, ADDR, INDR, CAST, LNOT, BNOT
      // --- Binary ops --- //
      // Arithmetic
      } else if (op == OpCode::ADD) {
        return static_cast<OpT>(res + leaf);
      } else if (op == OpCode::SUB) {
        return static_cast<OpT>(res - leaf);
      } else if (op == OpCode::MUL) {
        if constexpr (hd_std::is_same_v<OpT, bool>) {
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
      } else if (op == OpCode::MOD) {
        return static_cast<OpT>(op_traits<OpT>::mod(res, leaf));
      // Skipping FDIV
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
      // Logical
      } else if (op == OpCode::LAND) {
        return static_cast<OpT>(op_traits<OpT>::land(res, leaf));
      } else if (op == OpCode::LOR) {
        return static_cast<OpT>(op_traits<OpT>::lor(res, leaf));
      }
      // Skipping BAND, BOR, XOR, LSHFT, RSHFT
      return OpT{};
    }
  };
} // namespace ncarray

#endif // NCARRAY_EXPRESSION_INTERFACE_HH
