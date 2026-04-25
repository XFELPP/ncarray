/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_EXPRESSION_HH
#define NCARRAY_EXPRESSION_HH

#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"

#ifndef __CUDACC_RTC__
#ifdef __CUDACC__
#include "ncarray/device/casts.cuh"
#endif
#include "ncarray/dtype.hh"
#include "ncarray/host/casts.hh"
#include "ncarray/layout.hh"
#include "ncarray/op_code.hh"
#include "ncarray/op_traits.hh"

#include <cmath>

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cassert>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <type_traits>

using std::complex;
using std::false_type;
using std::is_base_of_v;
using std::true_type;

#else // only need a bit for NVRTC
#include <cuda/std/complex>
#include <cuda/std/type_traits>

using cuda::std::complex;
using cuda::std::false_type;
using cuda::std::is_base_of_v;
using cuda::std::true_type;
#endif // NVRTC guard

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
  concept Expression = is_base_of_v<ExpressionTag, T>;

  /**
   * Defines array-like expressions.
   */
  template <typename T>
  concept ArrayExpression = Expression<T> || ArrayLike<T>;

  template <typename T>
  struct is_std_complex : false_type {};

  template <typename T>
  struct is_std_complex<complex<T>> : true_type {};

  template <typename T>
  concept Complex = is_std_complex<T>::value;

  // Numeric comes from custom_types.hh (basically std::is_arithmetic_v)
  /**
   * Defines all supported types for use in expressions.
   *
   * This provides support for building expression trees beyond just a
   * a simple binary operation.
   */
  template <typename T>
  concept AnyExpression =
    Expression<T> || ArrayLike<T> || Numeric<T> || Complex<T> ||Vector2DType<T>;

#ifndef __CUDACC_RTC__
  template <class L, class R, OpCode O>
  struct BinaryExpression;

  template <class C>
  struct ExprArrayNode;

  template <typename T>
  struct ExprScalarNode;

  template <class Derived>
  struct ExprInterface {

    template <AnyExpression Node>
    NCA_HD inline auto build_node(Node& node) {
      if constexpr (ArrayLike<Node>) {
        using ViewType = typename Node::ViewType;

        return ExprArrayNode<ViewType>(node.view());
      } else if constexpr (Numeric<Node> || Vector2DType<Node>) {
        return ExprScalarNode<Node>(node);
      } else {
        return node;
      }
    }

    // Arithmetic

    template <AnyExpression RightNode>
    NCA_HD inline auto operator+(const RightNode& right) {
      return binary_operation<OpCode::ADD>(right);
    }

    template <AnyExpression RightNode>
    NCA_HD inline auto operator-(const RightNode& right) {
      return binary_operation<OpCode::SUB>(right);
    }

    template <AnyExpression RightNode>
    NCA_HD inline auto operator*(const RightNode& right) {
      return binary_operation<OpCode::MUL>(right);
    }

    template <AnyExpression RightNode>
    NCA_HD inline auto operator/(const RightNode& right) {
      return binary_operation<OpCode::DIV>(right);
    }

    // Comparisons

    template <AnyExpression RightNode>
    NCA_HD inline auto operator==(const RightNode& right) {
      return binary_operation<OpCode::EQ>(right);
    }

    template <AnyExpression RightNode>
    NCA_HD inline auto operator!=(const RightNode& right) {
      return binary_operation<OpCode::NE>(right);
    }

    template <AnyExpression RightNode>
    NCA_HD inline auto operator<(const RightNode& right) {
      return binary_operation<OpCode::LT>(right);
    }

    template <AnyExpression RightNode>
    NCA_HD inline auto operator<=(const RightNode& right) {
      return binary_operation<OpCode::LE>(right);
    }

    template <AnyExpression RightNode>
    NCA_HD inline auto operator>(const RightNode& right) {
      return binary_operation<OpCode::GT>(right);
    }

    template <AnyExpression RightNode>
    NCA_HD inline auto operator>=(const RightNode& right) {
      return binary_operation<OpCode::GE>(right);
    }

    // Logical

    template <AnyExpression RightNode>
    NCA_HD inline auto operator&&(const RightNode& right) {
      return binary_operation<OpCode::LAND>(right);
    }

    template <AnyExpression RightNode>
    NCA_HD inline auto operator||(const RightNode& right) {
      return binary_operation<OpCode::LOR>(right);
    }

  private:
    template <OpCode Op, AnyExpression RightNode>
    NCA_HD inline auto binary_operation(const RightNode& right) {
      const auto& self = static_cast<const Derived&>(*this);
      auto right_node = this->build_node(right);

      return BinaryExpression<
        Derived,
        std::decay_t<decltype(right_node)>,
        Op
      >(self, right_node);
    }
  };

  /**
   * The ExprArrayNode is an end-node of an operation.
   *
   * ExprArrayNodes (as of now) should by instances of ArrayImpl
   * specializations. For scalar operands, see the ScalarExpression.
   */
  template <class Array>
  struct ExprArrayNode
    : public ExpressionTag, public ExprInterface<ExprArrayNode<Array>> {

    const Array array;

    using MemType = typename Array::MemType;

    NCA_HD ExprArrayNode(const Array& c)
      : array(c)
    {}

    template <typename DestT>
    NCA_HD inline DestT eval(ssize_t idx) const {
      int src_idx { static_cast<int>(array.dtype()) };
      auto proxy = array[idx];
      const void* ptr { const_cast<const void*>(proxy.m_data) };

      if constexpr (std::is_same_v<MemType, HostTag>) {
#ifndef __CUDA_ARCH__
        return host::vm_cast_table<DestT>[src_idx](ptr);
#endif
      } else {
#ifdef __CUDACC__
        return device::device_cast<DestT>(src_idx, ptr);
#endif
      }
    }

    template <typename DestT>
    //NCA_HD inline DestT eval(ssize_t (&coords)[NCARRAY_MAX_NDIM]) const {
    NCA_HD inline DestT eval(ssize_t* coords) const {
      int src_idx { static_cast<int>(array.dtype()) };
      auto proxy = array[coords];
      const void* ptr { const_cast<const void*>(proxy.m_data) };

      if constexpr (std::is_same_v<MemType, HostTag>) {
#ifndef __CUDA_ARCH__
        return host::vm_cast_table<DestT>[src_idx](ptr);
#endif
      } else {
#ifdef __CUDACC__
        return device::device_cast<DestT>(src_idx, ptr);
#endif
      }
    }

    template <typename DestT, typename Coords>
    NCA_HD inline DestT eval(const Coords& coords) const {
      int src_idx { static_cast<int>(array.dtype()) };
      auto proxy = array[coords];
      const void* ptr{const_cast<const void*>(proxy.m_data)};

      if constexpr (std::is_same_v<MemType, HostTag>) {
#ifndef __CUDA_ARCH__
        return host::vm_cast_table<DestT>[src_idx](ptr);
#endif
      } else {
#ifdef __CUDACC__
        return device::device_cast<DestT>(src_idx, ptr);
#endif
      }
    }

    NCA_HD inline ssize_t size() const {
      return array.size();
    }

    NCA_HD inline DType dtype() const {
      return array.dtype();
    }

    NCA_HD inline const ssize_t* shape() const {
      return array.shape();
    }

    NCA_HD inline ssize_t ndim() const {
      return array.ndim();
    }
  };

  template <typename T, class Array>
  struct ExprTypedArrayNode
    : public ExpressionTag, public ExprInterface<ExprTypedArrayNode<T, Array>> {

    const Array array;

    NCA_HD ExprTypedArrayNode(const Array& c)
      : array(c)
    {}

    template <typename DestT>
    NCA_HD inline DestT eval(ssize_t idx) const {
      T& item = array[idx];
      return op_traits<T>::template cast<DestT>(item);
    }

    template <typename DestT>
    //NCA_HD inline DestT eval(ssize_t (&coords)[NCARRAY_MAX_NDIM]) const {
    NCA_HD inline DestT eval(ssize_t* coords) const {
      T& item = array[coords];
      return op_traits<T>::template cast<DestT>(item);
    }

    template <typename DestT, typename Coords>
    NCA_HD inline DestT eval(const Coords& coords) const {
      T& item = array[coords];
      return op_traits<T>::template cast<DestT>(item);
    }

    NCA_HD inline ssize_t size() const {
      return array.size();
    }

    NCA_HD inline DType dtype() const {
      return array.dtype();
    }

    NCA_HD inline const ssize_t* shape() const {
      return array.shape();
    }

    NCA_HD inline ssize_t ndim() const {
      return array.ndim();
    }
  };

  /**
   * A terminal node in an expression tree holding a single scalar value.
   */
  template <typename T>
  struct ExprScalarNode
    : public ExpressionTag, public ExprInterface<ExprScalarNode<T>> {

    using value_type = T;
    const T val;

    NCA_HD ExprScalarNode(const T& val_)
      : val(val_)
    {}

    template <typename DestT>
    NCA_HD inline DestT eval(ssize_t) const {
      return op_traits<T>::template cast<DestT>(val);
    }

    template <typename DestT>
    //NCA_HD inline DestT eval(ssize_t (&coords)[NCARRAY_MAX_NDIM]) const {
    NCA_HD inline DestT eval(ssize_t* coords) const {
      return op_traits<T>::template cast<DestT>(val);
    }

    template <typename DestT, typename Coords>
    NCA_HD inline DestT eval(const Coords& coords) const {
      return op_traits<T>::template cast<DestT>(val);
    }

    NCA_HD inline ssize_t size() const { return 0; }

    NCA_HD inline DType dtype() const { return dtype_traits<T>::value; }
  };

  template <class Array, OpCode Op>
  struct UnaryExpression
    : public ExpressionTag, public ExprInterface<UnaryExpression<Array, Op>> {

    const Array array;

    NCA_HD UnaryExpression(const Array& array_)
      : array(array_)
    {}

    template <typename DestT>
    NCA_HD inline DestT eval(ssize_t idx) const {
      if constexpr (Op == OpCode::NEG) {
        return -(array.template eval<DestT>(idx));
      } else if constexpr (Op == OpCode::LNOT) {
        return !(array.template eval<DestT>(idx));
      } else if constexpr (Op == OpCode::BNOT) {
        return ~(array.template eval<DestT>(idx));
      }
    }

    NCA_HD inline DType dtype() const {
      return determine_dtype_for_op(Op, array.dtype());
    }
  };


  template <class Left, class Right, OpCode Op>
  struct BinaryExpression
    : public ExpressionTag, public ExprInterface<BinaryExpression<Left, Right, Op>> {

    const Left left;
    const Right right;

    NCA_HD BinaryExpression(const Left& l, const Right& r)
      : left(l)
      , right(r)
    {}

    template <typename DestT>
    NCA_HD inline DestT eval(ssize_t idx) const {
      if constexpr (Op == OpCode::ADD) {
        return left.template eval<DestT>(idx) + right.template eval<DestT>(idx);
      } else if constexpr (Op == OpCode::SUB) {
        return left.template eval<DestT>(idx) - right.template eval<DestT>(idx);
      } else if constexpr (Op == OpCode::MUL) {
        if constexpr (std::is_same_v<DestT, bool>) {
          return left.template eval<DestT>(idx) && right.template eval<DestT>(idx);
        } else {
          return left.template eval<DestT>(idx) * right.template eval<DestT>(idx);
        }
      } else if constexpr (Op == OpCode::DIV) {
        DestT lhs = left.template eval<DestT>(idx);
        DestT rhs = right.template eval<DestT>(idx);

        using std::isfinite;
        bool is_finite { false };

        if (rhs == DestT(0)) {
          if constexpr (requires { lhs.real(); }) {
            is_finite = isfinite(lhs.real()) && isfinite(lhs.imag());
          } else {
            is_finite = isfinite(lhs);
          }
          return is_finite ? std::nan("") : lhs;
        }
        return lhs / rhs;
        // Comparisons
      } else if constexpr (Op == OpCode::EQ) {
        return left.template eval<DestT>(idx) == right.template eval<DestT>(idx);
      } else if constexpr (Op == OpCode::NE) {
        return left.template eval<DestT>(idx) != right.template eval<DestT>(idx);
      } else if constexpr (Op == OpCode::LT) {
        return left.template eval<DestT>(idx) < right.template eval<DestT>(idx);
      } else if constexpr (Op == OpCode::LE) {
        return left.template eval<DestT>(idx) <= right.template eval<DestT>(idx);
      } else if constexpr (Op == OpCode::GT) {
        return left.template eval<DestT>(idx) > right.template eval<DestT>(idx);
      } else if constexpr (Op == OpCode::GE) {
        return left.template eval<DestT>(idx) >= right.template eval<DestT>(idx);
        // Logical
      } else if constexpr (Op == OpCode::LAND) {
        return left.template eval<DestT>(idx) && right.template eval<DestT>(idx);
      } else if constexpr (Op == OpCode::LOR) {
        return left.template eval<DestT>(idx) || right.template eval<DestT>(idx);
      }
      return DestT{};
    }

    //template <typename DestT>
    //NCA_HD inline DestT eval(ssize_t (&coords)[NCARRAY_MAX_NDIM]) const {
    template <typename DestT, typename Coords>
    NCA_HD inline DestT eval(const Coords& coords) const {
      if constexpr (Op == OpCode::ADD) {
        return left.template eval<DestT>(coords) + right.template eval<DestT>(coords);
      } else if constexpr (Op == OpCode::SUB) {
        return left.template eval<DestT>(coords) - right.template eval<DestT>(coords);
      } else if constexpr (Op == OpCode::MUL) {
        if constexpr (std::is_same_v<DestT, bool>) {
          return left.template eval<DestT>(coords) && right.template eval<DestT>(coords);
        } else {
          return left.template eval<DestT>(coords) * right.template eval<DestT>(coords);
        }
      } else if constexpr (Op == OpCode::DIV) {
        DestT lhs = left.template eval<DestT>(coords);
        DestT rhs = right.template eval<DestT>(coords);

        using std::isfinite;
        bool is_finite { false };

        if (rhs == DestT(0)) {
          if constexpr (requires { lhs.real(); }) {
            is_finite = isfinite(lhs.real()) && isfinite(lhs.imag());
          } else {
            is_finite = isfinite(lhs);
          }
          return is_finite ? std::nan("") : lhs;
        }
        return lhs / rhs;
        // Comparisons
      } else if constexpr (Op == OpCode::EQ) {
        return left.template eval<DestT>(coords) == right.template eval<DestT>(coords);
      } else if constexpr (Op == OpCode::NE) {
        return left.template eval<DestT>(coords) != right.template eval<DestT>(coords);
      } else if constexpr (Op == OpCode::LT) {
        return left.template eval<DestT>(coords) < right.template eval<DestT>(coords);
      } else if constexpr (Op == OpCode::LE) {
        return left.template eval<DestT>(coords) <= right.template eval<DestT>(coords);
      } else if constexpr (Op == OpCode::GT) {
        return left.template eval<DestT>(coords) > right.template eval<DestT>(coords);
      } else if constexpr (Op == OpCode::GE) {
        return left.template eval<DestT>(coords) >= right.template eval<DestT>(coords);
        // Logical
      } else if constexpr (Op == OpCode::LAND) {
        return left.template eval<DestT>(coords) && right.template eval<DestT>(coords);
      } else if constexpr (Op == OpCode::LOR) {
        return left.template eval<DestT>(coords) || right.template eval<DestT>(coords);
      }
      return DestT{};
    }


    NCA_HD inline ssize_t size() const {
      ssize_t l_size = left.size();
      return l_size > 0 ? l_size : right.size();
    }

    NCA_HD inline DType dtype() const {
      return determine_dtype_for_op(Op, left.dtype());
    }

    NCA_HD inline const ssize_t* shape() const {
      return left.shape();
    }
    NCA_HD inline ssize_t ndim() const {
      return left.ndim();
    }
  };

  // --- Additional Traits and Concepts --- //

  // Traits and checks on array nodes

  template <class Expr>
  struct is_array_node : std::false_type {};

  template <class Expr>
  struct is_typed_array_node : std::false_type {};

  template <class Arr>
  struct is_array_node<ExprArrayNode<Arr>> : std::true_type {};

  template <typename T, class Arr>
  struct is_typed_array_node<ExprTypedArrayNode<T, Arr>> : std::true_type {};

  template <class Expr>
  constexpr bool is_array_node_v = is_array_node<Expr>::value;

  template <class Expr>
  constexpr bool is_typed_array_node_v = is_typed_array_node<Expr>::value;

  // Traits and checks on scalar nodes

  template <class Expr>
  struct is_scalar_node : std::false_type {};

  template <typename T>
  struct is_scalar_node<ExprScalarNode<T>> : std::true_type {};

  template <class Expr>
  constexpr bool is_scalar_node_v = is_scalar_node<Expr>::value;

  // Traits, concepts and checks on binary expressions

  template <class Expr, class L, class R, OpCode Op>
  concept BinaryExprNode = std::is_base_of_v<Expr, BinaryExpression<L, R, Op>>;

  template <class Expr>
  struct is_binary_node : std::false_type {};

  template <class L, class R, OpCode O>
  struct is_binary_node<BinaryExpression<L, R, O>> : std::true_type {};

  template <class Expr>
  constexpr bool is_binary_node_v = is_binary_node<Expr>::value;

  template <typename Expr>
  struct op_code_of;

  template <class L, class R, OpCode Op>
  struct op_code_of<BinaryExpression<L, R, Op>> {
    static constexpr OpCode value = Op;
  };

  // Traversal and counting

  template <typename Expr>
  struct count_leaves {
    static constexpr int value = 1;
  };

  template <class L, class R, OpCode O>
  struct count_leaves<BinaryExpression<L, R, O>> {
    static constexpr int value = count_leaves<L>::value + count_leaves<R>::value;
  };

  template <typename Expr>
  constexpr int count_leaves_v = count_leaves<Expr>::value;

  // --- Collection Nodes --- //

  /**
   * Traverse an expression tree and get the first array node.
   */
  template <typename Expr>
  const auto& get_ref_layout(const Expr& expr) {
    if constexpr (is_binary_node_v<Expr>) {
      return get_ref_layout(expr.left);
    } else if constexpr (is_array_node_v<Expr> ||
                         is_typed_array_node_v<Expr>){
      return expr.array;
    } else {
      return expr;
    }
  }

  template <typename Expr>
  void collect_mv_data(const Expr& expr,
                       const void** data_out,
                       DType* dtype_out,
                       OpCode* ops_out,
                       int& leaf_idx,
                       int& op_idx) {

    if constexpr (is_binary_node_v<Expr>) {
      // Traverse Left
      collect_mv_data(expr.left, data_out, dtype_out, ops_out, leaf_idx, op_idx);

      // Store the Operation that merges this branch
      ops_out[op_idx++] = op_code_of<Expr>::value;

      // Traverse Right
      collect_mv_data(expr.right, data_out, dtype_out, ops_out, leaf_idx, op_idx);
    } else {
      // Leaf: store the pointers
      if constexpr (requires { expr.arr_views; }) {
        data_out[leaf_idx] = expr.arr_views[leaf_idx].data();
        dtype_out[leaf_idx] = expr.dtype();
        leaf_idx++;
      } else {
        data_out[leaf_idx] = expr.array.data();
        dtype_out[leaf_idx] = expr.array.dtype();
        leaf_idx++;
      }
    }
  }

  template <typename Expr>
  bool check_layouts(const Expr& expr, const auto& ref_layout) {
    if (std::is_same_v<decltype(ref_layout), SOArrayPolicy>) {
      return false;
    }
    if constexpr (is_binary_node_v<Expr>) {
      return
        check_layouts(expr.left, ref_layout) && check_layouts(expr.right, ref_layout);
    } else {
      if constexpr (requires { expr.arr_views; }) {
        return false;
        //return expr.arr_views[0].layout_matches(ref_layout);
      } else {
        //return expr.array.layout_matches(ref_layout);
        bool match = expr.array.layout_matches(ref_layout);
        if (!match) {
        }
        return match;
      }
    }
  }

  /**
   * Optimization for the Expression Tree
   * 1. Object Recognition
   * 2. View Packing
   * 3. Topological Deduplication (CSE - common subexpression elimination)
   *     - (a + b) * (a + b) uses (a + b) twice
   * 4. Algebraic pruning (identity removals)
   * 5. Stride pruning (broadcast optimizations)
   *     - E.g. zero strides convert to scalars
   * 6. DType narrowing where needed.
   */

  /**

template <typename Expr>
const auto& get_ref_layout(const Expr& expr) {
  if constexpr (is_binary_node_v<Expr>) return get_ref_layout(expr.left);
  else return expr.array; // Returns the first ArrayImpl found
}
// In your Engine:
auto& ref = get_ref_layout(expr);
if (check_layouts(expr, ref)) {
    // We can safely collapse into a single Layout object!
    // Now collect pointers, dtypes, and ops...
}
   */

  /**
   * The ExprMVArrayNode is a specialized tree node to optimize (portions)
   * of expression that may contain the same array viewed in different ways.
   * MV stands for Multi-View.
   *
   * E.g.: You add arr[sl(0,-1,2), sl(0,-1,2)] and arr[sl(1,-1,2), sl(1,-1,2)]. The
   * data for traversing this expression only needs to be stored once since the operands
   * are identical.
   */
  template <class Layout, int NViews, class MemType>
  struct ExprMVArrayNode
    : public ExpressionTag
    , public ExprInterface<ExprMVArrayNode<Layout, NViews, MemType>>
  {
    const Layout layout;
    DType dtypes[NViews];
    const void* data[NViews];
    std::ptrdiff_t offsets[NViews - 1];
    OpCode ops[NViews - 1];

    NCA_HD ExprMVArrayNode(const Layout& layout_,
                           const DType (&dtypes_)[NViews],
                           const void* (&data_)[NViews],
                           const OpCode (&ops_)[NViews - 1])
      : layout(layout_)
    {
      //const std::uint8_t* base { static_cast<const std::uint8_t*>(data) };
      for (int i = 0; i < NViews; ++i) {
        dtypes[i] = dtypes_[i];
        data[i] = data_[i];
        /*
        if (i > 0) {
          offsets[i - 1] = static_cast<const std::uint8_t*>(data_[i]) - base;
        }
        */
        if (i < NViews - 1) {
          ops[i] = ops_[i];
        }
      }
    }

    //template <typename DestT>
    //NCA_HD inline DestT eval(ssize_t (&coords)[NCARRAY_MAX_NDIM]) const {
    template <typename DestT, typename Coords>
    NCA_HD inline DestT eval(const Coords& coords) const {
      DestT res { 0 };

      #pragma unroll NViews
      for (int i = 0; i < NViews; ++i) {
        /*
        const void* ptr;
        if (i == 0) {
          ptr = data;
          //ptr = n_data;
        } else {
          ptr = data + offsets[i - 1];
          //ptr = n_data + offsets[i - 1];
        }
        const void* leaf_ptr = layout.advance(ptr, coords);
        */
        const void* leaf_ptr = layout.advance(data[i], coords);
        int src_idx { static_cast<int>(dtypes[i]) };

        DestT leaf_res;
        if constexpr (std::is_same_v<MemType, HostTag>) {
#ifndef __CUDA_ARCH__
          leaf_res = host::vm_cast_table<DestT>[src_idx](leaf_ptr);
#endif
        } else {
#ifdef __CUDACC__
          leaf_res = device::device_cast<DestT>(src_idx, leaf_ptr);
#endif
        }

        if (i == 0) {
          res = leaf_res;
        } else {
          res = this->template apply_op<DestT>(res, leaf_res, ops[i - 1]); // For N views, have N - 1 Ops
        }
      }
      return res;

      /*
      DestT vals[NViews];
      #pragma unroll NViews
      for (int i = 0; i < NViews; ++i) {
        const void* ptr;
        if (i == 0) {
          ptr = data;
        } else {
          ptr = data + offsets[i - 1];
        }
        const void* leaf_ptr = layout.advance(ptr, coords);
        int src_idx { static_cast<int>(dtypes[i]) };
        if constexpr (std::is_same_v<MemType, HostTag>) {
#ifndef __CUDA_ARCH__
          vals[i] = host::vm_cast_table<DestT>[src_idx](leaf_ptr);
#endif
        } else {
#ifdef __CUDACC__
          vals[i] = device::device_cast<DestT>(src_idx, leaf_ptr);
#endif
        }
      }

      DestT res { vals[0] };

      #pragma unroll NViews
      for (int i = 1; i < NViews; ++i) {
        res = this->template apply_op<DestT>(res, vals[i], ops[i - 1]);
      }
      return res;
      */
    }

    template <typename DestT>
    NCA_HD inline DestT eval(ssize_t idx) const {
      return DestT { 0 };
    }

    template <typename DestT>
    NCA_HD inline DestT apply_op(DestT res, DestT leaf, OpCode op) const {
      if (op == OpCode::ADD) {
        return res + leaf;
      } else if (op == OpCode::SUB) {
        return res - leaf;
      } else if (op == OpCode::MUL) {
        if constexpr (std::is_same_v<DestT, bool>) {
          return res && leaf;
        } else {
          return res * leaf;
        }
      } else if (op == OpCode::DIV) {
        using std::isfinite;
        bool is_finite { false };

        if (leaf == DestT(0)) {
          if constexpr (requires { res.real(); }) {
            is_finite = isfinite(res.real()) && isfinite(res.imag());
          } else {
            is_finite = isfinite(res);
          }
          return is_finite ? std::nan("") : res;
        }
        return res / leaf;
        // Comparisons
      }
      /*
      } else if  (op == OpCode::EQ) {
        return res == leaf;
      } else if  (op == OpCode::NE) {
        return res != leaf;
      } else if  (op == OpCode::LT) {
        return op_traits<DestT>::less(res, leaf);
      } else if  (op == OpCode::LE) {
        return op_traits<DestT>::le(res, leaf);
      } else if  (op == OpCode::GT) {
        return op_traits<DestT>::greater(res, leaf);
      } else if  (op == OpCode::GE) {
        return op_traits<DestT>::ge(res, leaf);
      }
      */
      return DestT{};
      /*
        // Logical
      } else if  (op == OpCode::LAND) {
        return res && leaf;
      } else if  (op == OpCode::LOR) {
        return res || leaf;
      }
      return DestT{};
      */
    }


    NCA_HD inline ssize_t size() const {
      return layout.size();
    }

    NCA_HD inline DType dtype() const {
      return dtypes[0];
    }

    NCA_HD inline const ssize_t* shape() const {
      return layout.shape();
    }

    NCA_HD inline ssize_t ndim() const {
      return layout.ndim();
    }
  };


  template <typename T, class Layout, int NViews, class MemType>
  struct ExprTypedMVArrayNode
    : public ExpressionTag
    , public ExprInterface<ExprTypedMVArrayNode<T, Layout, NViews, MemType>>
  {
    const Layout layout;
    const void* data[NViews];
    OpCode ops[NViews - 1];

    NCA_HD ExprTypedMVArrayNode(const Layout& layout_,
                                const DType dtype_,
                                const void* (&data_)[NViews],
                                const OpCode (&ops_)[NViews - 1])
      : layout(layout_)
    {
      for (int i = 0; i < NViews; ++i) {
        data[i] = data_[i];
        if (i < NViews - 1) {
          ops[i] = ops_[i];
        }
      }
    }

    template <typename DestT, typename Coords>
    NCA_HD inline DestT eval(const Coords& coords) const {

      DestT values[NViews];
      #pragma unroll NViews
      for (int i = 0; i < NViews; ++i) {
        const void* leaf_ptr = layout.advance(data[i], coords);
        T item = *reinterpret_cast<const T*>(leaf_ptr);
        values[i] = op_traits<T>::template cast<DestT>(item);
      }

      DestT res { 0 };
      #pragma unroll NViews
      for (int i = 0; i < NViews; ++i) {
        if (i == 0) {
          res = values[i];
        } else {
          res = this->template apply_op<DestT>(res, values[i], ops[i - 1]);
        }
      }

      return res;
      /*
      DestT res { 0 };

      #pragma unroll NViews
      for (int i = 0; i < NViews; ++i) {
        const void* leaf_ptr = layout.advance(data[i], coords);

        T item = *reinterpret_cast<const T*>(leaf_ptr);
        DestT leaf_res = op_traits<T>::template cast<DestT>(item);
        if (i == 0) {
          res = leaf_res;
        } else {
          // NOTE: For N views, we currently assume N - 1 Ops
          // This ignores unary ops on the first view.
          res = this->template apply_op<DestT>(res, leaf_res, ops[i - 1]);
        }
      }
      return res;
      */
    }

    template <typename DestT>
    NCA_HD inline DestT eval(ssize_t idx) const {
      return DestT { 0 };
    }

    template <typename DestT>
    NCA_HD inline DestT apply_op(DestT res, DestT leaf, OpCode op) const {
      if (op == OpCode::ADD) {
        return res + leaf;
      } else if (op == OpCode::SUB) {
        return res - leaf;
      } else if (op == OpCode::MUL) {
        if constexpr (std::is_same_v<DestT, bool>) {
          return res && leaf;
        } else {
          return res * leaf;
        }
      } else if (op == OpCode::DIV) {
        using std::isfinite;
        bool is_finite { false };

        if (leaf == DestT(0)) {
          if constexpr (requires { res.real(); }) {
            is_finite = isfinite(res.real()) && isfinite(res.imag());
          } else {
            is_finite = isfinite(res);
          }
          return is_finite ? std::nan("") : res;
        }
        return res / leaf;
        // Comparisons
      }
      return DestT{};
    }


    NCA_HD inline ssize_t size() const {
      return layout.size();
    }

    NCA_HD inline DType dtype() const {
      return dtype_traits<T>::value;
    }

    NCA_HD inline const ssize_t* shape() const {
      return layout.shape();
    }

    NCA_HD inline ssize_t ndim() const {
      return layout.ndim();
    }
  };

  // --- Lifting helpers --- //

  /**
   * BuildExprNode specialization to convert a scalar into a scalar node.
   */
  template <typename T>
  struct BuildExprNode {
    using type = ExprScalarNode<T>;
    static NCA_HD auto get(const T& val) { return ExprScalarNode<T>(val); }
  };

  /**
   * BuildExprNode fall-through helper for expressions. An existing expression just
   * returns itself.
   */
  template <Expression Expr>
  struct BuildExprNode<Expr> {
    using type = Expr;
    static NCA_HD auto get(const Expr& val) { return val; }
  };

  /**
   * BuildExprNode specialization to convert an Array to an array node.
   */
  template <ArrayLike Arr>
  struct BuildExprNode<Arr> {
    using type = ExprArrayNode<typename Arr::ViewType>;
    static NCA_HD auto get(const Arr& val) { return type(val.view()); }
  };
#endif // ifndef __CUDACC_RTC__ (nvrtc guard)
} // namespace ncarray

#endif // NCARRAY_EXPRESSION_HH
