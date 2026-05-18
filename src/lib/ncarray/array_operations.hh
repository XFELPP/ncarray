/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ARRAY_OPERATIONS_HH
#define NCARRAY_ARRAY_OPERATIONS_HH

#include "ncarray/array_impl.hh"
#include "ncarray/engines.hh"
#include "ncarray/expression.hh"
#include "ncarray/host/casts.hh"
#include "ncarray/op_code.hh"
#include "ncarray/op_traits.hh"
#include "ncarray/mvnode.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cmath>
#include <complex>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ncarray {
  class index_error : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
  };

  // --- Expression Evaluation/Materialization --- //

  template <class L, class S>
  template <Expression Expr>
  ArrayImpl<L, S>& ArrayImpl<L, S>::operator=(const Expr& expr) {
    auto view = this->view();

    auto op = [&]<typename DestT>() {
      using MemType = typename std::decay_t<ArrayImpl<L, S>>::MemType;
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_binary_expression<DestT>(expr, view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_binary_expression<DestT>(expr, view);
      }
    };

    dispatch(this->dtype(), op);

    return *this;
  }

  // --- Generators --- //
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::iota() const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node(this->view());
    node.instrs.clear(); // Clear the load of the array, we only want the IDX values
    node.layouts.clear();
    node.data.clear();
    node.dtypes.clear(); // The node tracks expr_dtype separately than this vector
    node.instrs.push_back(pack_instruction(OpCode::IDX, 0));
    return node;
  }

  // --- Unary operations --- //
  // Negation
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator-() const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return -node;
  }

  // Increment
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator++() {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    auto inc_expr = ++node;
    auto op = [&] <typename DestT> () {
      using MemType = typename std::decay_t<ArrayImpl<L, S>>::MemType;
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_binary_expression<DestT>(inc_expr, view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_binary_expression<DestT>(inc_expr, view);
      }
    };

    dispatch(this->dtype(), op);
    return *this;
  }

  // Decrement
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator--() {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    auto dec_expr = --node;
    auto op = [&]<typename DestT>() {
      using MemType = typename std::decay_t<ArrayImpl<L, S>>::MemType;
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_binary_expression<DestT>(dec_expr, view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_binary_expression<DestT>(dec_expr, view);
      }
    };

    dispatch(this->dtype(), op);
    return *this;
  }

  // Logical not
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator!() const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return !node;
  }

  // --- Binary Operations --- //

  // Addition
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator+(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node + right;
  }

  // Subtraction
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator-(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node - right;
  }

  // Multiplication
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator*(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node * right;
  }

  // True division
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator/(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node / right;
  }

  // Modulo
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator%(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node % right;
  }

  // --- Inplace Binary Operations --- //

  template <class L, class S>
  inline ArrayImpl<L, S>&
  ArrayImpl<L, S>::operator+=(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    auto add_expr = node + right;
    auto op = [&]<typename DestT>() {
      using MemType = typename std::decay_t<ArrayImpl<L, S>>::MemType;
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_binary_expression<DestT>(add_expr, view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_binary_expression<DestT>(add_expr, view);
      }
    };

    dispatch(this->dtype(), op);
    return *this;
  }

  // Subtraction
  template <class L, class S>
  inline ArrayImpl<L, S>&
  ArrayImpl<L, S>::operator-=(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    auto sub_expr = node - right;
    auto op = [&]<typename DestT>() {
      using MemType = typename std::decay_t<ArrayImpl<L, S>>::MemType;
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_binary_expression<DestT>(sub_expr, view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_binary_expression<DestT>(sub_expr, view);
      }
    };

    dispatch(this->dtype(), op);
    return *this;
  }

  // Multiplication
  template <class L, class S>
  inline ArrayImpl<L, S>&
  ArrayImpl<L, S>::operator*=(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    auto mul_expr = node * right;
    auto op = [&]<typename DestT>() {
      using MemType = typename std::decay_t<ArrayImpl<L, S>>::MemType;
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_binary_expression<DestT>(mul_expr, view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_binary_expression<DestT>(mul_expr, view);
      }
    };

    dispatch(this->dtype(), op);
    return *this;
  }

  // True division
  template <class L, class S>
  inline ArrayImpl<L, S>&
  ArrayImpl<L, S>::operator/=(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    auto truediv_expr = node / right;
    auto op = [&]<typename DestT>() {
      using MemType = typename std::decay_t<ArrayImpl<L, S>>::MemType;
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_binary_expression<DestT>(truediv_expr, view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_binary_expression<DestT>(truediv_expr, view);
      }
    };

    dispatch(this->dtype(), op);
    return *this;
  }

  // --- Comparisons --- //

  // Is Equal
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator==(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node == right;
  }

  // Not equal
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator!=(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node != right;
  }

  // Less than
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator<(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node < right;
  }

  // Less than or equal
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator<=(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node <= right;
  }

  // Greater than
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator>(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node > right;
  }

  // Greater than or equal
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator>=(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node >= right;
  }

  // --- Logical Operations --- //

  // Logical and
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator&&(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node && right;
  }

  // Logical or
  template <class L, class S>
  inline ExprMVNode<typename ArrayImpl<L, S>::MemType>
  ArrayImpl<L, S>::operator||(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) const {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    return node || right;
  }

  // --- Logical Inplace Operations --- //

  // Logical and
  template <class L, class S>
  inline ArrayImpl<L, S>&
  ArrayImpl<L, S>::operator&=(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    auto and_expr = node && right;

    auto op = [&] <typename DestT> () {
      using MemType = typename std::decay_t<ArrayImpl<L, S>>::MemType;
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_binary_expression<DestT>(and_expr, view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_binary_expression<DestT>(and_expr, view);
      }
    };

    dispatch(this->dtype(), op);
    return *this;
  }

  // Logical or
  template <class L, class S>
  inline ArrayImpl<L, S>&
  ArrayImpl<L, S>::operator|=(const ExprMVNode<typename ArrayImpl<L, S>::MemType>& right) {
    using MemType = typename ArrayImpl<L, S>::MemType;

    ExprMVNode<MemType> node;
    auto view = this->view();
    node.build_node(view);

    auto or_expr = node || right;

    auto op = [&] <typename DestT> () {
      using MemType = typename std::decay_t<ArrayImpl<L, S>>::MemType;
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_binary_expression<DestT>(or_expr, view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_binary_expression<DestT>(or_expr, view);
      }
    };

    dispatch(this->dtype(), op);
    return *this;
  }

  // --- Axis-Aware Reductions --- //

  // sum
  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::sum(const std::vector<ssize_t>& axes) const {
    ssize_t new_shape[NCARRAY_MAX_NDIM];
    ssize_t new_ndim { 0 };
    ReductionParams params = build_reduction_params(axes,
                                                    this->ndim(),
                                                    this->shape(),
                                                    this->strides(),
                                                    new_shape,
                                                    new_ndim,
                                                    this->itemsize());

    auto dtype_op = [] <typename T> () {
      using AccumT = typename op_traits<T>::sum_type;
      return dtype_traits<AccumT>::value;
    };
    DType result_dtype = dispatch(this->dtype(), dtype_op);

    OwnerType result(new_ndim, new_shape, result_dtype);

    auto start_view = this->view();
    auto res_view = result.view();

    auto sum_op = [&] <typename SrcT> () {
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_reduce_axes<SrcT, SumTraits>(start_view, params, res_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_reduce_axes<SrcT, SumTraits>(start_view, params, res_view);
      }
    };

    dispatch(this->dtype(), sum_op);

    return result;
  }

  // max and argmax
  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::max(const std::vector<ssize_t>& axes) const {
    ssize_t new_shape[NCARRAY_MAX_NDIM];
    ssize_t new_ndim { 0 };
    ReductionParams params = build_reduction_params(axes,
                                                    this->ndim(),
                                                    this->shape(),
                                                    this->strides(),
                                                    new_shape,
                                                    new_ndim,
                                                    this->itemsize());

    DType result_dtype = this->dtype();

    OwnerType result(new_ndim, new_shape, result_dtype);

    auto start_view = this->view();
    auto res_view = result.view();

    auto max_op = [&] <typename SrcT> () {
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_reduce_axes<SrcT, MaxTraits>(start_view, params, res_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_reduce_axes<SrcT, MaxTraits>(start_view, params, res_view);
      }
    };

    dispatch(this->dtype(), max_op);

    return result;
  }

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::argmax(const std::vector<ssize_t>& axes) const {
    ssize_t new_shape[NCARRAY_MAX_NDIM];
    ssize_t new_ndim { 0 };
    ReductionParams params = build_reduction_params(axes,
                                                    this->ndim(),
                                                    this->shape(),
                                                    this->strides(),
                                                    new_shape,
                                                    new_ndim,
                                                    this->itemsize());

    DType result_dtype = dtype_traits<std::int64_t>::value;

    OwnerType result(new_ndim, new_shape, result_dtype);

    auto start_view = this->view();
    auto res_view = result.view();

    auto argmax_op = [&] <typename SrcT> () {
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_reduce_axes<SrcT, ArgmaxTraits>(start_view, params, res_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_reduce_axes<SrcT, ArgmaxTraits>(start_view, params, res_view);
      }
    };

    dispatch(this->dtype(), argmax_op);

    return result;
  }

  // min and argmin
  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::min(const std::vector<ssize_t>& axes) const {
    ssize_t new_shape[NCARRAY_MAX_NDIM];
    ssize_t new_ndim { 0 };
    ReductionParams params = build_reduction_params(axes,
                                                    this->ndim(),
                                                    this->shape(),
                                                    this->strides(),
                                                    new_shape,
                                                    new_ndim,
                                                    this->itemsize());

    DType result_dtype = this->dtype();

    OwnerType result(new_ndim, new_shape, result_dtype);

    auto start_view = this->view();
    auto res_view = result.view();

    auto min_op = [&] <typename SrcT> () {
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_reduce_axes<SrcT, MinTraits>(start_view, params, res_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_reduce_axes<SrcT, MinTraits>(start_view, params, res_view);
      }
    };

    dispatch(this->dtype(), min_op);

    return result;
  }

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::argmin(const std::vector<ssize_t>& axes) const {
    ssize_t new_shape[NCARRAY_MAX_NDIM];
    ssize_t new_ndim { 0 };
    ReductionParams params = build_reduction_params(axes,
                                                    this->ndim(),
                                                    this->shape(),
                                                    this->strides(),
                                                    new_shape,
                                                    new_ndim,
                                                    this->itemsize());

    DType result_dtype = dtype_traits<std::int64_t>::value;

    OwnerType result(new_ndim, new_shape, result_dtype);

    auto start_view = this->view();
    auto res_view = result.view();

    auto argmin_op = [&]<typename SrcT>() {
      if constexpr (std::is_same_v<MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_reduce_axes<SrcT, ArgminTraits>(start_view, params, res_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_reduce_axes<SrcT, ArgminTraits>(start_view, params, res_view);
      }
    };

    dispatch(this->dtype(), argmin_op);

    return result;
  }

  // mean, variance and standard deviation

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::mean(const std::vector<ssize_t>& axes) const {
    ssize_t new_shape[NCARRAY_MAX_NDIM];
    ssize_t new_ndim { 0 };
    ReductionParams params = build_reduction_params(axes,
                                                    this->ndim(),
                                                    this->shape(),
                                                    this->strides(),
                                                    new_shape,
                                                    new_ndim,
                                                    this->itemsize());

    auto dtype_op = []<typename T>() {
      using ResultT = typename op_traits<T>::truediv_type;
      return dtype_traits<ResultT>::value;
    };

    DType result_dtype = dispatch(this->dtype(), dtype_op);

    OwnerType result(new_ndim, new_shape, result_dtype);

    auto start_view = this->view();
    auto res_view = result.view();

    auto mean_operation = [&] <typename SrcT> () {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_reduce_axes<SrcT, MeanTraits>(start_view, params, res_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_reduce_axes<SrcT, MeanTraits>(start_view, params, res_view);
      }
    };

    dispatch(this->dtype(), mean_operation);

    return result;
  }

  // TODO: Make ddof work
  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::var(const std::vector<ssize_t>& axes, ssize_t ddof) const {
    ssize_t new_shape[NCARRAY_MAX_NDIM];
    ssize_t new_ndim { 0 };
    ReductionParams params = build_reduction_params(axes,
                                                    this->ndim(),
                                                    this->shape(),
                                                    this->strides(),
                                                    new_shape,
                                                    new_ndim,
                                                    this->itemsize());
    auto dtype_op = []<typename T>() {
      using ResultT = typename op_traits<T>::truediv_type;
      return dtype_traits<ResultT>::value;
    };

    DType result_dtype = dispatch(this->dtype(), dtype_op);

    OwnerType result(new_ndim, new_shape, result_dtype);

    auto start_view = this->view();
    auto res_view = result.view();

    auto var_operation = [&] <typename SrcT> () {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_reduce_axes<SrcT, VarTraits>(start_view, params, res_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_reduce_axes<SrcT, VarTraits>(start_view, params, res_view);
      }
    };

    dispatch(this->dtype(), var_operation);

    return result;
  }
  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::std(const std::vector<ssize_t>& axes, ssize_t ddof) const {
    ssize_t new_shape[NCARRAY_MAX_NDIM];
    ssize_t new_ndim { 0 };
    ReductionParams params = build_reduction_params(axes,
                                                    this->ndim(),
                                                    this->shape(),
                                                    this->strides(),
                                                    new_shape,
                                                    new_ndim,
                                                    this->itemsize());
    auto dtype_op = []<typename T>() {
      using ResultT = typename op_traits<T>::truediv_type;
      return dtype_traits<ResultT>::value;
    };

    DType result_dtype = dispatch(this->dtype(), dtype_op);

    OwnerType result(new_ndim, new_shape, result_dtype);

    auto start_view = this->view();
    auto res_view = result.view();

    auto std_operation = [&] <typename SrcT> () {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_reduce_axes<SrcT, StdTraits>(start_view, params, res_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_reduce_axes<SrcT, StdTraits>(start_view, params, res_view);
      }
    };

    dispatch(this->dtype(), std_operation);

    return result;
  }

  // Logical all and any

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::all(const std::vector<ssize_t>& axes) const {
    ssize_t new_shape[NCARRAY_MAX_NDIM];
    ssize_t new_ndim { 0 };
    ReductionParams params = build_reduction_params(axes,
                                                    this->ndim(),
                                                    this->shape(),
                                                    this->strides(),
                                                    new_shape,
                                                    new_ndim,
                                                    this->itemsize());

    OwnerType result(new_ndim, new_shape, dtype_traits<bool>::value);

    auto start_view = this->view();
    auto res_view = result.view();

    auto all_operation = [&] <typename SrcT> () {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_reduce_axes<SrcT, AllTraits>(start_view, params, res_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_reduce_axes<SrcT, AllTraits>(start_view, params, res_view);
      }
    };

    dispatch(this->dtype(), all_operation);

    return result;
  }
  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::any(const std::vector<ssize_t>& axes) const {
    ssize_t new_shape[NCARRAY_MAX_NDIM];
    ssize_t new_ndim { 0 };
    ReductionParams params = build_reduction_params(axes,
                                                    this->ndim(),
                                                    this->shape(),
                                                    this->strides(),
                                                    new_shape,
                                                    new_ndim,
                                                    this->itemsize());

    OwnerType result(new_ndim, new_shape, dtype_traits<bool>::value);

    auto start_view = this->view();
    auto res_view = result.view();

    auto any_operation = [&] <typename SrcT> () {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_reduce_axes<SrcT, AnyTraits>(start_view, params, res_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_reduce_axes<SrcT, AnyTraits>(start_view, params, res_view);
      }
    };

    dispatch(this->dtype(), any_operation);

    return result;
  }

  // --- Full Reductions (To Scalar) --- //

  template <class L, class S>
  Scalar ArrayImpl<L, S>::sum() const {
    auto arr_view = this->view();

    auto sum_operation = [&] <typename SrcT> () -> Scalar {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_full_reduce<SrcT, SumTraits>(arr_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_sum<SrcT>(arr_view);
      }
    };

    return dispatch(this->dtype(), sum_operation);
  }

  // max and argmax
  template <class L, class S>
  Scalar ArrayImpl<L, S>::max() const {
    auto arr_view = this->view();

    auto max_operation = [&] <typename SrcT> () -> Scalar {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_full_reduce<SrcT, MaxTraits>(arr_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_max<SrcT>(arr_view);
      }
    };

    return dispatch(this->dtype(), max_operation);
  }
  template <class L, class S>
  Scalar ArrayImpl<L, S>::argmax() const {
    auto arr_view = this->view();

    auto argmax_operation = [&] <typename SrcT> () -> Scalar {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_full_reduce<SrcT, ArgmaxTraits>(arr_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_argmax<SrcT>(arr_view);
      }
    };

    return dispatch(this->dtype(), argmax_operation);
  }

  // min and argmin
  template <class L, class S>
  Scalar ArrayImpl<L, S>::min() const {
    auto arr_view = this->view();

    auto min_operation = [&] <typename SrcT> () -> Scalar {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_full_reduce<SrcT, MinTraits>(arr_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_min<SrcT>(arr_view);
      }
    };

    return dispatch(this->dtype(), min_operation);
  }
  template <class L, class S>
  Scalar ArrayImpl<L, S>::argmin() const {
    auto arr_view = this->view();

    auto argmin_operation = [&] <typename SrcT> () -> Scalar {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_full_reduce<SrcT, ArgminTraits>(arr_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_argmin<SrcT>(arr_view);
      }
    };

    return dispatch(this->dtype(), argmin_operation);
  }

  // mean, variance and standard deviation

  template <class L, class S>
  Scalar ArrayImpl<L, S>::mean() const {
    auto arr_view = this->view();

    auto mean_operation = [&] <typename SrcT> () -> Scalar {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_full_reduce<SrcT, MeanTraits>(arr_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_mean<SrcT>(arr_view);
      }
    };

    return dispatch(this->dtype(), mean_operation);
  }
  template <class L, class S>
  Scalar ArrayImpl<L, S>::var(ssize_t ddof) const {
    auto arr_view = this->view();

    auto var_operation = [&] <typename SrcT> () -> Scalar {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_full_reduce<SrcT, VarTraits>(arr_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_var<SrcT>(arr_view, ddof);
      }
    };

    return dispatch(this->dtype(), var_operation);
  }
  template <class L, class S>
  Scalar ArrayImpl<L, S>::std(ssize_t ddof) const {
    auto arr_view = this->view();

    auto std_operation = [&] <typename SrcT> () -> Scalar {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_full_reduce<SrcT, StdTraits>(arr_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_std<SrcT>(arr_view, ddof);
      }
    };

    return dispatch(this->dtype(), std_operation);
  }

  // logical reduction ops - all and any

  template <class L, class S>
  Scalar ArrayImpl<L, S>::all() const {
    auto arr_view = this->view();
    auto all_operation = [&] <typename SrcT> () -> Scalar {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_full_reduce<SrcT, AllTraits>(arr_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_all<SrcT>(arr_view);
      }
    };

    return dispatch(this->dtype(), all_operation);
  }

  template <class L, class S>
  Scalar ArrayImpl<L, S>::any() const {
    auto arr_view = this->view();
    auto any_operation = [&] <typename SrcT> () -> Scalar {
      if constexpr (std::is_same_v<typename ArrayImpl<L, S>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_full_reduce<SrcT, AnyTraits>(arr_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_any<SrcT>(arr_view);
      }
    };

    return dispatch(this->dtype(), any_operation);
  }

  // --- Copy and Modification --- //

  template <class L, class S>
  void ArrayImpl<L, S>::copy_into(void* dest_buffer) const {
    // Convert to views to decrease binary size with fewer template instantiations
    auto arr_view = this->view();

    auto copy_op = [&]<typename T>() {
      T* dest = reinterpret_cast<T*>(dest_buffer);
      if constexpr (std::is_same_v<typename std::decay_t<ArrayImpl<L,S>>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_copy_into<T>(arr_view, dest);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_copy_into<T>(arr_view, dest);
      }
    };

    dispatch(this->dtype(), copy_op);
  }

  template <class L, class S>
  template <typename OutT>
  void ArrayImpl<L, S>::copy_into_astype(OutT* dest_buffer) const {
    auto arr_view = this->view();

    auto copy_op = [&] <typename SrcT> () {
      if constexpr (std::is_same_v<typename std::decay_t<ArrayImpl<L,S>>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_copy_into<SrcT>(arr_view, dest_buffer);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_copy_into<SrcT>(arr_view, dest_buffer);
      }
    };

    dispatch(this->dtype(), copy_op);
  }

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::to_contiguous() const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    OwnerType result(this->m_shape, this->m_dtype);

    auto copy_op = [&]<typename T>() {
      T* dest_ptr = reinterpret_cast<T*>(result.data());
      this->copy_into_astype(dest_ptr);
    };

    dispatch(this->m_dtype, copy_op);

    return result;
  }

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::astype(DType& dtype_out) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    OwnerType result(this->m_shape, this->m_dtype);

    auto copy_op = [&]<typename OutT>() {
      OutT* dest_ptr = reinterpret_cast<OutT*>(result.data());
      this->copy_into_astype<OutT>(dest_ptr);
    };

    dispatch(dtype_out, copy_op);

    return result;
  }

  template <class L, class S>
  void ArrayImpl<L, S>::assign(const ArrayLike auto& arr) {
    if (this->m_read_only) {
      throw type_error("Cannot modify a read-only view!");
    }

    // Only deal with identical shapes for now
    if (this->ndim() != arr.ndim()) {
      throw type_error("Shapes must match for assignment");
    }
    for (ssize_t i = 0; i < this->ndim(); ++i) {
      if (this->shape(i) != arr.shape(i)) {
        throw type_error("Shapes must match for assignment");
      }
    }

    auto dest_view = this->view();
    auto src_view = arr.view();

    auto assign_op = [&]<typename DestT>() {
      // Dispatch dest type - the engines do the double dispatch to get the src type
      if constexpr (std::is_same_v<typename std::decay_t<ArrayImpl<L,S>>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_assign<DestT>(dest_view, src_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_assign<DestT>(dest_view, src_view);
      }
    };
    dispatch(this->dtype(), assign_op);
  }

  template <class L, class S>
  void ArrayImpl<L, S>::fill(Scalar val) {
    if (this->m_read_only) {
      throw type_error("Cannot modify a read-only view!");
    }

    // Convert to views to decrease binary size with fewer template instantiations
    auto arr_view = this->view();

    auto fill_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<ArrayImpl<L,S>>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_fill<T>(arr_view, val);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_fill<T>(arr_view, val);
      }
    };

    dispatch(this->dtype(), fill_op);
  }

} // namespace ncarray

#endif // NCARRAY_ARRAY_OPERATIONS_HH
