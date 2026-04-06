/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ENGINES_HH
#define NCARRAY_ENGINES_HH

#include "ncarray/array_traits.hh"
#ifdef __CUDACC__
#include "ncarray/device/kernels.cuh"
#endif
#include "ncarray/device/mem_pool.cuh"
#include "ncarray/device/utilities.cuh"
#include "ncarray/host/elementwise.hh"
#include "ncarray/host/reductions.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <type_traits>

namespace ncarray {
  struct HostEngine {
    // --- Unary reductions --- //

    template <typename T, ArrayLike A>
    static Scalar execute_sum(const A& arr) {
      return host::sum_recursive<T>(arr);
    }

    template <typename T, ArrayLike A>
    static Scalar execute_mean(const A& arr) {
      return host::mean_recursive<T>(arr);
    }

    template <typename T, ArrayLike A>
    static Scalar execute_max(const A& arr) {
      return host::max_recursive<T>(arr);
    }

    template <typename T, ArrayLike A>
    static Scalar execute_min(const A& arr) {
      return host::min_recursive<T>(arr);
    }

    // --- Binary non-broadcast operations --- //

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    static void execute_add(const Left& left, const Right& right, ResultType& result) {
      host::add_recursive<T>(left, right, result);
    }

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    static void execute_sub(const Left& left, const Right& right, ResultType& result) {
      host::sub_recursive<T>(left, right, result);
    }

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    static void execute_mul(const Left& left, const Right& right, ResultType& result) {
      host::mul_recursive<T>(left, right, result);
    }

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    static void execute_truediv(const Left& left,
                                const Right& right,
                                ResultType& result) {
      host::truediv_recursive<T>(left, right, result);
    }

    // --- Inplace binary operations --- //

    template <typename T, ArrayLike Left, ArrayLike Right>
    static void execute_inplace_add(Left& left, const Right& right) {
      host::inplace_add_recursive<T>(left, right);
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    static void execute_inplace_sub(Left& left, const Right& right) {
      host::inplace_sub_recursive<T>(left, right);
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    static void execute_inplace_mul(Left& left, const Right& right) {
      host::inplace_mul_recursive<T>(left, right);
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    static void execute_inplace_truediv(Left& left, const Right& right) {
      host::inplace_truediv_recursive<T>(left, right);
    }

    // --- Binary operations with a scalar broadcast --- //

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_add_scalar(const Left& left,
                                   const Scalar& right,
                                   ResultType& result) {
      host::add_scalar_recursive<T>(left, right, result);
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_sub_scalar(const Left& left,
                                   const Scalar& right,
                                   ResultType& result) {
      host::sub_scalar_recursive<T>(left, right, result);
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_mul_scalar(const Left& left,
                                   const Scalar& right,
                                   ResultType& result) {
      host::mul_scalar_recursive<T>(left, right, result);
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_truediv_scalar(const Left& left,
                                       const Scalar& right,
                                       ResultType& result) {
      host::truediv_scalar_recursive<T>(left, right, result);
    }

    // --- Inplace binary operations with a scalar broadcast --- //

    template <typename T, ArrayLike Left>
    static void execute_inplace_add_scalar(Left& left, const Scalar& right) {
      host::inplace_add_scalar_recursive<T>(left, right);
    }

    template <typename T, ArrayLike Left>
    static void execute_inplace_sub_scalar(Left& left, const Scalar& right) {
      host::inplace_sub_scalar_recursive<T>(left, right);
    }

    template <typename T, ArrayLike Left>
    static void execute_inplace_mul_scalar(Left& left, const Scalar& right) {
      host::inplace_mul_scalar_recursive<T>(left, right);
    }

    template <typename T, ArrayLike Left>
    static void execute_inplace_truediv_scalar(Left& left, const Scalar& right) {
      host::inplace_truediv_scalar_recursive<T>(left, right);
    }

    // --- Logical and boolean operators --- //

    template <typename T, class Left, class Right, class Result>
    static void execute_equal(const Left& left, const Right& right, Result& result) {
      host::equal_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_not_equal(const Left& left,
                                  const Right& right,
                                  Result& result) {
      host::not_equal_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_less_than(const Left& left,
                                  const Right& right,
                                  Result& result) {
      host::less_than_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_less_equal_than(const Left& left,
                                        const Right& right,
                                        Result& result) {
      host::less_equal_than_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_greater_than(const Left& left,
                                     const Right& right,
                                     Result& result) {
      host::greater_than_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_greater_equal_than(const Left& left,
                                           const Right& right,
                                           Result& result) {
      host::greater_equal_than_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_logical_and(const Left& left, const Right& right, Result& result) {
      host::logical_and_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_logical_or(const Left& left, const Right& right, Result& result) {
      host::logical_or_recursive<T>(left, right, result);
    }

    template <typename T, class Array, class Result>
    static void execute_logical_not(const Array& arr, Result& result) {
      host::logical_not_recursive<T>(arr, result);
    }

    // --- Comparison operators with scalar broadcast --- //

    template <typename T, class Left, class Result>
    static void execute_equal_scalar(const Left& left,
                                     const Scalar& right,
                                     Result& result) {
      host::equal_scalar_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Result>
    static void execute_not_equal_scalar(const Left& left,
                                         const Scalar& right,
                                         Result& result) {
      host::not_equal_scalar_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Result>
    static void execute_less_than_scalar(const Left& left,
                                         const Scalar& right,
                                         Result& result) {
      host::less_than_scalar_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Result>
    static void execute_less_equal_than_scalar(const Left& left,
                                               const Scalar& right,
                                               Result& result) {
      host::less_equal_than_scalar_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Result>
    static void execute_greater_than_scalar(const Left& left,
                                            const Scalar& right,
                                            Result& result) {
      host::greater_than_scalar_recursive<T>(left, right, result);
    }

    template <typename T, class Left, class Result>
    static void execute_greater_equal_than_scalar(const Left& left,
                                                  const Scalar& right,
                                                  Result& result) {
      host::greater_equal_than_scalar_recursive<T>(left, right, result);
    }

    // --- Inplace logical operators --- //

    template <typename T, class Left, class Right>
    static void execute_inplace_logical_and(Left& left, const Right& right) {
      host::inplace_logical_and_recursive<T>(left, right);
    }

    template <typename T, class Left, class Right>
    static void execute_inplace_logical_or(Left& left, const Right& right) {
      host::inplace_logical_or_recursive<T>(left, right);
    }

    // --- Logical operators with scalar broadcast --- //

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_logical_and_scalar(const Left& left,
                                           const Scalar& right,
                                           ResultType& result) {
      host::logical_and_scalar_recursive<T>(left, right, result);
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_logical_or_scalar(const Left& left,
                                          const Scalar& right,
                                          ResultType& result) {
      host::logical_or_scalar_recursive<T>(left, right, result);
    }

    // --- Inplace logical operators with scalar broadcast --- //

    template <typename T, ArrayLike Left>
    static void execute_inplace_logical_and_scalar(Left& left, const Scalar& right) {
      host::inplace_logical_and_scalar_recursive<T>(left, right);
    }

    template <typename T, ArrayLike Left>
    static void execute_inplace_logical_or_scalar(Left& left, const Scalar& right) {
      host::inplace_logical_or_scalar_recursive<T>(left, right);
    }

    // --- Copy and Modification --- //

    template <typename T, ArrayLike Left>
    static void execute_fill(Left& left, const Scalar& val) {
      ssize_t starting_axis { 0 };
      auto fill_op_internal = [](auto&& arg) -> T {
        using FromT = std::decay_t<decltype(arg)>;
        return ncarray::op_traits<FromT>::template cast<T>(arg);
      };
      T target_val = std::visit(fill_op_internal, val);
      host::impl::fill_recursive<T>(left, left.data(), starting_axis, target_val);
    }

    template <typename T, ArrayLike Left, typename OutputType>
    static void execute_copy_into(Left& left, OutputType*& dest) {
      ssize_t starting_axis { 0 };
      host::impl::copy_into_recursive<T, OutputType>(left,
                                                     left.data(),
                                                     starting_axis,
                                                     dest);
    }

    template <typename DestT, ArrayLike Dest, ArrayLike Src>
    static void execute_assign(Dest& dest, const Src& src) {
      auto assign_op_internal = [&]<typename SrcT>() {
        ssize_t starting_axis { 0 };
        host::impl::assign_recursive<DestT, SrcT>(dest,
                                                  src,
                                                  dest.data(),
                                                  src.data(),
                                                  starting_axis);
      };

      dispatch(src.dtype(), assign_op_internal);
    }
  };

#ifdef __CUDACC__
  struct GPUEngine {
    // --- Binary non-broadcast operations --- //
    template <typename T, class Left, class Right, class Result>
    static void execute_add(const Left& left, const Right& right, Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      add_kernel<T><<<blocks, TPB>>>(left.view(), right.view(), result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_sub(const Left& left, const Right& right, Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      sub_kernel<T><<<blocks, TPB>>>(left.view(), right.view(), result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_mul(const Left& left, const Right& right, Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      mul_kernel<T><<<blocks, TPB>>>(left.view(), right.view(), result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_truediv(const Left& left, const Right& right, Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      truediv_kernel<T><<<blocks, TPB>>>(left.view(), right.view(), result.view());
      cudaDeviceSynchronize();
    }

    // --- Inplace binary operations --- //
    template <typename T, ArrayLike Left, ArrayLike Right>
    static void execute_inplace_add(Left& left, const Right& right) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_add_kernel<T><<<blocks, TPB>>>(left.view(), right.view());
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    static void execute_inplace_sub(Left& left, const Right& right) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_sub_kernel<T><<<blocks, TPB>>>(left.view(), right.view());
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    static void execute_inplace_mul(Left& left, const Right& right) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_mul_kernel<T><<<blocks, TPB>>>(left.view(), right.view());
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    static void execute_inplace_truediv(Left& left, const Right& right) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_truediv_kernel<T><<<blocks, TPB>>>(left.view(), right.view());
      cudaDeviceSynchronize();
    }

    // --- Binary operations with a scalar broadcast --- //

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_add_scalar(const Left& left,
                                   const Scalar& right,
                                   ResultType& result) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1) / TPB) };

      add_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val, result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_sub_scalar(const Left& left,
                                   const Scalar& right,
                                   ResultType& result) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1) / TPB) };

      sub_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val, result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_mul_scalar(const Left& left,
                                   const Scalar& right,
                                   ResultType& result) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };
      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1) / TPB) };

      mul_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val, result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_truediv_scalar(const Left& left,
                                       const Scalar& right,
                                       ResultType& result) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };
      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1) / TPB) };

      truediv_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val, result.view());
      cudaDeviceSynchronize();
    }

    // --- Inplace binary operations with a scalar broadcast --- //

    template <typename T, ArrayLike Left>
    static void execute_inplace_add_scalar(Left& left, const Scalar& right) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_add_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val);
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike Left>
    static void execute_inplace_sub_scalar(Left& left, const Scalar& right) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_sub_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val);
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike Left>
    static void execute_inplace_mul_scalar(Left& left, const Scalar& right) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_mul_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val);
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike Left>
    static void execute_inplace_truediv_scalar(Left& left, const Scalar& right) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_truediv_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val);
      cudaDeviceSynchronize();
    }

    // --- Logical and boolean operators --- //

    template <typename T, class Left, class Right, class Result>
    static void execute_equal(const Left& left, const Right& right, Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      equal_kernel<T><<<blocks, TPB>>>(left.view(), right.view(), result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_not_equal(const Left& left,
                                  const Right& right,
                                  Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      not_equal_kernel<T><<<blocks, TPB>>>(left.view(), right.view(), result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_less_than(const Left& left,
                                  const Right& right,
                                  Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      less_than_kernel<T><<<blocks, TPB>>>(left.view(), right.view(), result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_less_equal_than(const Left& left,
                                        const Right& right,
                                        Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      less_equal_than_kernel<T><<<blocks, TPB>>>(left.view(),
                                                 right.view(),
                                                 result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_greater_than(const Left& left,
                                     const Right& right,
                                     Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      greater_than_kernel<T><<<blocks, TPB>>>(left.view(), right.view(), result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_greater_equal_than(const Left& left,
                                           const Right& right,
                                           Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      greater_equal_than_kernel<T><<<blocks, TPB>>>(left.view(),
                                                    right.view(),
                                                    result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_logical_and(const Left& left,
                                    const Right& right,
                                    Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      logical_and_kernel<T><<<blocks, TPB>>>(left.view(), right.view(), result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_logical_or(const Left& left, const Right& right, Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      logical_or_kernel<T><<<blocks, TPB>>>(left.view(), right.view(), result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Array, class Result>
    static void execute_logical_not(const Array& arr, Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };

      logical_not_kernel<T><<<blocks, TPB>>>(arr.view(), result.view());
      cudaDeviceSynchronize();
    }

    // --- Comparison operators with scalar broadcast --- //

    template <typename T, class Left, class Result>
    static void execute_equal_scalar(const Left& left,
                                     const Scalar& right,
                                     Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      equal_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val, result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Result>
    static void execute_not_equal_scalar(const Left& left,
                                         const Scalar& right,
                                         Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      not_equal_scalar_kernel<T><<<blocks, TPB>>>(left.view(),
                                                  scalar_val,
                                                  result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Result>
    static void execute_less_than_scalar(const Left& left,
                                         const Scalar& right,
                                         Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      less_than_scalar_kernel<T><<<blocks, TPB>>>(left.view(),
                                                  scalar_val,
                                                  result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Result>
    static void execute_less_equal_than_scalar(const Left& left,
                                               const Scalar& right,
                                               Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      less_equal_than_scalar_kernel<T><<<blocks, TPB>>>(left.view(),
                                                        scalar_val,
                                                        result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Result>
    static void execute_greater_than_scalar(const Left& left,
                                            const Scalar& right,
                                            Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      greater_than_scalar_kernel<T><<<blocks, TPB>>>(left.view(),
                                                     scalar_val,
                                                     result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Result>
    static void execute_greater_equal_than_scalar(const Left& left,
                                                  const Scalar& right,
                                                  Result& result) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      greater_equal_than_scalar_kernel<T><<<blocks, TPB>>>(left.view(),
                                                           scalar_val,
                                                           result.view());
      cudaDeviceSynchronize();
    }

    // --- Inplace logical operators --- //

    template <typename T, class Left, class Right>
    static void execute_inplace_logical_and(Left& left, Right& right) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_logical_and_kernel<T><<<blocks, TPB>>>(left.view(), right.view());
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right>
    static void execute_inplace_logical_or(Left& left, Right& right) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_logical_or_kernel<T><<<blocks, TPB>>>(left.view(), right.view());
      cudaDeviceSynchronize();
    }

    // --- Logical operators with scalar broadcast --- //

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_logical_and_scalar(const Left& left,
                                           const Scalar& right,
                                           ResultType& result) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1) / TPB) };

      logical_and_scalar_kernel<T><<<blocks, TPB>>>(left.view(),
                                                    scalar_val,
                                                    result.view());
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_logical_or_scalar(const Left& left,
                                          const Scalar& right,
                                          ResultType& result) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1) / TPB) };

      logical_or_scalar_kernel<T><<<blocks, TPB>>>(left.view(),
                                                   scalar_val,
                                                   result.view());
      cudaDeviceSynchronize();
    }

    // --- Inplace logical operators with scalar broadcast --- //

    template <typename T, ArrayLike Left>
    static void execute_inplace_logical_and_scalar(Left& left, const Scalar& right) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_logical_and_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val);
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike Left>
    static void execute_inplace_logical_or_scalar(Left& left, const Scalar& right) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks{static_cast<int>((left.size() + TPB - 1)) / TPB};

      inplace_logical_or_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val);
      cudaDeviceSynchronize();
    }

    // --- Reductions --- //

    template <typename T, class ArrayT>
    static Scalar execute_sum(const ArrayT& arr) {
      using AccumT = typename op_traits<T>::sum_type;
      CircularDevicePool<AccumT>& mem_pool = CircularDevicePool<AccumT>::instance();
      using MemEntry = typename CircularDevicePool<AccumT>::MemEntry;

      MemEntry ptrs { mem_pool.next() };

      *ptrs.h_ptr = AccumT { 0 };

      constexpr int TPB { 256 };
      int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };

      sum_kernel<TPB, T><<<blocks, TPB>>>(arr.view(), ptrs.d_ptr);
      cudaDeviceSynchronize();

      return Scalar { *ptrs.h_ptr };
    }

    template <typename T, class ArrayT>
    static Scalar execute_mean(const ArrayT& arr) {
      using AccumT = typename op_traits<T>::sum_type;
      using ResultT = typename op_traits<T>::truediv_type;

      CircularDevicePool<AccumT>& mem_pool = CircularDevicePool<AccumT>::instance();
      using MemEntry = typename CircularDevicePool<AccumT>::MemEntry;

      MemEntry ptrs { mem_pool.next() };

      *ptrs.h_ptr = AccumT { 0 };

      constexpr int TPB { 256 };
      int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };

      sum_kernel<TPB, T><<<blocks, TPB>>>(arr.view(), ptrs.d_ptr);
      cudaDeviceSynchronize();

      return Scalar {
        static_cast<ResultT>(*ptrs.h_ptr) / static_cast<double>(arr.size())
      };
    }

    template <typename T, class ArrayT>
    static Scalar execute_max(const ArrayT& arr) {
      CircularDevicePool<T>& mem_pool = CircularDevicePool<T>::instance();
      using MemEntry = typename CircularDevicePool<T>::MemEntry;

      MemEntry ptrs { mem_pool.next() };

      *ptrs.h_ptr = op_traits<T>::lowest();

      constexpr int TPB { 256 };
      int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };

      max_kernel<TPB, T><<<blocks, TPB>>>(arr.view(), ptrs.d_ptr);
      cudaDeviceSynchronize();

      return Scalar { *ptrs.h_ptr };
    }

    template <typename T, class ArrayT>
    static Scalar execute_min(const ArrayT& arr) {
      CircularDevicePool<T>& mem_pool = CircularDevicePool<T>::instance();
      using MemEntry = typename CircularDevicePool<T>::MemEntry;

      MemEntry ptrs { mem_pool.next() };

      *ptrs.h_ptr = op_traits<T>::max();

      constexpr int TPB { 256 };
      int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };

      min_kernel<TPB, T><<<blocks, TPB>>>(arr.view(), ptrs.d_ptr);
      cudaDeviceSynchronize();

      return Scalar { *ptrs.h_ptr };
    }

    // --- Copy and modification --- //
    template <typename T, class ArrayT>
    static void execute_fill(const ArrayT& arr, Scalar val) {
      auto cast_op = [](auto&& arg) -> T {
        using FromT = std::decay_t<decltype(arg)>;

        return ncarray::op_traits<FromT>::template cast<T>(arg);
      };
      T target_val = std::visit(cast_op, val);

      int TPB { 256 };
      int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };

      fill_kernel<<<blocks, TPB>>>(arr.view(), target_val);
      cudaDeviceSynchronize();
    }

    template <typename T, ArrayLike ArrayT, typename OutputType>
    static void execute_copy_into(const ArrayT& arr, OutputType* dest) {
      cudaPointerAttributes dest_attrs;
      cudaError_t err = cudaPointerGetAttributes(&dest_attrs, dest);
      bool dest_is_dev { false };
      if (err == cudaSuccess) {
        dest_is_dev =
          dest_attrs.type == cudaMemoryTypeDevice ||
          dest_attrs.type == cudaMemoryTypeManaged;
      } else {
        cudaGetLastError();
      }

      // NOTE: cudaMemcpyDefault can presumably hide some of this complexity
      //       however, it hasn't been working realiably, seemingly, so manual it is.
      // We also use this for host->device transfers to make sure CPU-bound
      // implementations don't need to know about CUDA. This means we have to
      // check src memory type as well, though.
      using SrcMemType = typename ArrayT::MemType;
      bool src_is_dev = std::is_same_v<SrcMemType, DevTag>;

      if (arr.is_contiguous() && std::is_same_v<T, OutputType>) {
        // Simplest case, the array is contiguous and there are no casts
        auto copy_kind = [src_is_dev, dest_is_dev] () {
          if (src_is_dev) {
            return dest_is_dev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
          }
          return dest_is_dev ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
        }();

        CHECK_CUDA_ERROR(cudaMemcpy(dest,
                                    arr.data(),
                                    arr.size() * sizeof(T),
                                    copy_kind));
      } else {
        // Non-contiguous or we have to cast
        int TPB { 256 };
        int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };
        if (dest_is_dev && src_is_dev) {
          // Casting copy from device to device
          copy_into_kernel<T, OutputType><<<blocks, TPB>>>(dest, arr.view());
          cudaDeviceSynchronize();
        } else if (src_is_dev) {
          // Casting copy from device to host
          // TODO: Make this more optimized...
          // Create a temporary contiguous buffer then cudaMemcpy
          OutputType* d_tmp { nullptr };
          CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, arr.size() * sizeof(OutputType)));
          copy_into_kernel<T, OutputType><<<blocks, TPB>>>(d_tmp, arr.view());
          cudaDeviceSynchronize();
          CHECK_CUDA_ERROR(cudaMemcpy(dest,
                                      d_tmp,
                                      arr.size() * sizeof(OutputType),
                                      cudaMemcpyDeviceToHost));
          CHECK_CUDA_ERROR(cudaFree(d_tmp));
        } else if (dest_is_dev) {
          // Casting copy from host to device
          // Copy to a contiguous buffer first for simplicity...
          // TODO: Optimize this since it implies TWO copies atm...

          // Create temporary contiguous buffer like (casting into it)
          // NOTE: I suspect that an issue arises only when you have integrated and
          //       dedicated GPUs (like on a laptop)
          //       There is some issue where pageable host memory is not working in DMAs
          //       Switch to using mapped and pinned memory
          cudaGetLastError(); // Flush any hidden errors
          OutputType* h_tmp { nullptr };
          std::size_t nbytes {
            static_cast<std::size_t>(arr.size()) * sizeof(OutputType)
          };
          CHECK_CUDA_ERROR(cudaMallocHost(reinterpret_cast<void**>(&h_tmp), nbytes));
          OutputType* h_ref = h_tmp; // Host copy routine takes *& - so pass a second
          HostEngine::execute_copy_into<T>(arr, h_ref);

          cudaGetLastError(); // Flush any hidden errors...

          // Now copy into destination on device
          CHECK_CUDA_ERROR(cudaMemcpy(dest, h_tmp, nbytes, cudaMemcpyHostToDevice));
          cudaDeviceSynchronize();
          cudaGetLastError(); // Flush any hidden errors...
          CHECK_CUDA_ERROR(cudaFreeHost(h_tmp));
        } else {
          // This shouldn't happen... but somehow ended up with host-to-host
          // transfer in GPUEngine...
          HostEngine::execute_copy_into<T>(arr, dest);
        }
      }
    }

    template <typename DestT, ArrayLike Dest, ArrayLike Src>
    static void execute_assign(Dest& dest, const Src& src) {
      auto assign_op_internal = [&] <typename SrcT> () {
        GPUEngine::execute_copy_into<SrcT>(src,
                                           reinterpret_cast<DestT*>(dest.data()));
      };

      dispatch(src.dtype(), assign_op_internal);
    }
  };
#endif
} // namespace ncarray

#endif // NCARRAY_ENGINES_HH
