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
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    static void execute_inplace_sub(Left& left, const Right& right) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_sub_kernel<T><<<blocks, TPB>>>(left.view(), right.view());
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    static void execute_inplace_mul(Left& left, const Right& right) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_mul_kernel<T><<<blocks, TPB>>>(left.view(), right.view());
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    static void execute_inplace_truediv(Left& left, const Right& right) {
      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1)) / TPB };

      inplace_truediv_kernel<T><<<blocks, TPB>>>(left.view(), right.view());
    }

    // --- Binary operations with a scalar broadcast --- //

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_add_scalar(const Left& left, const Scalar& right, ResultType& result) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1) / TPB) };

      add_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val, result.view());
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_sub_scalar(const Left& left, const Scalar& right, ResultType& result) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1) / TPB) };

      sub_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val, result.view());
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_mul_scalar(const Left& left, const Scalar& right, ResultType& result) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };
      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1) / TPB) };

      mul_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val, result.view());
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    static void execute_truediv_scalar(const Left& left, const Scalar& right, ResultType& result) {
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };
      T scalar_val = std::visit(cast_op, right);

      int TPB { 256 };
      int blocks { static_cast<int>((left.size() + TPB - 1) / TPB) };

      truediv_scalar_kernel<T><<<blocks, TPB>>>(left.view(), scalar_val, result.view());
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
    }


    // --- Reductions --- //

    template <typename T, class ArrayT>
    static Scalar execute_sum(const ArrayT& arr) {
      using AccumT = typename op_traits<T>::sum_type;
      static CircularDevicePool<AccumT> mem_pool {
        CircularDevicePool<AccumT>::instance()
      };
      using MemEntry = typename CircularDevicePool<AccumT>::MemEntry;
      MemEntry ptrs { mem_pool.next() };

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

      static CircularDevicePool<AccumT> mem_pool {
        CircularDevicePool<AccumT>::instance()
      };
      using MemEntry = typename CircularDevicePool<AccumT>::MemEntry;
      MemEntry ptrs { mem_pool.next() };

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
      static CircularDevicePool<T> mem_pool {
        CircularDevicePool<T>::instance()
      };
      using MemEntry = typename CircularDevicePool<T>::MemEntry;
      MemEntry ptrs { mem_pool.next() };

      constexpr int TPB { 256 };
      int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };

      max_kernel<TPB, T><<<blocks, TPB>>>(arr.view(), ptrs.d_ptr);
      cudaDeviceSynchronize();

      return Scalar { *ptrs.h_ptr };
    }

    template <typename T, class ArrayT>
    static Scalar execute_min(const ArrayT& arr) {
      static CircularDevicePool<T> mem_pool {
        CircularDevicePool<T>::instance()
      };
      using MemEntry = typename CircularDevicePool<T>::MemEntry;
      MemEntry ptrs { mem_pool.next() };

      constexpr int TPB { 256 };
      int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };

      min_kernel<TPB, T><<<blocks, TPB>>>(arr.view(), ptrs.d_ptr);
      cudaDeviceSynchronize();

      return Scalar { *ptrs.h_ptr };
    }

    // --- Copy and modification --- //
    template <typename T, class ArrayT>
    static void execute_fill(const ArrayT& arr, T val) {
      int TPB { 256 };
      int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };

      fill_kernel<<<blocks, TPB>>>(arr.view(), val);
      cudaDeviceSynchronize();
    }
  };
#endif

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
                                   ResultType result) {
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

    template <typename T, ArrayLike Dest, ArrayLike Src>
    static void execute_assign(Dest& dest, const Src& src) {
      auto assign_op_internal = [&]<typename SrcT>() {
        ssize_t starting_axis { 0 };
        host::impl::assign_recursive<Dest, Src>(dest,
                                                src,
                                                dest.data(),
                                                src.data(),
                                                starting_axis);
      };

      dispatch(src.dtype(), assign_op_internal);
    }
  };

} // namespace ncarray

#endif // NCARRAY_ENGINES_HH
