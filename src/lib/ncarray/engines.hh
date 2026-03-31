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
#include "ncarray/device/kernels.cuh"

#include <type_traits>

namespace ncarray {
  struct GPUEngine {
    // --- Binary operations --- //
    template <typename T, class Left, class Right, class Result>
    static void execute_add(const Left& left, const Right& right, Result& result) {
      int TPB { 256 };
      int blocks { (left.size() + TPB - 1) / TPB };

      add_kernel<T, Left, Right, Result><<<blocks, TPB>>>(left, right, result);
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_sub(const Left& left, const Right& right, Result& result) {
      int TPB { 256 };
      int blocks { (left.size() + TPB - 1) / TPB };

      sub_kernel<T, Left, Right, Result><<<blocks, TPB>>>(left, right, result);
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_mul(const Left& left, const Right& right, Result& result) {
      int TPB { 256 };
      int blocks { (left.size() + TPB - 1) / TPB };

      mul_kernel<T, Left, Right, Result><<<blocks, TPB>>>(left, right, result);
      cudaDeviceSynchronize();
    }

    template <typename T, class Left, class Right, class Result>
    static void execute_truediv(const Left& left, const Right& right, Result& result) {
      int TPB { 256 };
      int blocks { (left.size() + TPB - 1) / TPB };

      truediv_kernel<T, Left, Right, Result><<<blocks, TPB>>>(left, right, result);
      cudaDeviceSynchronize();
    }

    // --- Reductions --- //
    template <typename T, class ArrayT>
    static void execute_sum(const ArrayT& arr, typename op_traits<T>::sum_type* result) {
      int TPB { 256 };
      int blocks { (arr.size() + TPB - 1) / TPB };

      sum_kernel<TPB, ArrayT, T><<<blocks, TPB>>>(arr, result);
      cudaDeviceSynchronize();
    }

    template <typename T, class ArrayT>
  static void execute_max(const ArrayT& arr, T* result) {
      int TPB { 256 };
      int blocks { (arr.size() + TPB - 1) / TPB };

      max_kernel<TPB, ArrayT, T><<<blocks, TPB>>>(arr, result);
      cudaDeviceSynchronize();
    }

    template <typename T, class ArrayT>
    static void execute_min(const ArrayT& arr, T* result) {
      int TPB { 256 };
      int blocks { (arr.size() + TPB - 1) / TPB };

      min_kernel<TPB, ArrayT, T><<<blocks, TPB>>>(arr, result);
      cudaDeviceSynchronize();
    }
    /*
    template <typename T, class ArrayT>
    static void execute_mean(const ArrayT& arr,) {
      using AccumT = typename op_traits<T>::sum_type;
      using ResultT = typename op_traits<T>::truediv_type;

      int TPB { 256 };
      int blocks { (arr.size() + TPB - 1) / TPB };

      mean_kernel<TPB><<<blocks, TPB>>>(arr, );
      cudaDeviceSynchronize();
    }
    */
  };

} // namespace ncarray

#endif // NCARRAY_ENGINES_HH
