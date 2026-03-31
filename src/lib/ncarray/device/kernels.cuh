/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_DEVICE_KERNELS_HH
#define NCARRAY_DEVICE_KERNELS_HH

#include "ncarray/array_traits.hh"
#include "ncarray/device/elementwise.cuh"
#include "ncarray/device/reductions.cuh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

namespace ncarray {
  // --- Binary operations --- //
  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void add_kernel(const LeftT left, const RightT right, OutT out) {
    ncarray::device::block_add<T, LeftT, RightT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void sub_kernel(const LeftT left, const RightT right, OutT out) {
    ncarray::device::block_sub<T, LeftT, RightT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void mul_kernel(const LeftT left, const RightT right, OutT out) {
    ncarray::device::block_mul<T, LeftT, RightT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void truediv_kernel(const LeftT left, const RightT right, OutT out) {
    ncarray::device::block_truediv<T, LeftT, RightT, OutT>(left, right, out);
  }

  // --- Reductions --- //
  template <int BlockSize, ViewArrayLike ArrayT, typename T>
  __global__ void sum_kernel(const ArrayT arr,
                             typename op_traits<T>::sum_type* res) {
    using AccumT = typename op_traits<T>::sum_type;
    AccumT block_res_sum = ncarray::device::block_sum<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      atomicAdd(res, block_res_sum);
    }
  }

  template <int BlockSize, ViewArrayLike ArrayT, typename T>
  __global__ void max_kernel(const ArrayT arr, T* res) {
    T block_res_max = ncarray::device::block_max<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      atomicMax(res, block_res_max);
    }
  }

  template <int BlockSize, ViewArrayLike ArrayT, typename T>
  __global__ void min_kernel(const ArrayT arr, T* res) {
    T block_res_min = ncarray::device::block_min<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      atomicMin(res, block_res_min);
    }
  }

  /*
  template <int BlockSize, ViewArrayLike ArrayT, typename T>
  __global__ void mean_kernel(const ArrayT arr,
                              typename op_traits<T>::truediv_type* res) {
    using ResultT = typename op_traits<T>::truediv_type;
    ResultT block_res_mean = ncarray::device::block_mean<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      res[0] = block_res_mean;
    }
  }
  */
} // namespace ncarray

#endif // NCARRAY_DEVICE_KERNELS_HH
