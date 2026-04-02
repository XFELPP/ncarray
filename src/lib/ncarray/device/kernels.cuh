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
#include "ncarray/device/atomic.cuh"
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

  // --- Inplace binary operations --- //

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT>
  __global__ void inplace_add_kernel(LeftT left, const RightT right) {
    device::inplace_block_add<T, LeftT, RightT>(left, right);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT>
  __global__ void inplace_sub_kernel(LeftT left, const RightT right) {
    device::inplace_block_sub<T, LeftT, RightT>(left, right);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT>
  __global__ void inplace_mul_kernel(LeftT left, const RightT right) {
    device::inplace_block_mul<T, LeftT, RightT>(left, right);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT>
  __global__ void inplace_truediv_kernel(LeftT left, const RightT right) {
    device::inplace_block_truediv<T, LeftT, RightT>(left, right);
  }

  // --- Binary operations with a scalar broadcast --- //

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void add_scalar_kernel(const LeftT left, const T scalar, OutT out) {
    device::block_scalar_add(left, scalar, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void sub_scalar_kernel(const LeftT left, const T scalar, OutT out) {
    device::block_scalar_sub(left, scalar, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void mul_scalar_kernel(const LeftT left, const T scalar, OutT out) {
    device::block_scalar_mul(left, scalar, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void truediv_scalar_kernel(const LeftT left, const T scalar, OutT out) {
    device::block_scalar_truediv(left, scalar, out);
  }

  // --- Inplace binary operations with a scalar broadcast --- //

  template <typename T, ViewArrayLike LeftT>
  __global__ void inplace_add_scalar_kernel(LeftT left, const T scalar) {
    device::inplace_block_scalar_add<T, LeftT>(left, scalar);
  }

  template <typename T, ViewArrayLike LeftT>
  __global__ void inplace_sub_scalar_kernel(LeftT left, const T scalar) {
    device::inplace_block_scalar_sub<T, LeftT>(left, scalar);
  }

  template <typename T, ViewArrayLike LeftT>
  __global__ void inplace_mul_scalar_kernel(LeftT left, const T scalar) {
    device::inplace_block_scalar_mul<T, LeftT>(left, scalar);
  }

  template <typename T, ViewArrayLike LeftT>
  __global__ void inplace_truediv_scalar_kernel(LeftT left, const T scalar) {
    device::inplace_block_scalar_truediv<T, LeftT>(left, scalar);
  }

  // --- Reductions --- //
  template <int BlockSize, typename T, ViewArrayLike ArrayT>
  __global__ void sum_kernel(const ArrayT arr,
                             typename op_traits<T>::sum_type* res) {
    using AccumT = typename op_traits<T>::sum_type;
    AccumT block_res_sum = ncarray::device::block_sum<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      device::nca_atomic_add(res, block_res_sum);
    }
  }

  template <int BlockSize, typename T, ViewArrayLike ArrayT>
  __global__ void max_kernel(const ArrayT arr, T* res) {
    T block_res_max = ncarray::device::block_max<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      device::nca_atomic_min(res, block_res_max);
    }
  }

  template <int BlockSize, typename T, ViewArrayLike ArrayT>
  __global__ void min_kernel(const ArrayT arr, T* res) {
    T block_res_min = ncarray::device::block_min<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      device::nca_atomic_min(res, block_res_min);
    }
  }

  // --- Copy and Modification --- //
  template <ViewArrayLike OutT, typename T>
  __global__ void fill_kernel(OutT out, T val) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };

    if (idx < out.size()) {
      out.template operator[]<T>(idx) = val;
    }
  }
} // namespace ncarray

#endif // NCARRAY_DEVICE_KERNELS_HH
