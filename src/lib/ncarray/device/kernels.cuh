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
#include "ncarray/custom_types.hh"
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

  // --- Logical and boolean operators --- //

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void equal_kernel(const LeftT left, const RightT right, OutT out) {
    ncarray::device::block_equal<T, LeftT, RightT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void not_equal_kernel(const LeftT left, const RightT right, OutT out) {
    ncarray::device::block_not_equal<T, LeftT, RightT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void less_than_kernel(const LeftT left, const RightT right, OutT out) {
    ncarray::device::block_less_than<T, LeftT, RightT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void less_equal_than_kernel(const LeftT left,
                                         const RightT right,
                                         OutT out) {
    ncarray::device::block_less_equal_than<T, LeftT, RightT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void greater_than_kernel(const LeftT left, const RightT right, OutT out) {
    ncarray::device::block_greater_than<T, LeftT, RightT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void greater_equal_than_kernel(const LeftT left,
                                            const RightT right,
                                            OutT out) {
    ncarray::device::block_greater_equal_than<T, LeftT, RightT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void logical_and_kernel(const LeftT left, const RightT right, OutT out) {
    ncarray::device::block_logical_and<T, LeftT, RightT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT, ViewArrayLike OutT>
  __global__ void logical_or_kernel(const LeftT left, const RightT right, OutT out) {
    ncarray::device::block_logical_or<T, LeftT, RightT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike ArrayT, ViewArrayLike OutT>
  __global__ void logical_not_kernel(const ArrayT arr, OutT out) {
    ncarray::device::block_logical_not<T, ArrayT, OutT>(arr, out);
  }

  // --- Comparison operators with scalar broadcast --- //

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void equal_scalar_kernel(const LeftT left, const T right, OutT out) {
    ncarray::device::block_scalar_equal<T, LeftT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void not_equal_scalar_kernel(const LeftT left, const T right, OutT out) {
    ncarray::device::block_scalar_not_equal<T, LeftT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void less_than_scalar_kernel(const LeftT left, const T right, OutT out) {
    ncarray::device::block_scalar_less_than<T, LeftT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void less_equal_than_scalar_kernel(const LeftT left,
                                                const T right,
                                                OutT out) {
    ncarray::device::block_scalar_less_equal_than<T, LeftT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void greater_than_scalar_kernel(const LeftT left,
                                             const T right,
                                             OutT out) {
    ncarray::device::block_scalar_greater_than<T, LeftT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void greater_equal_than_scalar_kernel(const LeftT left,
                                                   const T right,
                                                   OutT out) {
    ncarray::device::block_scalar_greater_equal_than<T, LeftT, OutT>(left, right, out);
  }

  // --- Inplace logical operators --- //

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT>
  __global__ void inplace_logical_and_kernel(LeftT left, const RightT right) {
    ncarray::device::inplace_block_logical_and<T, LeftT, RightT>(left, right);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike RightT>
  __global__ void inplace_logical_or_kernel(LeftT left, const RightT right) {
    ncarray::device::inplace_block_logical_or<T, LeftT, RightT>(left, right);
  }

  // --- Logical operators with scalar broadcast --- //

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void logical_and_scalar_kernel(const LeftT left, const T right, OutT out) {
    ncarray::device::block_scalar_logical_and<T, LeftT, OutT>(left, right, out);
  }

  template <typename T, ViewArrayLike LeftT, ViewArrayLike OutT>
  __global__ void logical_or_scalar_kernel(const LeftT left, const T right, OutT out) {
    ncarray::device::block_scalar_logical_or<T, LeftT, OutT>(left, right, out);
  }

  // --- Inplace logical operators with scalar broadcast --- //

  template <typename T, ViewArrayLike LeftT>
  __global__ void inplace_logical_and_scalar_kernel(LeftT left, const T scalar) {
    ncarray::device::inplace_block_scalar_logical_and<T, LeftT>(left, scalar);
  }

  template <typename T, ViewArrayLike LeftT>
  __global__ void inplace_logical_or_scalar_kernel(LeftT left, const T scalar) {
    ncarray::device::inplace_block_scalar_logical_or<T, LeftT>(left, scalar);
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
      device::nca_atomic_max(res, block_res_max);
    }
  }
  template <int BlockSize, typename T, ViewArrayLike ArrayT>
  __global__ void argmax_kernel(const ArrayT arr, KeyValPair<ssize_t,T>* max_pair) {
    KeyValPair<ssize_t, T> block_res_argmax =
      ncarray::device::block_argmax<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      device::nca_atomic_argmax<T>(max_pair, block_res_argmax);
    }
  }

  template <int BlockSize, typename T, ViewArrayLike ArrayT>
  __global__ void min_kernel(const ArrayT arr, T* res) {
    T block_res_min = ncarray::device::block_min<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      device::nca_atomic_min(res, block_res_min);
    }
  }
  template <int BlockSize, typename T, ViewArrayLike ArrayT>
  __global__ void argmin_kernel(const ArrayT arr, KeyValPair<ssize_t, T>* min_pair) {
    KeyValPair<ssize_t, T> block_res_argmin =
      ncarray::device::block_argmin<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      device::nca_atomic_argmin<T>(min_pair, block_res_argmin);
    }
  }

  template <int BlockSize, typename T, ViewArrayLike ArrayT>
  __global__ void var_kernel(const ArrayT arr,
                             VarAccumulator<typename op_traits<T>::truediv_type>* res) {
    using ResultT = typename op_traits<T>::truediv_type;
    // Welford approach. Thread computes its triplet for the grid strided segment
    auto block_res_var = ncarray::device::block_var<BlockSize, ArrayT, ResultT>(arr);

    if (threadIdx.x == 0) {
      device::nca_atomic_accumulator_merge<ResultT>(res, block_res_var);
    }
  }

  // --- Copy and Modification --- //
  template <ViewArrayLike OutT, typename T>
  __global__ void fill_kernel(OutT out, T val) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };

    if (idx < out.size()) {
      out[idx] = val;
    }
  }

  template <typename T, typename DestT, ViewArrayLike Src>
  __global__ void copy_into_kernel(DestT* dest, const Src src) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };
    if (idx < src.size()) {
      dest[idx] = op_traits<T>::template cast<DestT>(src[idx]);
    }
  }

  template <typename DestT, typename SrcT, ViewArrayLike Dest, ViewArrayLike Src>
  __global__ void copy_view_into_view_kernel(Dest dest, const Src src) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };
    if (idx < src.size()) {
      dest[idx] = op_traits<SrcT>::template cast<DestT>(src[idx]);
    }
  }

  // Logical ops - all and any
  template <int BlockSize, typename T, ViewArrayLike ArrayT>
  __global__ void all_kernel(const ArrayT arr, bool* res) {
    bool block_res_all = ncarray::device::block_all<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      device::nca_atomic_logical_and(res, block_res_all);
    }
  }

  template <int BlockSize, typename T, ViewArrayLike ArrayT>
  __global__ void any_kernel(const ArrayT arr, bool* res) {
    bool block_res_any = ncarray::device::block_any<BlockSize, ArrayT, T>(arr);

    if (threadIdx.x == 0) {
      device::nca_atomic_logical_or(res, block_res_any);
    }
  }
} // namespace ncarray

#endif // NCARRAY_DEVICE_KERNELS_HH
