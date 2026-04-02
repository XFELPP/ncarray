/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_DEVICE_REDUCTIONS_HH
#define NCARRAY_DEVICE_REDUCTIONS_HH

#include "ncarray/array_traits.hh"

#include "cub/cub.cuh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

namespace ncarray {
  namespace device {
    namespace impl {
      template <int BlockSize, typename T, class ArrayT, class Op>
      __device__ inline T block_reduce_transform(const ArrayT& arr, Op op, T identity) {
        typedef cub::BlockReduce<T, BlockSize> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        unsigned tid { threadIdx.x };
        T thread_val { identity };

        for (ssize_t i = static_cast<ssize_t>(tid); i < arr.size(); i += BlockSize) {
          T& item = arr[i];
          thread_val = op(thread_val, item);
        }

        return BlockReduce(temp_storage).Reduce(thread_val, op);
      }
    } // namespace impl

    template <int BlockSize, class ArrayT, typename T>
    __device__ inline auto block_sum(const ArrayT& arr) {
      using AccumT = typename op_traits<T>::sum_type;

      return impl::block_reduce_transform<BlockSize, AccumT>(arr,
                                                             [] __device__ (auto& a, auto& b) {
                                                               return a + b;
                                                             },
                                                             AccumT { 0 });
    }

    template <int BlockSize, class ArrayT, typename T>
    __device__ inline T block_max(const ArrayT& arr) {
      return impl::block_reduce_transform<BlockSize, T>(arr,
                                                        [] __device__ (auto& a, auto& b) {
                                                          if (op_traits<T>::greater(a, b)) {
                                                            return a;
                                                          }
                                                          return b;
                                                        },
                                                        op_traits<T>::lowest());
    }

    template <int BlockSize, class ArrayT, typename T>
    __device__ inline T block_min(const ArrayT& arr) {
      return impl::block_reduce_transform<BlockSize, T>(arr,
                                                        [] __device__ (auto& a, auto& b) {
                                                          if (op_traits<T>::less(a, b)) {
                                                            return a;
                                                          }
                                                          return b;
                                                        },
                                                        op_traits<T>::max());
    }

    template <int BlockSize, class ArrayT, typename T>
    __device__ inline auto block_mean(const ArrayT& arr) {
      using AccumT = typename op_traits<T>::sum_type;
      using ResultT = typename op_traits<T>::truediv_type;

      AccumT total_sum { block_sum<BlockSize, ArrayT, AccumT>(arr) };

      return static_cast<ResultT>(total_sum) / static_cast<double>(arr.size());
    }
  } // namespace device
} // namespace ncarray

#endif // NCARRAY_DEVICE_REDUCTIONS_HH
