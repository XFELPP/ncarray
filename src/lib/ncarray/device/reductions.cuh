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
#include "ncarray/custom_types.hh"

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
      /**
       * Foundation function for generic block collective reductions.
       *
       * A transformation operation is provided to turn a pair of indices and values
       * into a new type for the accumulation operation. After which the reduction
       * function is used to accumulate. The operations are performed in a grid-striped
       * fashion.
       *
       * @tparam BlockSize The size of the block for CUB helpers (TPB - threads/block)
       * @tparam T The datatype of the array.
       * @tparam AccumT The datatype of accumulated output.
       * @tparam ArrayT The *array* type (not datatype).
       * @tparam TransformOp Type of the transformation function.
       * @tparam ReduceOp Type of the reduction function.
       * @param arr The input array.
       * @param transform The transformation function.
       * @param reduce The reduction function.
       * @param identity The starting identity value for the operation.
       */
      template <
        int BlockSize,
        typename T,
        typename AccumT,
        class ArrayT,
        class TransformOp,
        class ReduceOp
      >
      __device__ inline AccumT block_reduce_transform(const ArrayT& arr,
                                                      TransformOp transform,
                                                      ReduceOp reduce,
                                                      AccumT identity) {
        typedef cub::BlockReduce<AccumT, BlockSize> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        unsigned tid { blockIdx.x * blockDim.x + threadIdx.x };
        unsigned stride { blockDim.x * gridDim.x };
        AccumT thread_val { identity };

        for (ssize_t i = static_cast<ssize_t>(tid); i < arr.size(); i += stride) {
          T& item = arr[i];
          thread_val = reduce(thread_val, transform(i, item));
        }

        return BlockReduce(temp_storage).Reduce(thread_val, reduce);
      }
    } // namespace impl

    /**
     * Perform a block-wide collective sum reduction.
     *
     * @tparam BlockSize The size of the block for CUB helpers (TPB - threads/block)
     * @tparam ArrayT The *array* type (not datatype).
     * @tparam T The datatype of the array.
     * @param arr The input array.
     */
    template <int BlockSize, class ArrayT, typename T>
    __device__ inline auto block_sum(const ArrayT& arr) {
      using AccumT = typename op_traits<T>::sum_type;

      auto cast_op = [&] __device__ (ssize_t idx, T val) {
        return static_cast<AccumT>(val);
      };
      auto sum_op = [&] __device__ (auto a, auto b) {
        return a + b;
      };

      return impl::block_reduce_transform<BlockSize, T, AccumT>(arr,
                                                                cast_op,
                                                                sum_op,
                                                                AccumT { 0 });
    }

    /**
     * Find the maximum across the block.
     *
     * @tparam BlockSize The size of the block for CUB helpers (TPB - threads/block)
     * @tparam ArrayT The *array* type (not datatype).
     * @tparam T The datatype of the array.
     * @param arr The input array.
     */
    template <int BlockSize, class ArrayT, typename T>
    __device__ inline T block_max(const ArrayT& arr) {
      auto pass_op = [&] __device__ (ssize_t idx, T val) { return val; };
      auto max_op = [&] __device__ (auto a, auto b) {
        if (op_traits<T>::greater(a, b)) {
          return a;
        }
        return b;
      };
      return impl::block_reduce_transform<BlockSize, T, T>(arr,
                                                           pass_op,
                                                           max_op,
                                                           op_traits<T>::lowest());
    }
    /**
     * Find the index of the maximum value across the block.
     *
     * @tparam BlockSize The size of the block for CUB helpers (TPB - threads/block)
     * @tparam ArrayT The *array* type (not datatype).
     * @tparam T The datatype of the array.
     * @param arr The input array.
     */
    template <int BlockSize, class ArrayT, typename T>
    __device__ inline KeyValPair<ssize_t, T> block_argmax(const ArrayT& arr) {
      auto create_pair = [&] __device__ (ssize_t idx, T val) {
        return KeyValPair<ssize_t, T>(idx, val);
      };
      auto argmax_op = [&] __device__ (auto a, auto b) {
        if (op_traits<T>::greater(a.val, b.val)) {
          return a;
        }
        return b;
      };

      using Pair = KeyValPair<ssize_t, T>;
      Pair identity { -1, op_traits<T>::lowest() };
      return impl::block_reduce_transform<BlockSize, T, Pair>(arr,
                                                              create_pair,
                                                              argmax_op,
                                                              identity);
    }

    /**
     * Find the minimum across the block.
     *
     * @tparam BlockSize The size of the block for CUB helpers (TPB - threads/block)
     * @tparam ArrayT The *array* type (not datatype).
     * @tparam T The datatype of the array.
     * @param arr The input array.
     */
    template <int BlockSize, class ArrayT, typename T>
    __device__ inline T block_min(const ArrayT& arr) {
      auto pass_op = [&] __device__ (ssize_t idx, T val) { return val; };
      auto min_op = [&] __device__ (auto a, auto b) {
        if (op_traits<T>::less(a, b)) {
          return a;
        }
        return b;
      };
      return impl::block_reduce_transform<BlockSize, T, T>(arr,
                                                           pass_op,
                                                           min_op,
                                                           op_traits<T>::max());
    }
    /**
     * Find the index of the minimum value across the block.
     *
     * @tparam BlockSize The size of the block for CUB helpers (TPB - threads/block)
     * @tparam ArrayT The *array* type (not datatype).
     * @tparam T The datatype of the array.
     * @param arr The input array.
     */
    template <int BlockSize, class ArrayT, typename T>
    __device__ inline KeyValPair<ssize_t, T> block_argmin(const ArrayT& arr) {
      auto create_pair = [&] __device__ (ssize_t idx, T val) {
        return KeyValPair<ssize_t, T>(idx, val);
      };
      auto argmin_op = [&] __device__ (auto a, auto b) {
        if (op_traits<T>::less(a.val, b.val)) {
          return a;
        }
        return b;
      };
      using Pair = KeyValPair<ssize_t, T>;
      Pair identity { -1, op_traits<T>::max() };
      return impl::block_reduce_transform<BlockSize, T, Pair>(arr,
                                                              create_pair,
                                                              argmin_op,
                                                              identity);
    }

    /**
     * Find the variance of the values across the block.
     *
     * @tparam BlockSize The size of the block for CUB helpers (TPB - threads/block)
     * @tparam ArrayT The *array* type (not datatype).
     * @tparam T The datatype of the array.
     * @param arr The input array.
     */
    template <int BlockSize, class ArrayT, typename T>
    __device__ inline VarAccumulator<typename op_traits<T>::truediv_type>
    block_var(const ArrayT& arr) {
      using ResultT = op_traits<T>::truediv_type;
      using AccumT = VarAccumulator<ResultT>;
      auto create_accumulator = [&] __device__(ssize_t idx, T val) {
        return AccumT(1.0, static_cast<ResultT>(val), ResultT { 0.0 });
      };

      // NOTE: This is different than NumPy for complex numbers
      // TODO: Consider whether to make it NumPy style
      auto var_op = [&] __device__ (auto a, auto b) {
        return AccumT::merge(a, b);
      };

      AccumT identity { 0.0, ResultT { 0.0 }, ResultT { 0.0 } };

      return impl::block_reduce_transform<BlockSize, T, AccumT>(arr,
                                                                create_accumulator,
                                                                var_op,
                                                                identity);
    }

    // Logical ops - all and any
    /**
     * Return true if all values are truthy.
     *
     * @tparam BlockSize The size of the block for CUB helpers (TPB - threads/block)
     * @tparam ArrayT The *array* type (not datatype).
     * @tparam T The datatype of the array.
     * @param arr The input array.
     */
    template <int BlockSize, class ArrayT, typename T>
    __device__ inline bool block_all(const ArrayT& arr) {
      auto cast_op = [&] __device__ (ssize_t idx, T val) {
        return op_traits<T>::template cast<bool>(val);
      };
      auto all_op = [&] __device__ (auto a, auto b) { return a && b; };

      bool identity { true };

      return impl::block_reduce_transform<BlockSize, T, bool>(arr,
                                                              cast_op,
                                                              all_op,
                                                              identity);
    }

    /**
     * Return true if any of the values are truthy.
     *
     * @tparam BlockSize The size of the block for CUB helpers (TPB - threads/block)
     * @tparam ArrayT The *array* type (not datatype).
     * @tparam T The datatype of the array.
     * @param arr The input array.
     */
    template <int BlockSize, class ArrayT, typename T>
    __device__ inline bool block_any(const ArrayT& arr) {
      auto cast_op = [&] __device__ (ssize_t idx, T val) {
        return op_traits<T>::template cast<bool>(val);
      };
      auto any_op = [&] __device__ (auto a, auto b) {
        return a || b;
      };

      bool identity { false };

      return impl::block_reduce_transform<BlockSize, T, bool>(arr,
                                                              cast_op,
                                                              any_op,
                                                              identity);
    }
  } // namespace device
} // namespace ncarray

#endif // NCARRAY_DEVICE_REDUCTIONS_HH
