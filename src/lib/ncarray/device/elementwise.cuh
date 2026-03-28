/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_DEVICE_ELEMENTWISE_HH
#define NCARRAY_DEVICE_ELEMENTWISE_HH

#include "ncarray/array_traits.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

namespace ncarray {
  namespace device {
    namespace impl {
      template <class LeftT, class RightT, class OutT, class Op>
      __device__ inline void block_binary_transform(const LeftT& left,
                                                    const RightT& right,
                                                    OutT& result,
                                                    Op op) {
        unsigned tid { threadIdx.x };
        unsigned stride { blockDim.x };

        for (ssize_t i = static_cast<ssize_t>(tid); i < left.size(); i += stride) {
          result[i] = op(left[i], right[i]);
        }
      }
    } // namespace impl

    template <class LeftT, class RightT, class OutT>
    __device__ inline void block_add(const LeftT& left,
                                     const RightT& right,
                                     OutT& out) {

      impl::block_binary_transform(left,
                                   right,
                                   out,
                                   [] __device__ (auto a, auto b) { return a + b; });
      //__syncthreads();
    }

    template <class LeftT, class RightT, class OutT>
    __device__ inline void block_sub(const LeftT& left,
                                     const RightT& right,
                                     OutT& out) {

      impl::block_binary_transform(left,
                                   right,
                                   out,
                                   [] __device__(auto a, auto b) { return a - b; });
      //__syncthreads();
    }

    template <class LeftT, class RightT, class OutT>
    __device__ inline void block_mul(const LeftT& left,
                                     const RightT& right,
                                     OutT& out) {

      impl::block_binary_transform(left,
                                   right,
                                   out,
                                   [] __device__(auto a, auto b) { return a * b; });
      //__syncthreads();
    }

    template <class LeftT, class RightT, class OutT>
    __device__ inline void block_truediv(const LeftT& left,
                                         const RightT& right,
                                         OutT& out) {

      impl::block_binary_transform(left,
                                   right,
                                   out,
                                   [] __device__(auto a, auto b) { return a / b; });
      //__syncthreads();
    }
  } // namespace device
} // namespace ncarray

#endif // NCARRAY_DEVICE_ELEMENTWISE_HH
