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

#ifndef NCARRAY_MAX_NDIM
#define NCARRAY_MAX_NDIM 10
#endif

namespace ncarray {
  namespace device {
    namespace impl {
      template <typename T, class LeftT, class RightT, class OutT, class Op>
      __device__ inline void block_binary_transform(const LeftT& left,
                                                    const RightT& right,
                                                    OutT& result,
                                                    Op op) {
        unsigned tid { threadIdx.x };
        unsigned stride { blockDim.x };

        for (ssize_t i = static_cast<ssize_t>(tid); i < left.size(); i += stride) {
          ssize_t coords[NCARRAY_MAX_NDIM];
          ssize_t tmp_idx { i };

          for (ssize_t d = left.ndim() - 1; d >= 0; --d) {
            coords[d] = tmp_idx % left.shape(d);
            tmp_idx /= left.shape(d);
          }

          const void* lhs_ptr = const_cast<void*>(left.data());
          for (int d = 0; d < left.ndim(); ++d) {
            lhs_ptr = left.advance(lhs_ptr, d, coords[d]);
          }


          void* rhs_ptr = const_cast<void*>(right.data());

          ssize_t diff = left.ndim() - right.ndim();
          for (int d = 0; d < right.ndim(); ++d) {
            ssize_t r_coord = (right.shape(d) == 1) ? 0 : coords[d + diff];
            rhs_ptr = right.advance(rhs_ptr, d, r_coord);
          }
          void* res_ptr = result.data();
          for (int d = 0; d < result.ndim(); ++d) {
            res_ptr = result.advance(res_ptr, d, coords[d]);
          }

          *static_cast<T*>(res_ptr) =
            op(*static_cast<const T*>(lhs_ptr), *static_cast<const T*>(rhs_ptr));
        }
      }
    } // namespace impl

    template <typename T, class LeftT, class RightT, class OutT>
    __device__ inline void block_add(const LeftT& left,
                                     const RightT& right,
                                     OutT& out) {

      impl::block_binary_transform(left,
                                   right,
                                   out,
                                   [] __device__ (auto a, auto b) { return a + b; });
      //__syncthreads();
    }

    template <typename T, class LeftT, class RightT, class OutT>
    __device__ inline void block_sub(const LeftT& left,
                                     const RightT& right,
                                     OutT& out) {

      impl::block_binary_transform(left,
                                   right,
                                   out,
                                   [] __device__(auto a, auto b) { return a - b; });
      //__syncthreads();
    }

    template <typename T, class LeftT, class RightT, class OutT>
    __device__ inline void block_mul(const LeftT& left,
                                     const RightT& right,
                                     OutT& out) {

      impl::block_binary_transform(left,
                                   right,
                                   out,
                                   [] __device__(auto a, auto b) { return a * b; });
      //__syncthreads();
    }

    template <typename T, class LeftT, class RightT, class OutT>
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
