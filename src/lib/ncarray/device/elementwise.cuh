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

#ifdef __CUDACC_RTC__
typedef long long ssize_t;
#else
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif
#endif

#ifndef NCARRAY_MAX_NDIM
#define NCARRAY_MAX_NDIM 10
#endif

namespace ncarray {
  namespace device {
    namespace impl {
      template <typename T, typename ResultT, class ArrayT, class OutT, class Op>
      __device__ inline void block_unary_transform(const ArrayT& arr,
                                                   OutT& result,
                                                   Op op) {
        unsigned tid { threadIdx.x };
        unsigned stride { blockDim.x };

        for (ssize_t i = static_cast<ssize_t>(tid); i < arr.size(); i += stride) {
          ssize_t coords[NCARRAY_MAX_NDIM];
          ssize_t tmp_idx { i };

          for (ssize_t d = arr.ndim() - 1; d >= 0; --d) {
            coords[d] = tmp_idx % arr.shape(d);
            tmp_idx /= arr.shape(d);
          }

          const void* ptr = const_cast<void*>(arr.data());
          for (int d = 0; d < arr.ndim(); ++d) {
            ptr = arr.advance(ptr, d, coords[d]);
          }

          void* res_ptr = result.data();
          for (int d = 0; d < result.ndim(); ++d) {
            res_ptr = result.advance(res_ptr, d, coords[d]);
          }

          *static_cast<ResultT*>(res_ptr) = op(*static_cast<const T*>(ptr));
        }
      }

      template <
        typename T,
        typename ResultT,
        class LeftT,
        class RightT,
        class OutT,
        class Op
      >
      __device__ inline void block_binary_transform(const LeftT& left,
                                                    const RightT& right,
                                                    OutT& result,
                                                    Op op) {
        unsigned tid { blockIdx.x * blockDim.x + threadIdx.x };
        unsigned stride { blockDim.x * gridDim.x };

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

          *static_cast<ResultT*>(res_ptr) =
            op(*static_cast<const T*>(lhs_ptr), *static_cast<const T*>(rhs_ptr));
        }
      }

      template <typename T, typename ResultT, class ArrayT, class Op>
      __device__ inline void block_inplace_unary_transform(ArrayT& arr, Op op) {
        unsigned tid { blockIdx.x * blockDim.x + threadIdx.x };
        unsigned stride { blockDim.x * gridDim.x };

        for (ssize_t i = static_cast<ssize_t>(tid); i < arr.size(); i += stride) {
          ssize_t coords[NCARRAY_MAX_NDIM];
          ssize_t tmp_idx { i };

          for (ssize_t d = arr.ndim() - 1; d >= 0; --d) {
            coords[d] = tmp_idx % arr.shape(d);
            tmp_idx /= arr.shape(d);
          }

          void* ptr = const_cast<void*>(arr.data());
          for (int d = 0; d < arr.ndim(); ++d) {
            ptr = arr.advance(ptr, d, coords[d]);
          }

          op(*static_cast<T*>(ptr));
        }
      }

      template <typename T, class LeftT, class RightT, class Op>
      __device__ inline void block_inplace_binary_transform(LeftT& left,
                                                            const RightT& right,
                                                            Op op) {
        unsigned tid { blockIdx.x * blockDim.x + threadIdx.x };
        unsigned stride { blockDim.x * gridDim.x };

        for (ssize_t i = static_cast<ssize_t>(tid); i < left.size(); i += stride) {
          ssize_t coords[NCARRAY_MAX_NDIM];
          ssize_t tmp_idx { i };

          for (ssize_t d = left.ndim() - 1; d >= 0; --d) {
            coords[d] = tmp_idx % left.shape(d);
            tmp_idx /= left.shape(d);
          }

          void* lhs_ptr = const_cast<void*>(left.data());
          for (int d = 0; d < left.ndim(); ++d) {
            lhs_ptr = left.advance(lhs_ptr, d, coords[d]);
          }

          void* rhs_ptr = const_cast<void*>(right.data());

          ssize_t diff = left.ndim() - right.ndim();
          for (int d = 0; d < right.ndim(); ++d) {
            ssize_t r_coord = (right.shape(d) == 1) ? 0 : coords[d + diff];
            rhs_ptr = right.advance(rhs_ptr, d, r_coord);
          }

          op(*static_cast<T*>(lhs_ptr), *static_cast<const T*>(rhs_ptr));
        }
      }

      /**
       * Scalar broadcast version of block_binary_transform.
       */
      template <typename T, typename ResultT, class LeftT, class OutT, class Op>
      __device__ inline void block_binary_scalar_transform(const LeftT& left,
                                                           const T& scalar_val,
                                                           OutT& result,
                                                           Op op) {
        unsigned tid { blockIdx.x * blockDim.x + threadIdx.x };
        unsigned stride { blockDim.x * gridDim.x };

        for (ssize_t i = static_cast<ssize_t>(tid); i < left.size(); i += stride) {
          ssize_t coords[NCARRAY_MAX_NDIM];
          ssize_t tmp_idx { i };
          for (ssize_t d = left.ndim() - 1; d >= 0; --d) {
            coords[d] = tmp_idx % left.shape(d);
            tmp_idx /= left.shape(d);
          }

          const void* lhs_ptr = left.data();
          for (int d = 0; d < left.ndim(); ++d) {
            lhs_ptr = left.advance(lhs_ptr, d, coords[d]);
          }

          void* res_ptr = result.data();
          for (int d = 0; d < result.ndim(); ++d) {
            res_ptr = result.advance(res_ptr, d, coords[d]);
          }
          *static_cast<ResultT*>(res_ptr) =
            op(*static_cast<const T*>(lhs_ptr), scalar_val);
        }
      }

      template <typename T, class LeftT, class Op>
      __device__ inline void block_inplace_binary_scalar_transform(LeftT& left,
                                                                   const T& scalar_val,
                                                                   Op op) {
        unsigned tid { blockIdx.x * blockDim.x + threadIdx.x };
        unsigned stride { blockDim.x * gridDim.x };

        for (ssize_t i = static_cast<ssize_t>(tid); i < left.size(); i += stride) {
          ssize_t coords[NCARRAY_MAX_NDIM];
          ssize_t tmp_idx { i };

          for (ssize_t d = left.ndim() - 1; d >= 0; --d) {
            coords[d] = tmp_idx % left.shape(d);
            tmp_idx /= left.shape(d);
          }

          void* lhs_ptr = const_cast<void*>(left.data());
          for (int d = 0; d < left.ndim(); ++d) {
            lhs_ptr = left.advance(lhs_ptr, d, coords[d]);
          }

          op(*static_cast<T*>(lhs_ptr), scalar_val);
        }
      }
    } // namespace impl
  } // namespace device
} // namespace ncarray

#endif // NCARRAY_DEVICE_ELEMENTWISE_HH
