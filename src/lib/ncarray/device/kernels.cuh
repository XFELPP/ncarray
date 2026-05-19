/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_DEVICE_KERNELS_HH
#define NCARRAY_DEVICE_KERNELS_HH

#include "ncarray/array_element_proxy.hh"
#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"
#include "ncarray/expression/interface.hh"
#ifndef __CUDACC_RTC__
#include "ncarray/device/atomic.cuh"
#include "ncarray/device/elementwise.cuh"
#include "ncarray/device/reductions.cuh"
#include "ncarray/device/utilities.cuh"
#endif
#include "ncarray/layout.hh"
#include "ncarray/op_traits.hh"

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

namespace ncarray {
  // --- Binary operations --- //

#define EVALUTE_EXPR_STATIC_COORDS(RANK)       \
  StaticCoords<RANK, unsigned> coords;         \
                                               \
  result.lin_to_md(b_idx, coords);             \
                                               \
  DestT& res_out = result[coords];             \
  res_out = expr.template eval<DestT>(coords); \
  return;

  template <typename DestT, ArrayExpression Expr, class Result>
  __device__ __forceinline__ void execute_expression_d(Expr expr, Result result) {
    const unsigned size { static_cast<unsigned>(result.size()) };
    unsigned b_idx { blockIdx.x * blockDim.x + threadIdx.x };

    // NOTE: By pulling the reference out first, you avoid a type dispatch inside
    //       The proxy's operator=. I.e. result[coords] = ... will cause a type-safe
    //       check on the datatype. Using operator T& bypasses. This is safe because
    //       because we launched with the datatype - its the kernel tparam DestT!
    switch (result.ndim()) {
    case 1: {
      EVALUTE_EXPR_STATIC_COORDS(1)
    }
    case 2: {
      EVALUTE_EXPR_STATIC_COORDS(2)
    }
    case 3: {
      EVALUTE_EXPR_STATIC_COORDS(3)
    }
    case 4: {
      EVALUTE_EXPR_STATIC_COORDS(4)
    }
    case 5: {
      EVALUTE_EXPR_STATIC_COORDS(5)
    }
    case 6: {
      EVALUTE_EXPR_STATIC_COORDS(6)
    }
    case 7: {
      EVALUTE_EXPR_STATIC_COORDS(7)
    }
    case 8: {
      EVALUTE_EXPR_STATIC_COORDS(8)
    }
    case 9: {
      EVALUTE_EXPR_STATIC_COORDS(9)
    }
    case 10:
    default: {
      EVALUTE_EXPR_STATIC_COORDS(10)
    }
    }
  }

  /**
   * Evaluate/execute/materialize an expression tree into a result array.
   *
   * Running through the expression's eval function will collapse the tree. Expressions
   * can contain many nodes (arithmetic, logical, unary, etc.).
   *
   * @tparam DestT The result array's *data* type.
   * @tparam CoordsT The indexing struct type. Templating on this is critical!
   *         The coordinate indexing structs have constexpr size retrieval. The compiler
   *         is able to perfectly unroll loops needed for array traversal, and you
   *         avoid stack spilling by using these compile-time objects.
   * @tparam Expr The type of the expression.
   * @tparam Result The type of the result array.
   * @param expr The expression tree.
   * @param result The result array.
   * @param size The total size of the array.
   */
  template <typename DestT, ArrayExpression Expr, class Result>
  __global__ void execute_expression_kernel(Expr expr, Result result) {
    const unsigned size { static_cast<unsigned>(result.size()) };
    unsigned b_idx { blockIdx.x * blockDim.x + threadIdx.x };

    if (b_idx < size) {
      execute_expression_d<DestT>(expr, result);
    }
  }

#ifndef __CUDACC_RTC__

  // --- Reductions --- //

#define AXES_REDUCE_STATIC_COORDS(RANK)                                     \
  StaticCoords<RANK, unsigned> coords;                                      \
  ssize_t j { 0 };                                                          \
  ssize_t tmp_i { idx };                                                    \
                                                                            \
  _Pragma("unroll")                                                         \
  for (int dim = RANK - 1; dim >= 0; --dim) {                               \
    ssize_t coord;                                                          \
    if constexpr (Pow2) {                                                   \
      coord = (tmp_i >> params.shifts[dim]) & params.masks[dim];            \
    } else {                                                                \
      coord = (tmp_i % params.shape[dim]);                                  \
      tmp_i /= params.shape[dim];                                           \
    }                                                                       \
    coords[dim] = static_cast<unsigned>(coord);                             \
    j += coord * params.strides[dim];                                       \
  }                                                                         \
                                                                            \
  T& item = arr[coords];                                                    \
                                                                            \
  auto out_proxy = static_index(out, j);                                    \
  reduce_op(item, scratch, out_proxy, idx, j);

  /**
   * Perform a reduction using 1 bin per block along specified axes.
   *
   * @tparam Pow2 If dims are all powers of 2, division/modulos can be substituted for bitwise ops.
   * @tparam T The datatype of the array.
   * @tparam AccumT The datatype of the accumulator for the reduction.
   * @tparam Array The input array type.
   * @tparam OutT The output array type.
   * @tparam ReducerType The reducer type, specifying the reduction operation.
   * @param arr The input array.
   * @param scratch A scratch array for holding intermediate results.
   * @param out The output array for holding the reduced result.
   * @param params The reduction parameters specifying axes to reduce.
   * @param reducer The reducer object holding the reduction lambdas.
   */
  template <
    bool Pow2,
    typename T,
    typename AccumT,
    ViewArrayLike Array,
    ViewArrayLike OutT,
    class ReducerType
  >
  __global__ void axes_reduce_kernel(const Array arr,
                                     AccumT* scratch,
                                     OutT out,
                                     ReductionParams params,
                                     ReducerType reducer) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };

    if (idx < arr.size()) {

      auto reduce_op = [&] __device__ (T item,
                                       auto scratch_ptr,
                                       auto out_proxy,
                                       ssize_t idx,
                                       ssize_t j) {
        using OutDType = typename ReducerType::OutT;
        AccumT val = reducer.transform(idx, item);

        if constexpr (sizeof(AccumT) == 4 || sizeof(AccumT) == 8) {
          unsigned mask = __match_any_sync(__activemask(),
                                           static_cast<unsigned long long>(j));

          int leader { __ffs(mask) - 1 };
          int lane { static_cast<int>(threadIdx.x % 32) };

          unsigned current_mask { mask };
          current_mask &= ~(1U << leader); // The leader already has its value
          while (current_mask) {
            int src_lane { __ffs(current_mask) - 1 };

            unsigned offset = (src_lane > lane) ? (src_lane - lane) : 0;

            AccumT other = nca_shfl_down(mask, val, offset);

            if (lane == leader) {
              val = reducer.reduce(val, other);
            }

            current_mask &= ~(1U << src_lane);
          }

          if (lane == leader) {
            OutDType& out_item = out_proxy;

            if constexpr (std::is_same_v<OutDType, AccumT>) {
              reducer.atomic(&out_item, val);
            } else {
              reducer.dual_atomic(&scratch_ptr[j], val, &out_item, params.ddof);
            }
          }
        } else {
          OutDType& out_item = out_proxy;
          if constexpr (std::is_same_v<OutDType, AccumT>) {
            reducer.atomic(&out_item, val);
          } else {
            reducer.dual_atomic(&scratch_ptr[j], val, &out_item, params.ddof);
          }
        }
      };

      switch (params.ndim) {
      case 1: {
        AXES_REDUCE_STATIC_COORDS(1)
        return;
      }
      case 2: {
        AXES_REDUCE_STATIC_COORDS(2)
        return;
      }
      case 3: {
        AXES_REDUCE_STATIC_COORDS(3)
        return;
      }
      case 4: {
        AXES_REDUCE_STATIC_COORDS(4)
        return;
      }
      case 5: {
        AXES_REDUCE_STATIC_COORDS(5)
        return;
      }
      case 6: {
        AXES_REDUCE_STATIC_COORDS(6)
        return;
      }
      case 7: {
        AXES_REDUCE_STATIC_COORDS(7)
        return;
      }
      case 8: {
        AXES_REDUCE_STATIC_COORDS(8)
        return;
      }
      case 9: {
        AXES_REDUCE_STATIC_COORDS(9)
        return;
      }
      case 10:
      default: {
        AXES_REDUCE_STATIC_COORDS(10)
        return;
      }
      }
    }
  }

#define FILL_STATIC_BIN_COORDS(RANK)                                        \
  StaticCoords<RANK, unsigned> coords;                                      \
  auto base_idx { 0 };                                                      \
  auto tmp_j { j };                                                         \
                                                                            \
  _Pragma("unroll")                                                         \
  for (int dim = RANK - 1; dim >= 0; --dim) {                               \
    if (params.strides[dim] != 0) {                                         \
      auto coord = tmp_j % params.shape[dim];                               \
      coords[dim] = coord;                                                  \
      base_idx += coord * params.in_strides[dim];                           \
      tmp_j /= params.shape[dim];                                           \
    } else {                                                                \
      coords[dim] = 0;                                                      \
    }                                                                       \
  }                                                                         \
                                                                            \
  AccumT bin_sum { identity };                                              \
  for (auto a = 0; a < reduced_elements; ++a) {                             \
    _Pragma("unroll")                                                       \
    for (auto d = 0; d < RANK; ++d) {                                       \
      if (d < params.ndim && params.strides[d] == 0) {                      \
        coords[d] = bin_coords[a][d];                                       \
      }                                                                     \
    }                                                                       \
                                                                            \
    T& item = arr[coords];                                                  \
                                                                            \
    auto global_idx { base_idx + bin_offsets[a] };                          \
    bin_sum = reducer.reduce(bin_sum, reducer.transform(global_idx, item)); \
  }                                                                         \
                                                                            \
  auto res = reducer.store(bin_sum, params.ddof);                           \
  using OutDType = decltype(res);                                           \
  OutDType& out_item = static_index(out, j);                                \
  out_item = res;

  /**
   * Perform a reduction using 1 bin per thread along specified axes.
   *
   * @tparam T The datatype of the array.
   * @tparam AccumT The datatype of the accumulator for the reduction.
   * @tparam Array The input array type.
   * @tparam OutT The output array type.
   * @tparam ReducerType The reducer type, specifying the reduction operation.
   * @param arr The input array.
   * @param out The output array for holding the reduced result.
   * @param params The reduction parameters specifying axes to reduce.
   * @param reduced_elements Total number of reduced elements.
   * @param identity The identity value for this kind of reduction.
   * @param reducer The reducer object holding the reduction lambdas.
   */
  template <
    typename T,
    typename AccumT,
    ViewArrayLike Array,
    ViewArrayLike OutT,
    class ReducerType
  >
  __global__ void axes_bin_parallel_kernel(Array arr,
                                           OutT out,
                                           ReductionParams params,
                                           ssize_t reduced_elements,
                                           AccumT identity,
                                           ReducerType reducer) {
    __shared__ ssize_t bin_coords[32][NCARRAY_MAX_NDIM];
    __shared__ ssize_t bin_offsets[32];

    ssize_t j { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };
    if (j >= out.size()) {
      return;
    }

    if (threadIdx.x < reduced_elements) {
      ssize_t a { threadIdx.x };
      ssize_t tmp_a { a };
      ssize_t offset_i { 0 };

      for (ssize_t dim = params.ndim - 1; dim >= 0; --dim) {
        if (params.strides[dim] == 0) {
          ssize_t coord { tmp_a % params.shape[dim] };
          bin_coords[a][dim] = coord;
          offset_i += coord * params.in_strides[dim];
          tmp_a /= params.shape[dim];
        }
      }

      bin_offsets[a] = offset_i;
    }

    __syncthreads();

    switch (params.ndim) {
    case 1: {
      FILL_STATIC_BIN_COORDS(1)
      return;
    }
    case 2: {
      FILL_STATIC_BIN_COORDS(2)
      return;
    }
    case 3: {
      FILL_STATIC_BIN_COORDS(3)
      return;
    }
    case 4: {
      FILL_STATIC_BIN_COORDS(4)
      return;
    }
    case 5: {
      FILL_STATIC_BIN_COORDS(5)
      return;
    }
    case 6: {
      FILL_STATIC_BIN_COORDS(6)
      return;
    }
    case 7: {
      FILL_STATIC_BIN_COORDS(7)
      return;
    }
    case 8: {
      FILL_STATIC_BIN_COORDS(8)
      return;
    }
    case 9: {
      FILL_STATIC_BIN_COORDS(9)
      return;
    }
    case 10:
    default: {
      FILL_STATIC_BIN_COORDS(10)
      return;
    }
    }
  }

#define WARP_REDUCE_STATIC_COORDS(RANK)                                         \
  StaticCoords<RANK, unsigned> base_coords;                                     \
  ssize_t base_idx { 0 };                                                       \
  auto tmp_j { j };                                                             \
                                                                                \
  _Pragma("unroll")                                                             \
  for (int dim = RANK - 1; dim >= 0; --dim) {                                   \
    if (params.strides[dim] != 0) {                                             \
      auto coord = tmp_j % params.shape[dim];                                   \
      base_coords[dim] = static_cast<unsigned>(coord);                          \
      base_idx += coord * params.in_strides[dim];                               \
      tmp_j /= params.shape[dim];                                               \
    } else {                                                                    \
      base_coords[dim] = 0;                                                     \
    }                                                                           \
  }                                                                             \
                                                                                \
  auto warp_transform = [=] __device__ (ssize_t a, const Array& arr_inner) {    \
    StaticCoords<RANK, unsigned> coords = base_coords;                          \
    ssize_t offset_i { 0 };                                                     \
    ssize_t tmp_a { a };                                                        \
                                                                                \
    _Pragma("unroll")                                                           \
    for (int dim = RANK - 1; dim >= 0; --dim) {                                 \
      if (params.strides[dim] == 0) {                                           \
        auto coord = tmp_a % params.shape[dim];                                 \
        coords[dim] = static_cast<unsigned>(coord);                             \
        offset_i += coord * params.in_strides[dim];                             \
        tmp_a /= params.shape[dim];                                             \
      }                                                                         \
    }                                                                           \
                                                                                \
    T& item = arr_inner[coords];                                                \
                                                                                \
    return reducer.transform(base_idx + offset_i, item);                        \
  };                                                                            \
                                                                                \
  auto reduce_op = [&] __device__ (AccumT i, AccumT val) {                      \
    return reducer.reduce(i, val);                                              \
  };                                                                            \
                                                                                \
  AccumT warp_res =                                                             \
    device::impl::warp_reduce_transform<BlockSize, T, AccumT>(arr,              \
                                                              warp_transform,   \
                                                              reduced_elements, \
                                                              reduce_op,        \
                                                              identity);        \
                                                                                \
  if ((threadIdx.x % 32) == 0) {                                                \
    auto res = reducer.store(warp_res, params.ddof);                            \
    using OutDType = decltype(res);                                             \
    OutDType& out_item = static_index(out, j);                                  \
    out_item = res;                                                             \
  }

  /**
   * Perform a reduction using 1 bin per warp along specified axes.
   *
   * @tparam BlockSize The size of the block for the kernel launch.
   * @tparam T The datatype of the array.
   * @tparam AccumT The datatype of the accumulator for the reduction.
   * @tparam Array The input array type.
   * @tparam OutT The output array type.
   * @tparam ReducerType The reducer type, specifying the reduction operation.
   * @param arr The input array.
   * @param out The output array for holding the reduced result.
   * @param params The reduction parameters specifying axes to reduce.
   * @param reduced_elements Total number of reduced elements.
   * @param identity The identity value for this kind of reduction.
   * @param reducer The reducer object holding the reduction lambdas.
   */
  template <
    int BlockSize,
    typename T,
    typename AccumT,
    ViewArrayLike Array,
    ViewArrayLike OutT,
    class ReducerType
  >
  __global__ void axes_warp_reduce_kernel(const Array arr,
                                          OutT out,
                                          ReductionParams params,
                                          ssize_t reduced_elements,
                                          AccumT identity,
                                          ReducerType reducer) {
    // One reduction bin per warp
    unsigned wid { (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32) };
    ssize_t j { static_cast<ssize_t>(wid) };

    if (j >= out.size()) {
      return;
    }

    switch (params.ndim) {
    case 1: {
      WARP_REDUCE_STATIC_COORDS(1)
      return;
    }
    case 2: {
      WARP_REDUCE_STATIC_COORDS(2)
      return;
    }
    case 3: {
      WARP_REDUCE_STATIC_COORDS(3)
      return;
    }
    case 4: {
      WARP_REDUCE_STATIC_COORDS(4)
      return;
    }
    case 5: {
      WARP_REDUCE_STATIC_COORDS(5)
      return;
    }
    case 6: {
      WARP_REDUCE_STATIC_COORDS(6)
      return;
    }
    case 7: {
      WARP_REDUCE_STATIC_COORDS(7)
      return;
    }
    case 8: {
      WARP_REDUCE_STATIC_COORDS(8)
      return;
    }
    case 9: {
      WARP_REDUCE_STATIC_COORDS(9)
      return;
    }
    case 10:
    default: {
      WARP_REDUCE_STATIC_COORDS(10)
      return;
    }
    }
  }

  /**
   * Perform a global reduction of the array to a single scalar.
   *
   * @tparam BlockSize The size of the block for the kernel launch.
   * @tparam T The datatype of the array.
   * @tparam Array The input array type.
   * @tparam ReducerType The reducer type, specifying the reduction operation.
   * @param arr The input array.
   * @param reducer The reducer object holding the reduction lambdas.
   * @param accum Pointer to the memory for the reduced result.
   */
  template <int BlockSize, typename T, ViewArrayLike Array, class ReducerType>
  __global__ void reduce_kernel(const Array arr,
                                ReducerType reducer,
                                typename ReducerType::AccumT* accum) {
    using AccumT = typename ReducerType::AccumT;

    auto transform_op = [&] __device__ (ssize_t idx, T val) {
      return reducer.transform(idx, val);
    };
    auto reduce_op = [&] __device__ (AccumT a, AccumT b) {
      return reducer.reduce(a, b);
    };

    AccumT identity { reducer.identity() };

    AccumT block_accum =
      device::impl::block_reduce_transform<BlockSize, T, AccumT>(arr,
                                                                 transform_op,
                                                                 reduce_op,
                                                                 identity);

    if (threadIdx.x == 0) {
      reducer.atomic(accum, block_accum);
    }
  }
#endif // CUDACC_RTC guard (for nvrtc)

  // --- Copy and Modification --- //

  template <ViewArrayLike OutT, typename T>
  __device__ __forceinline__ void fill_d(OutT out, T val) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };

    if (idx < out.size()) {
      T& out_item = static_index(out, idx);
      out_item = val;
    }
  }

  template <ViewArrayLike OutT, typename T>
  __global__ void fill_kernel(OutT out, T val) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };

    if (idx < out.size()) {
      fill_d(out, val);
    }
  }

  template <typename DestT, typename SrcT, ViewArrayLike Src>
  __device__ __forceinline__ void copy_into_d(DestT* dest, const Src src) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };

    SrcT& src_item = static_index(src, idx);
    dest[idx] = op_traits<SrcT>::template cast<DestT>(src_item);
  }

  template <typename DestT, typename SrcT, ViewArrayLike Src>
  __global__ void copy_into_kernel(DestT* dest, const Src src) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };
    if (idx < src.size()) {
      copy_into_d(dest, src);
    }
  }

  template <typename DestT, typename SrcT, ViewArrayLike Dest, ViewArrayLike Src>
  __device__ __forceinline__ void copy_view_into_view_d(Dest dest, const Src src) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };

    DestT& dest_item = static_index(dest, idx);
    SrcT& src_item = static_index(src, idx);
    dest_item = op_traits<SrcT>::template cast<DestT>(src_item);
  }

  template <typename DestT, typename SrcT, ViewArrayLike Dest, ViewArrayLike Src>
  __global__ void copy_view_into_view_kernel(Dest dest, const Src src) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };
    if (idx < src.size()) {
      copy_view_into_view_d<DestT>(dest, src);
    }
  }

  // --- Raw Memory Helpers --- //

  template <typename T>
  __global__ void fill_raw_kernel(T* out, T val, ssize_t num) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };

    if (idx < num) {
      out[idx] = val;
    }
  }

  // --- Scattering operations --- //

#ifndef __CUDACC_RTC__
  template <
    typename DestT,
    typename IndexT,
    typename SrcT,
    ViewArrayLike Dest,
    ViewArrayLike Index,
    ViewArrayLike Src
  >
  __global__ void scatter_add_kernel(Dest dest, const Index indices, const Src src) {
    ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };

    if (idx < src.size()) {
      ssize_t target { op_traits<IndexT>::template cast<ssize_t>(indices[idx]) };
      if (target >= 0 && target < dest.size()) {
        SrcT& src_item = static_index(src, idx);
        DestT casted = op_traits<SrcT>::template cast<DestT>(src_item);
        DestT& dest_item = static_index(dest, target);
        device::nca_atomic_add(&dest_item, casted);
      }
    }
  }
#endif // nvrtc guard
} // namespace ncarray

#endif // NCARRAY_DEVICE_KERNELS_HH
