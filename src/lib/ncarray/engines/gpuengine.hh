/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ENGINES_GPUENGINE_HH
#define NCARRAY_ENGINES_GPUENGINE_HH

#include "ncarray/array_impl.hh"
#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"
#ifdef __CUDACC__
#include "ncarray/device/atomic.cuh"
#include "ncarray/device/kernels.cuh"
#include "ncarray/device/warp.cuh"
#endif
#include "ncarray/device/mem_pool.cuh"
#include "ncarray/device/utilities.cuh"
#include "ncarray/engines/hostengine.hh" // For fallback to host copy routines
#include "ncarray/expression.hh"
#include "ncarray/indexing.hh"
#include "ncarray/jit/rtcompiler.hh"
#include "ncarray/layout.hh"
#include "ncarray/mvnode.hh"
#include "ncarray/op_traits.hh"
#include "ncarray/reductions.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <type_traits>

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

#ifndef NCA_D
#ifdef __CUDACC__
#define NCA_D __device__
#else
#define NCA_D
#endif
#endif

namespace ncarray {
#ifdef __CUDACC__
  /**
   * The GPUEngine is ultimately responsible for dispatching all operations involving
   * arrays on the GPU. This includes binary operations, reductions, unary
   * operations, copies, assignments, fills and so on.
   */
  struct GPUEngine {

    // --- Binary Operations --- //

    /**
     * Execute a binary expression.
     *
     * In reality, this function dispatches all operations with the exception of
     * single array reductions, for which there an axis-aware, and full-to-scalar
     * functions exist.
     *
     * A runtime compiler uses NVRTC to JIT compile an expression which is built
     * via a dynamic expression node object when writing operator-based expressions.
     * The runtime compiler will cache the PTX in the user cache directory for faster
     * lookup on subsequent runs (given that the expression is identical). A fallback
     * routine is also provided for larger expressions which is AOT compiled and included
     * in the shared library.
     *
     * @tparam DestT The datatype of the final output array.
     * @tparam Expr The class of the expression object.
     * @tparam Result The class of the result array.
     * @param expr The runtime-constructed node containing the operands and operations.
     * @param result The pre-allocated result array. The `expr` node will be able to
     *        provide the correct size/shape and datatype to construct this array to
     *        pass into this function.
     */
    template <typename DestT, class Expr, OwningArrayLike Result>
    static void execute_binary_expression(const Expr& expr, Result& result) {
      const int size { static_cast<int>(result.size()) };
      constexpr int TPB { 256 };

      int blocks { static_cast<int>(size + TPB - 1) / TPB };

      // NOTE: RuntimeCompiler will cache the compiled kernel. For the duration
      // of the program, it is cached in emmory. It is also saved to disk for lookup
      // on subsequent runs of the application (XDG_CONFIG, or Windows equivalent)
      if constexpr (is_exprmv_node_v<Expr>) {

        if (can_linearize(expr)) {
          auto n_views { expr.layouts.size() };
          auto n_scalars { expr.scalars.size() };
          auto kernel =
            RuntimeCompiler::instance().get_expr_kernel(result.dtype(),
                                                        expr.dtype(),
                                                        n_views,
                                                        n_scalars,
                                                        expr.instrs,
                                                        expr.soarray);

          // NOTE: I would just use a normal vector<DestT> ... but vector<bool> problems...
          using ScalarStorageT = std::conditional_t<std::is_same_v<DestT, bool>,
                                                    std::uint8_t,
                                                    DestT>;
          std::vector<ScalarStorageT> casted_scalars;
          casted_scalars.reserve(expr.scalars.size());

          auto cast_op = [&](auto&& val) {
            using ScalarT = std::decay_t<decltype(val)>;
            // Second cast is a NOOP for everything but bool
            return static_cast<ScalarStorageT>(op_traits<ScalarT>::template cast<DestT>(val));
          };
          for (const auto& scalar : expr.scalars) {
            casted_scalars.push_back(std::visit(cast_op, scalar));
          }

          std::vector<void*> args;
          for (std::size_t i = 0; i < n_views; ++i) {
            args.push_back(const_cast<void*>(reinterpret_cast<const void*>(&expr.data[i])));
            args.push_back(const_cast<void*>(reinterpret_cast<const void*>(&expr.layouts[i])));
          }

          for (std::size_t i = 0; i < n_scalars; ++i) {
            args.push_back(static_cast<void*>(&casted_scalars[i]));
          }

          auto view = result.view();
          args.push_back(reinterpret_cast<void*>(&view));

          CUresult launch_res = cuLaunchKernel(kernel,
                                               blocks, 1, 1, // Grid dims  (x, y, z)
                                               TPB, 1, 1,    // Block dims (x, y, z)
                                               0,            // Shmem in bytes
                                               0,            // Stream
                                               args.data(),  // Kernel args
                                               NULL);

          if (launch_res != CUDA_SUCCESS) {
            throw std::runtime_error("Kernel launch failed!");
          }
          cuCtxSynchronize();
        } else {
          auto vm = DynamicExprMVNode(expr);

          execute_expression_kernel<DestT><<<blocks, TPB>>>(vm, result.view());
        }
      }

      cudaDeviceSynchronize();
    }

    // --- Axis-Aware Reductions --- //

    /**
     * A simple helper to determine if the size is a power of 2.
     */
    static bool is_pow2(ssize_t n) { return n > 0 && (n & (n - 1)) == 0; }

    /**
     * Perform reduction along a set of axes.
     *
     * Currently, only a block-wide binning approach is used for all reduction factors.
     * Kernels are available that can perform 1 bin per warp, or 1 bin per thread,
     * for users interested in testing these for their use cases.
     *
     * For full reduction to a scalar, there is another function available with
     * generalized atomics across the grid.
     *
     * @tparam SrcT The datatype of the array.
     * @tparam Traits The holder struct for the type of reduction.
     * @tparam Array The array type.
     * @tparam Result The output array type.
     * @param arr Array to be reduced.
     * @param params Reduction parameters - i.e., axes to be reduced.
     * @param res Pre-allocated array for the output data.
     */
    template <
      typename SrcT,
      ReductionTraits<SrcT> Traits,
      ViewArrayLike Array,
      ViewArrayLike Result
    >
    static void execute_reduce_axes(const Array& arr,
                                    const ReductionParams& params,
                                    Result& res) {
      using AccumT = typename Traits::AccumT<SrcT>;

      using OutT = typename Traits::OutT<SrcT>;

      AccumT identity { Traits::template identity<SrcT>() };

      constexpr int TPB { 256 };

      ssize_t reduction_factor { arr.size() / res.size() };

      Reducer<SrcT, Traits> reducer;

      int blocks { static_cast<int>((arr.size() + TPB - 1) / TPB) };
      bool use_shifts { true };
      for (ssize_t dim = 0; dim < arr.ndim(); ++dim) {
        if (!GPUEngine::is_pow2(arr.shape()[dim])) {
          use_shifts = false;
          break;
        }
      }

      AccumT* d_scratch { nullptr };
      cudaMalloc(reinterpret_cast<void**>(&d_scratch),
                 res.size() * sizeof(AccumT));

      fill_raw_kernel<AccumT><<<blocks, TPB>>>(d_scratch, identity, res.size());

      if (use_shifts) {
        axes_reduce_kernel<true, SrcT><<<blocks, TPB>>>(arr,
                                                        d_scratch,
                                                        res.view(),
                                                        params,
                                                        reducer);
      } else {
        axes_reduce_kernel<false, SrcT><<<blocks, TPB>>>(arr,
                                                         d_scratch,
                                                         res.view(),
                                                         params,
                                                         reducer);
      }

      cudaDeviceSynchronize();
    }

    // --- Full Reductions (To Scalar) --- //

    /**
     * Perform a complete reduction of the array to a single scalar.
     *
     * @tparam SrcT The datatype of the array.
     * @tparam Traits The holder struct for the type of reduction.
     * @tparam Array The array type.
     * @param arr Array to be reduced.
     */
    template <
      typename SrcT,
      ReductionTraits<SrcT> Traits,
      class Array
    >
    static Scalar execute_full_reduce(const Array& arr) {
      using AccumT = typename Traits::AccumT<SrcT>;

      using OutT = typename Traits::OutT<SrcT>;

      Reducer<SrcT, Traits> reducer;
      AccumT identity { reducer.identity() };

      constexpr int TPB { 256 };

      int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };

      CircularDevicePool<AccumT>& mem_pool = CircularDevicePool<AccumT>::instance();
      using MemEntry = typename CircularDevicePool<AccumT>::MemEntry;

      MemEntry ptrs { mem_pool.next() };

      *ptrs.h_ptr = identity;

      reduce_kernel<TPB, SrcT, Array, Reducer<SrcT, Traits>><<<blocks, TPB>>>(arr.view(),
                                                                              reducer,
                                                                              ptrs.d_ptr);
      cudaDeviceSynchronize();

      OutT final_res { reducer.store(*ptrs.h_ptr) };

      return Scalar { final_res };
    }

    // --- Copy and Modification --- //

    template <typename T, class ArrayT>
    static void execute_fill(const ArrayT& arr, Scalar val) {
      using DestLayoutP = typename ArrayT::LayoutPolicy;
      constexpr bool dest_is_soarr = std::is_same_v<DestLayoutP, SOArrayPolicy>;

      auto cast_op = [](auto&& arg) -> T {
        using FromT = std::decay_t<decltype(arg)>;

        return ncarray::op_traits<FromT>::template cast<T>(arg);
      };
      T target_val = std::visit(cast_op, val);

      int TPB { 256 };
      int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };

      auto kernel =
        RuntimeCompiler::instance().get_fill_kernel(arr.dtype(), dest_is_soarr);

      auto view = arr.view();

      std::vector<void*> args;
      args.push_back(reinterpret_cast<void*>(&view));
      args.push_back(reinterpret_cast<void*>(&target_val));

      CUresult launch_res = cuLaunchKernel(kernel,
                                           blocks, 1, 1, // Grid dims  (x, y, z)
                                           TPB, 1, 1,    // Block dims (x, y, z)
                                           0,            // Shmem in bytes
                                           0,            // Stream
                                           args.data(),  // Kernel args
                                           NULL);

      if (launch_res != CUDA_SUCCESS) {
        throw std::runtime_error("Kernel launch failed!");
      }
      cuCtxSynchronize();
      cudaDeviceSynchronize();
    }

    template <typename SrcT, ArrayLike ArrayT, typename DestT>
    static void execute_copy_into(const ArrayT& arr, DestT* dest) {
      cudaPointerAttributes dest_attrs;
      cudaError_t err = cudaPointerGetAttributes(&dest_attrs, dest);
      bool dest_is_dev { false };
      if (err == cudaSuccess) {
        dest_is_dev =
          dest_attrs.type == cudaMemoryTypeDevice ||
          dest_attrs.type == cudaMemoryTypeManaged;
      } else {
        cudaGetLastError();
      }

      using SrcLayoutP = typename ArrayT::LayoutPolicy;
      constexpr bool src_is_soarr = std::is_same_v<SrcLayoutP, SOArrayPolicy>;

      // NOTE: cudaMemcpyDefault can presumably hide some of this complexity
      //       however, it hasn't been working realiably, seemingly, so manual it is.
      // We also use this for host->device transfers to make sure CPU-bound
      // implementations don't need to know about CUDA. This means we have to
      // check src memory type as well, though.
      using SrcMemType = typename ArrayT::MemType;
      bool src_is_dev = std::is_same_v<SrcMemType, DevTag>;

      if (arr.is_contiguous() && std::is_same_v<SrcT, DestT>) {
        // Simplest case, the array is contiguous and there are no casts
        auto copy_kind = [src_is_dev, dest_is_dev] () {
          if (src_is_dev) {
            return dest_is_dev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
          }
          return dest_is_dev ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
        }();

        CHECK_CUDA_ERROR(cudaMemcpy(dest,
                                    arr.data(),
                                    arr.size() * sizeof(SrcT),
                                    copy_kind));
      } else {
        // Non-contiguous or we have to cast
        int TPB { 256 };
        int blocks { static_cast<int>((arr.size() + TPB - 1)) / TPB };
        if (dest_is_dev && src_is_dev) {
          // Casting copy from device to device
          auto kernel = RuntimeCompiler::instance().get_copy_kernel(dtype_traits<DestT>::value,
                                                                    arr.dtype(),
                                                                    src_is_soarr);
          auto view = arr.view();

          std::vector<void*> args;
          args.push_back(reinterpret_cast<void*>(&dest));
          args.push_back(reinterpret_cast<void*>(&view));

          CUresult launch_res = cuLaunchKernel(kernel, blocks, 1, 1, // Grid dims  (x, y, z)
                                               TPB, 1, 1,            // Block dims (x, y, z)
                                               0,                    // Shmem in bytes
                                               0,                    // Stream
                                               args.data(),          // Kernel args
                                               NULL);

          if (launch_res != CUDA_SUCCESS) {
            throw std::runtime_error("Kernel launch failed!");
          }
          cuCtxSynchronize();
          cudaDeviceSynchronize();
        } else if (src_is_dev) {
          // Casting copy from device to host
          // TODO: Make this more optimized...
          // Create a temporary contiguous buffer then cudaMemcpy
          DestT* d_tmp { nullptr };
          CHECK_CUDA_ERROR(cudaMalloc(&d_tmp, arr.size() * sizeof(DestT)));
          //copy_into_kernel<DestT, SrcT><<<blocks, TPB>>>(d_tmp, arr.view());

          auto kernel = RuntimeCompiler::instance().get_copy_kernel(dtype_traits<DestT>::value,
                                                                    arr.dtype(),
                                                                    src_is_soarr);
          auto view = arr.view();

          std::vector<void*> args;
          args.push_back(reinterpret_cast<void*>(&d_tmp));
          args.push_back(reinterpret_cast<void*>(&view));

          CUresult launch_res = cuLaunchKernel(kernel, blocks, 1, 1, // Grid dims  (x, y, z)
                                               TPB, 1, 1,            // Block dims (x, y, z)
                                               0,                    // Shmem in bytes
                                               0,                    // Stream
                                               args.data(),          // Kernel args
                                               NULL);

          if (launch_res != CUDA_SUCCESS) {
            throw std::runtime_error("Kernel launch failed!");
          }
          cuCtxSynchronize();
          cudaDeviceSynchronize();

          CHECK_CUDA_ERROR(cudaMemcpy(dest,
                                      d_tmp,
                                      arr.size() * sizeof(DestT),
                                      cudaMemcpyDeviceToHost));
          CHECK_CUDA_ERROR(cudaFree(d_tmp));
        } else if (dest_is_dev) {
          // Casting copy from host to device
          // Copy to a contiguous buffer first for simplicity...
          // TODO: Optimize this since it implies TWO copies atm...

          // Create temporary contiguous buffer like (casting into it)
          // NOTE: I suspect that an issue arises only when you have integrated and
          //       dedicated GPUs (like on a laptop)
          //       There is some issue where pageable host memory is not working in DMAs
          //       Switch to using mapped and pinned memory
          cudaGetLastError(); // Flush any hidden errors
          DestT* h_tmp { nullptr };
          std::size_t nbytes {
            static_cast<std::size_t>(arr.size()) * sizeof(DestT)
          };
          CHECK_CUDA_ERROR(cudaMallocHost(reinterpret_cast<void**>(&h_tmp), nbytes));
          DestT* h_ref = h_tmp; // Host copy routine takes *& - so pass a second
          HostEngine::execute_copy_into<SrcT>(arr, h_ref);

          cudaGetLastError(); // Flush any hidden errors...

          // Now copy into destination on device
          CHECK_CUDA_ERROR(cudaMemcpy(dest, h_tmp, nbytes, cudaMemcpyHostToDevice));
          cudaDeviceSynchronize();
          cudaGetLastError(); // Flush any hidden errors...
          CHECK_CUDA_ERROR(cudaFreeHost(h_tmp));
        } else {
          // This shouldn't happen... but somehow ended up with host-to-host
          // transfer in GPUEngine...
          HostEngine::execute_copy_into<SrcT>(arr, dest);
        }
      }
    }

    template <typename DestT, ArrayLike Dest, ArrayLike Src>
    static void execute_assign(Dest& dest, const Src& src) {
      // TODO: Make this more efficient!!
      // This function handles non-contiguous copies
      using DestMemType = typename Dest::MemType;
      using SrcMemType = typename Src::MemType;

      using DestLayoutP = typename Dest::LayoutPolicy;
      using SrcLayoutP = typename Src::LayoutPolicy;

      constexpr bool dest_is_dev = std::is_same_v<DestMemType, DevTag>;
      constexpr bool src_is_dev = std::is_same_v<SrcMemType, DevTag>;

      constexpr bool dest_is_soarr = std::is_same_v<DestLayoutP, SOArrayPolicy>;
      constexpr bool src_is_soarr = std::is_same_v<SrcLayoutP, SOArrayPolicy>;

      using NCDevArray = NCOwnerFor<DevTag>;

      int TPB { 256 };
      int blocks { static_cast<int>((src.size() + TPB - 1)) / TPB };
      // Build a temporary contiguous device array
      auto dest_dtype = dtype_traits<DestT>::value;
      if constexpr (dest_is_dev) {
        if constexpr (src_is_dev) {
          // Device to device
          //auto dev_dev_op = [&] <typename SrcT> () {
          //  copy_view_into_view_kernel<DestT, SrcT><<<blocks, TPB>>>(dest, src);
          //};
          //dispatch(src.dtype(), dev_dev_op);

          auto kernel =
            RuntimeCompiler::instance().get_copy_view_into_view_kernel(dest.dtype(),
                                                                       src.dtype(),
                                                                       dest_is_soarr,
                                                                       src_is_soarr);
          auto dest_view = dest.view();
          auto src_view = src.view();

          std::vector<void*> args;
          args.push_back(reinterpret_cast<void*>(&dest_view));
          args.push_back(reinterpret_cast<void*>(&src_view));

          CUresult launch_res = cuLaunchKernel(kernel, blocks, 1, 1, // Grid dims  (x, y, z)
                                               TPB, 1, 1,            // Block dims (x, y, z)
                                               0,                    // Shmem in bytes
                                               0,                    // Stream
                                               args.data(),          // Kernel args
                                               NULL);

          if (launch_res != CUDA_SUCCESS) {
            throw std::runtime_error("Kernel launch failed!");
          }
          cuCtxSynchronize();

          cudaDeviceSynchronize();
        } else {
          // Host to device
          NCDevArray tmp(dest.ndim(), dest.shape(), dest_dtype);
          auto tmp_view = tmp.view();

          auto host_dev_op = [&] <typename SrcT> () {
            GPUEngine::execute_copy_into<SrcT>(src, reinterpret_cast<DestT*>(tmp.data()));
          };
          dispatch(src.dtype(), host_dev_op);

          auto kernel =
            RuntimeCompiler::instance().get_copy_view_into_view_kernel(dest.dtype(),
                                                                       src.dtype(),
                                                                       dest_is_soarr,
                                                                       src_is_soarr);
          auto dest_view = dest.view();
          auto src_view = tmp.view();

          std::vector<void*> args;
          args.push_back(reinterpret_cast<void*>(&dest_view));
          args.push_back(reinterpret_cast<void*>(&src_view));

          CUresult launch_res = cuLaunchKernel(kernel, blocks, 1, 1, // Grid dims  (x, y, z)
                                               TPB, 1, 1,            // Block dims (x, y, z)
                                               0,                    // Shmem in bytes
                                               0,                    // Stream
                                               args.data(),          // Kernel args
                                               NULL);

          if (launch_res != CUDA_SUCCESS) {
            throw std::runtime_error("Kernel launch failed!");
          }
          cuCtxSynchronize();

          cudaDeviceSynchronize();

          //copy_view_into_view_kernel<DestT, DestT><<<blocks, TPB>>>(dest, tmp_view);
          cudaDeviceSynchronize();
        }
      } else if constexpr (src_is_dev) {
        // Device to host
        NCDevArray tmp(dest.ndim(), dest.shape(), dest_dtype);
        auto tmp_view = tmp.view();

        //auto dev_host_op = [&]<typename SrcT>() {
        //  copy_view_into_view_kernel<DestT, SrcT><<<blocks, TPB>>>(tmp_view, src);
        //};
        //dispatch(src.dtype(), dev_host_op);
        //cudaDeviceSynchronize();

        auto kernel =
          RuntimeCompiler::instance().get_copy_view_into_view_kernel(dest.dtype(),
                                                                     src.dtype(),
                                                                     dest_is_soarr,
                                                                     src_is_soarr);
        auto dest_view = tmp_view;
        auto src_view = src.view();

        std::vector<void*> args;
        args.push_back(reinterpret_cast<void*>(&dest_view));
        args.push_back(reinterpret_cast<void*>(&src_view));

        CUresult launch_res = cuLaunchKernel(kernel, blocks, 1, 1, // Grid dims  (x, y, z)
                                             TPB, 1, 1,            // Block dims (x, y, z)
                                             0,                    // Shmem in bytes
                                             0,                    // Stream
                                             args.data(),          // Kernel args
                                             NULL);

        if (launch_res != CUDA_SUCCESS) {
          throw std::runtime_error("Kernel launch failed!");
        }
        cuCtxSynchronize();

        cudaDeviceSynchronize();

        GPUEngine::execute_copy_into<DestT>(tmp, reinterpret_cast<DestT*>(dest.data()));
      } else {
        // Shouldn't happen... but got host to host
        HostEngine::execute_copy_into<DestT>(src, dest);
      }
    }
  };
#endif
} // namespace ncarray

#endif // NCARRAY_ENGINES_GPUENGINE_HH
