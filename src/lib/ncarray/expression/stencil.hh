/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_EXPRESSION_STENCIL_HH
#define NCARRAY_EXPRESSION_STENCIL_HH

#include "ncarray/array_impl.hh"
#include "ncarray/expression/interface.hh"
#include "ncarray/expression/mvnode.hh"
#include "ncarray/indexing.hh"
#include "ncarray/jit/device/extensions.hh"
#include "ncarray/jit/device/rtcompiler.hh"

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

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
  /**
   * @brief A Stencil for defining expressions over an array with determined offsets.
   *
   * A Stencil holds the offsets and set of operations for reducing the host-time
   * cost of constructing the expression evaluation infrastructure each time that an
   * expression is evaluated. If the same operation(s) is(are) being performed many
   * times with solely the underlying pointers being changed, then the Stencil can
   * reduce the overall execution time significantly.
   *
   * A Stencil can only be created to evaluate operations that run on a single array.
   * Examples include convolutions, or reductions that can be expressed in terms
   * of arithmetic operations on sub-views of the array.
   *
   * @tparam NDIM The number of dimensions of the array operands.
   */
  template <int NDIM>
  class Stencil {
  public:
    Stencil() = default;

    /**
     * @brief Create a stencil from a set of relative offsets and a math function.
     *
     * @tparam T The array datatype.
     * @tparam Func The lambda type for the expression to be evaluated.
     * @param offsets_in List of StaticCoords representing the window (e.g. {0,0}, {0,1}...)
     * @param is_pointer_axis A vector of flags for whether each dimension is a pointer
     *        dim. The compiled kernel will only work for arrays that have the same
     *        kinds of pointer dimensions (although shape and stride can vary freely).
     * @param op_func A lambda that takes a vector of views and returns an expression.
     * @param ext Optionally provide code to run before/after the main expression.
     * @param is_soarr A boolean flag as to whether SOArrayPolicy traversal is needed.
     */
    template <typename SrcT, typename Func>
    static Stencil<NDIM> create(const std::vector<StaticCoords<NDIM>>& offsets_in,
                                const std::vector<std::uint8_t>& is_pointer_axis,
                                Func&& op_func,
                                const device::StencilJITExtensions& ext = {},
                                bool is_soarr = false) {
      Stencil s;
      int num_views { static_cast<int>(offsets_in.size()) };
      s.m_offsets = offsets_in;

      ssize_t unit[NDIM];
      for (int i = 0; i < NDIM; ++i) {
        unit[i] = 1;
      }

      // This will be a placeholder for shape, strides and offsets
      Metadata meta;
      meta.set(unit, NDIM);

      std::vector<NCViewFor<DevTag>> placeholders;
      for (int i=0; i < num_views; ++i) {
        // NOTE: It is fine to use nullptr for the data here. The expression is
        // NOT actually evaluted. This just builds the kernel.
        placeholders.push_back(NCViewFor<DevTag>(nullptr,
                                                 meta, // shape
                                                 meta, // strides
                                                 meta, // offsets
                                                 dtype_traits<SrcT>::value,
                                                 -1,
                                                 false));
      }
      // Construct the expression
      auto expr = op_func(placeholders);
      s.m_instrs = expr.instrs;
      s.m_work_dtype = expr.work_dtype;
      s.m_expr_dtype = expr.dtype();

      auto expr_scalar_cnt { expr.scalars.size() };
      // Pre-allocate buffer for constants. Largest supported scalar type is 32 bytes.
      s.m_constants_buf.resize(expr_scalar_cnt * 32);
      s.m_constants_offsets.resize(expr_scalar_cnt);

      unsigned scalar_cnt { 0 };
      std::uint16_t scalar_off { 0 };

      auto scalar_cast = [&](auto&& val) {
        auto* bytes = reinterpret_cast<const std::uint8_t*>(&val);

        std::uint16_t off = scalar_off;
        s.m_constants_offsets[scalar_cnt] = off;
        scalar_off += sizeof(val);

        for (unsigned i = 0; i < sizeof(val); ++i) {
          s.m_constants_buf[off + i] = bytes[i];
        }

        scalar_cnt++;
      };

      for (const auto& scalar : expr.scalars) {
        std::visit(scalar_cast, scalar);
      }

      // Trim off any remaining bytes - scalar_off was incremented by the final
      // constant's size, so all bytes should be included in that count
      s.m_constants_buf.resize(scalar_off);

      // Build the single kernel to be reused by the stencil going forward.
      s.m_kernel =
        device::RuntimeCompiler::instance().get_stencil_expr_kernel<NDIM>(s.m_expr_dtype,
                                                                          dtype_traits<SrcT>::value,
                                                                          s.m_work_dtype,
                                                                          s.m_offsets,
                                                                          s.m_instrs,
                                                                          expr.scalars,
                                                                          is_pointer_axis,
                                                                          ext,
                                                                          is_soarr);

      return s;
    }

    /**
     * @brief Apply the stencil.
     *
     * Run the pre-constructed kernel on an input array and a pre-created output array.
     * The input must match the input used for creation of the kernel in the number
     * and location of pointer axes as well as the dimensionality. A CUDA stream may be
     * provided to run the execution on, if desired.
     *
     * NOTE: If a stream is provided for running the kernel on, this function will NOT
     * synchronize. This constrasts with other APIs which always do before returning
     * control the host caller. This is because if a stream is passed in order to record
     * the kernel as part of a graph, synchronization is NOT permitted in the capture.
     * Since it is not possible to know whether the stream is provided in order to
     * construct a graph, or not, all synchronization is the caller's responsibility
     * when providing a stream.
     *
     * @tparam Src The input array type.
     * @tparam Dest The output array type.
     * @tparam Args... The set of additional arguments (if any) for pro/epilogue code.
     * @param src The input array.
     * @param dest The output array.
     * @param stream Optionally, provide a stream to launch the kernel on.
     */
    template <ViewArrayLike Src, OwningArrayLike Dest, typename... Args>
    void apply(const Src& src,
               Dest& dest,
               std::optional<cudaStream_t> stream = std::nullopt,
               Args&&... extra_args) {
      constexpr int TPB { 256 };
      int blocks { static_cast<int>(dest.size() + TPB - 1) / TPB };

      void* src_ptr { src.data() };
      void* dest_ptr { dest.data() };

      // We have 4 args for the src and dest layouts and data pointers. Add the
      // the number of scalars/constants to that count for total number of args
      // If extensions are provided via the extra_args, then those will also be
      // added in.
      std::vector<void*> args;
      args.reserve(4 + sizeof...(Args) + m_constants_offsets.size());
      args.push_back(reinterpret_cast<void*>(&src_ptr));
      if constexpr (std::is_base_of_v<SOArrayPolicy, Src>) {
        args.push_back(reinterpret_cast<void*>(&const_cast<SOArrayPolicy&>(static_cast<const SOArrayPolicy&>(src))));
      } else {
        args.push_back(reinterpret_cast<void*>(&const_cast<SOArrayPolicy&>(reinterpret_cast<const SOArrayPolicy&>(src))));
      }

      args.push_back(&dest_ptr);
      if constexpr (std::is_base_of_v<SOArrayPolicy, Src>) {
        args.push_back(reinterpret_cast<void*>(&static_cast<SOArrayPolicy&>(dest)));
      } else {
        args.push_back(reinterpret_cast<void*>(&reinterpret_cast<SOArrayPolicy&>(dest)));
      }

      // Include the custom JIT extension parameters
      (args.push_back(const_cast<void*>(reinterpret_cast<const void*>(&extra_args))), ...);

      // Finally pack in the constants
      for (std::size_t i = 0; i < m_constants_offsets.size(); ++i) {
        auto offset { m_constants_offsets[i] };
        args.push_back(&m_constants_buf[offset]);
      }

      CUresult launch_res = cuLaunchKernel(m_kernel,                         // Kernel
                                           blocks, 1, 1,                     // Grid dims (x, y, z)
                                           TPB, 1, 1,                        // Block dims (x, y, z)
                                           0,                                // Shmem in bytes
                                           stream.has_value() ? *stream : 0, // Stream
                                           args.data(),                      // Kernel args
                                           NULL);

      if (launch_res != CUDA_SUCCESS) {
        const char* err_name;
        cuGetErrorName(launch_res, &err_name);
        printf("Error=%s\n", err_name);
        throw std::runtime_error("Stencil kernel launch failed!");
      }

      if (!stream.has_value()) {
        // When creating graphs, synchronization is expressly prohibited.
        // When passing a stream, the user is therefore responsible for synchronizing
        // the graph launches.
        cuCtxSynchronize();
        cudaDeviceSynchronize();
      }
    }

  private:
    CUfunction m_kernel;
    std::vector<Instruction> m_instrs;
    std::vector<StaticCoords<NDIM>> m_offsets;

    std::vector<std::uint8_t> m_constants_buf;
    std::vector<std::uint16_t> m_constants_offsets;

    DType m_work_dtype;
    DType m_expr_dtype;
  };
} // namespace ncarray

#endif // NCARRAY_EXPRESSION_STENCIL_HH
