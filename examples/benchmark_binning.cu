#include "ncarray/ncarrays.hh"
#include "ncarray/ncdevarrays.cuh"

#include "ncarray/jit/rtcompiler.hh"

#include "ncarray/expression/stencil.hh"

#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <type_traits>
#include <vector>

void usage(char* progname) {
  std::cerr << "Usage: " << progname << " -p <n_panels> -r <n_rows> -c <n_cols> -t <npix_bin> -b <npix_bin> -n <niter>"
            << std::endl
            << std::endl
            << R"a(
Benchmark fused binning of an image of size (<n_panels>, <n_rows>, <n_cols>).

Args:
  -p <n_panels> Number of "panels" (first of 3 dimensions)
  -r <n_rows>   Number of "rows" (second of 3 dimensions)
  -c <n_cols>   Number of "columns" (third of 3 dimensions)
  -t <npix_bin> Number of pixels to bin on the first axis.
  -b <npix_bin> Number of pixels to bin on the other axes.
  -n <niter>    Number of iterations for timing.
)a" << std::endl;
}

void test_fused_generic_add_bin(ncarray::NCDevArrayView data,
                                std::vector<ssize_t>& res_shape,
                                ssize_t n_iter,
                                ssize_t n_panels,
                                ssize_t n_rows,
                                ssize_t n_cols,
                                ssize_t npix,
                                ssize_t npix_rest) {
  std::cout << "Will test fused sum of views algorithm.";

  std::cout << std::endl << "Initializing data with shape: "
            << "(" << n_panels << ", " << n_rows << ", " << n_cols << ")" << std::endl;

  ncarray::NCArray host_res(res_shape, ncarray::DType::float32);
  double total_build_time { 0.0 };
  double total_exec_time { 0.0 };

  using sl = ncarray::Slice;

  float scale_factor { 1.0f / (npix * npix_rest * npix_rest) };

  ncarray::NCDevArray res(res_shape, ncarray::DType::float32);

  for (ssize_t i = 0; i < n_iter; ++i) {
    auto build_start = std::chrono::high_resolution_clock::now();

    ncarray::ExprMVNode<ncarray::DevTag> expr;
    for (ssize_t p = 0; p < npix; ++p) {
      for (ssize_t r = 0; r < npix_rest; ++r) {
        for (ssize_t c = 0; c < npix_rest; ++c) {
          if (p == 0 && r == 0 && c == 0) {
            expr.build_node(data(sl(p, n_panels, npix), sl(r, n_rows, npix_rest), sl(c, n_cols, npix_rest)));
          } else {
            expr =
              expr + data(sl(p, n_panels, npix), sl(r, n_rows, npix_rest), sl(c, n_cols, npix_rest));
          }
        }
      }
    }

    auto build_end = std::chrono::high_resolution_clock::now();

    auto exec_start = std::chrono::high_resolution_clock::now();

    res = expr * scale_factor;

    auto exec_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_diff = build_end - build_start;
    std::chrono::duration<double> exec_diff = exec_end - exec_start;
    if (i > 0) {
      total_build_time += build_diff.count();
      total_exec_time += exec_diff.count();
    }
  }

  cudaDeviceSynchronize();
  res.copy_into_astype<float>(reinterpret_cast<float*>(host_res.data()));
  std::cout << "Binning build phase took: "
            << (total_build_time / (n_iter - 1)) * 1e6 << " microseconds" << std::endl
            << "Binning exec phase took: "
            << (total_exec_time / (n_iter - 1)) * 1e6 << " microseconds" << std::endl;

  auto first_proxy = host_res[{0, 0, 0}];
  float& first_val = first_proxy;
  auto p_end { static_cast<std::uint64_t>(res_shape[0] - 1) };
  auto r_end { static_cast<std::uint64_t>(res_shape[1] - 1) };
  auto c_end { static_cast<std::uint64_t>(res_shape[2] - 1) };
  auto last_proxy = host_res[{ p_end, r_end, c_end }];
  float& last_val = last_proxy;

  float expected { (npix * npix_rest * npix_rest) * scale_factor };

  std::cout << "Result[0, 0, 0]: " << first_val << " (Expected: " << expected << ")" << std::endl
            << "Result[" << p_end << ", " << r_end << ", " << c_end << "]: " << last_val
            << " (Expected: " << expected << ")" << std::endl;

  if (std::abs(first_val - expected) < 1e-5) {
    std::cout << "SUCCESS: Values are correct." << std::endl;
  } else {
    std::cout << "FAILURE: Value mismatch." << std::endl;
  }
}

void test_fused_generic_add_bin_decomposed(ncarray::NCDevArrayView data,
                                           std::vector<ssize_t>& res_shape,
                                           ssize_t n_iter,
                                           ssize_t n_panels,
                                           ssize_t n_rows,
                                           ssize_t n_cols,
                                           ssize_t npix,
                                           ssize_t npix_rest) {
  std::cout << "Will test fused sum of views algorithm. Bypassing some infrastructure."
            << std::endl
            << "This test will only build arguments once to isolate the kernel itself."
            << std::endl;

  std::cout << std::endl << "Initializing data with shape: "
            << "(" << n_panels << ", " << n_rows << ", " << n_cols << ")" << std::endl;

  ncarray::NCArray host_res(res_shape, ncarray::DType::float32);
  double total_build_time { 0.0 };
  double total_prep_time { 0.0 };
  double total_exec_time { 0.0 };

  using sl = ncarray::Slice;

  float scale_factor { 1.0f / (npix * npix_rest * npix_rest) };

  ncarray::NCDevArray res(res_shape, ncarray::DType::float32);

  constexpr int TPB { 256 };
  int blocks { static_cast<int>(res.size() + TPB - 1) / TPB };
  for (ssize_t i = 0; i < n_iter; ++i) {
    auto build_start = std::chrono::high_resolution_clock::now();

    ncarray::ExprMVNode<ncarray::DevTag> expr;
    for (ssize_t p = 0; p < npix; ++p) {
      for (ssize_t r = 0; r < npix_rest; ++r) {
        for (ssize_t c = 0; c < npix_rest; ++c) {
          if (p == 0 && r == 0 && c == 0) {
            expr.build_node(data(sl(p, n_panels, npix), sl(r, n_rows, npix_rest), sl(c, n_cols, npix_rest)));
          } else {
            expr =
              expr + data(sl(p, n_panels, npix), sl(r, n_rows, npix_rest), sl(c, n_cols, npix_rest));
          }
        }
      }
    }

    expr = expr * scale_factor;

    auto build_end = std::chrono::high_resolution_clock::now();

    // This is what is done inside the engines behind the syntactic sugar of the operators
    if (ncarray::can_linearize(expr)) {
      auto n_views { expr.layouts.size() };
      auto n_scalars { expr.scalars.size() };
      ncarray::DType src_dtype { expr.dtypes.empty() ? expr.work_dtype : expr.dtypes[0] };
      ncarray::DType work_dtype { expr.work_dtype };
      auto kernel = ncarray::RuntimeCompiler::instance().get_expr_kernel(res.dtype(),
                                                                         src_dtype,
                                                                         work_dtype,
                                                                         n_views,
                                                                         n_scalars,
                                                                         expr.ndim(),
                                                                         expr.shape(),
                                                                         expr.instrs,
                                                                         expr.soarray);
      auto launch_op = [&] <typename WorkT> () {
        // Sadly prep is missing the lambda dispatch time
        auto prep_start = std::chrono::high_resolution_clock::now();
        // NOTE: I would just use a normal vector<DestT> ... but vector<bool> problems...
        using ScalarStorageT = std::conditional_t<std::is_same_v<WorkT, bool>,
                                                  std::uint8_t,
                                                  WorkT>;
        std::vector<ScalarStorageT> casted_scalars;
        casted_scalars.reserve(expr.scalars.size());

        auto cast_op = [&](auto&& val) {
          using ScalarT = std::decay_t<decltype(val)>;
          // Second cast is a NOOP for everything but bool
          return static_cast<ScalarStorageT>(ncarray::op_traits<ScalarT>::template cast<WorkT>(val));
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

        auto view = res.view();
        args.push_back(reinterpret_cast<void*>(&view));

        auto prep_end = std::chrono::high_resolution_clock::now();

        auto exec_start = std::chrono::high_resolution_clock::now();
        CUresult launch_res = cuLaunchKernel(kernel,
                                             blocks, 1, 1, // Grid dims  (x, y, z)
                                             TPB, 1, 1,    // Block dims (x, y, z)
                                             0,            // Shmem in bytes
                                             0,            // Stream
                                             args.data(),  // Kernel args
                                             NULL);
        cuCtxSynchronize();
        cudaDeviceSynchronize();
        auto exec_end = std::chrono::high_resolution_clock::now();

        if (i > 0) {
          std::chrono::duration<double> prep_diff = prep_end - prep_start;
          std::chrono::duration<double> exec_diff = exec_end - exec_start;
          total_prep_time += prep_diff.count();
          total_exec_time += exec_diff.count();
        }
        if (launch_res != CUDA_SUCCESS) {
          throw std::runtime_error("Kernel launch failed!");
        }
      };

      dispatch(work_dtype, launch_op);

    } else {
      auto launch_op = [&] <typename DestT> () {
        // Sadly prep is missing the lambda dispatch time...
        auto prep_start = std::chrono::high_resolution_clock::now();

        auto total_bytes = ncarray::bytes_for_dynamic_vm(expr);
        auto& mem_pool = ncarray::CircularDevicePool<uint8_t, 1024 * 1024>::instance();
        using MemEntry = typename ncarray::CircularDevicePool<uint8_t, 1024 * 1024>::MemEntry;

        MemEntry ptrs { mem_pool.get_block(total_bytes) };
        auto vm = ncarray::get_dynamic_mv_node(expr, ptrs.h_ptr, ptrs.d_ptr);
        auto prep_end = std::chrono::high_resolution_clock::now();

        auto exec_start = std::chrono::high_resolution_clock::now();
        execute_expression_kernel<DestT><<<blocks, TPB>>>(vm, res.view());

        cudaDeviceSynchronize();
        auto exec_end = std::chrono::high_resolution_clock::now();

        if (i > 0) {
          std::chrono::duration<double> prep_diff = prep_end - prep_start;
          std::chrono::duration<double> exec_diff = exec_end - exec_start;
          total_prep_time += prep_diff.count();
          total_exec_time += exec_diff.count();
        }
      };

      dispatch(ncarray::DType::float32, launch_op);
    }

    std::chrono::duration<double> build_diff = build_end - build_start;

    if (i > 0) {
      total_build_time += build_diff.count();
    }
  }

  res.copy_into_astype<float>(reinterpret_cast<float*>(host_res.data()));
  std::cout << "Binning build phase took: "
            << (total_build_time / (n_iter - 1)) * 1e6 << " microseconds" << std::endl
            << "Binning prep phase for VM setup took: "
            << (total_build_time / (n_iter - 1)) * 1e6 << " microseconds" << std::endl
            << "Binning exec phase took: "
            << (total_exec_time / (n_iter - 1)) * 1e6 << " microseconds" << std::endl;

  auto first_proxy = host_res[{0, 0, 0}];
  float& first_val = first_proxy;
  auto p_end { static_cast<std::uint64_t>(res_shape[0] - 1) };
  auto r_end { static_cast<std::uint64_t>(res_shape[1] - 1) };
  auto c_end { static_cast<std::uint64_t>(res_shape[2] - 1) };
  auto last_proxy = host_res[{ p_end, r_end, c_end }];
  float& last_val = last_proxy;

  float expected { (npix * npix_rest * npix_rest) * scale_factor };

  std::cout << "Result[0, 0, 0]: " << first_val << " (Expected: " << expected << ")" << std::endl
            << "Result[" << p_end << ", " << r_end << ", " << c_end << "]: " << last_val
            << " (Expected: " << expected << ")" << std::endl;

  if (std::abs(first_val - expected) < 1e-5) {
    std::cout << "SUCCESS: Values are correct." << std::endl;
  } else {
    std::cout << "FAILURE: Value mismatch." << std::endl;
  }
}

void test_fused_stencil_binning(ncarray::NCDevArrayView data,
                                std::vector<ssize_t>& res_shape,
                                ssize_t n_iter,
                                ssize_t n_panels,
                                ssize_t n_rows,
                                ssize_t n_cols,
                                ssize_t npix,
                                ssize_t npix_rest,
                                bool use_graph = false) {
  std::cout << "Will test fused sum of views algorithm with pre-compiled Stencil.";

  std::cout << std::endl << "Initializing data with shape: "
            << "(" << n_panels << ", " << n_rows << ", " << n_cols << ")" << std::endl;

  ncarray::NCArray host_res(res_shape, ncarray::DType::float32);

  double total_alloc_time { 0.0 };
  double total_stencil_time { 0.0 };
  double total_exec_time { 0.0 };

  using sl = ncarray::Slice;

  float scale_factor { 1.0f / (npix * npix_rest * npix_rest) };

  // The stencil requires a few pieces of information at creation time:
  // 1. The "offsets" which are really indices that create the sub-views of the array
  // 2. The compiled kernel is specific to arrays with the same number of location
  //    of pointer axes, so the pointer axis info must be provided.
  // 3. Whether to use SOArrayPolicy as the layout traversal.
  auto alloc_start = std::chrono::high_resolution_clock::now();
  ncarray::NCDevArray res(res_shape, ncarray::DType::float32);
  auto alloc_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> alloc_diff = alloc_end - alloc_start;
  total_alloc_time += alloc_diff.count();

  auto stencil_start = std::chrono::high_resolution_clock::now();
  std::vector<ncarray::StaticCoords<3>> offsets;
  for (ssize_t p = 0; p < npix; ++p) {
    for (ssize_t r = 0; r < npix_rest; ++r) {
      for (ssize_t c = 0; c < npix_rest; ++c) {
        offsets.push_back({p, r, c});
      }
    }
  }

  std::vector<std::uint8_t> is_pointer_axis;
  for (auto dim = 0; dim < data.ndim(); dim++) {
    if (data.is_pointer_axis(dim)) {
      is_pointer_axis.push_back(1);
    } else {
      is_pointer_axis.push_back(0);
    }
  }

  auto binning_expr = [&](auto views) {
    ncarray::ExprMVNode<ncarray::DevTag> expr(views[0]);

    for (std::size_t i = 1; i < views.size(); ++i) {
      expr = expr + views[i];
    }

    return expr * scale_factor;
  };

  auto binning_stencil = ncarray::Stencil<3>::create<float>(offsets,
                                                            is_pointer_axis,
                                                            binning_expr,
                                                            /*is_soarr=*/false);

  auto stencil_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> stencil_diff = stencil_end - stencil_start;
  total_stencil_time += stencil_diff.count();

  if (use_graph) {
    std::cout << "  Will try using graphs." << std::endl;
    cudaGraph_t graph;
    cudaStream_t stream = ncarray::alloc_stream();
    auto capture_start = std::chrono::high_resolution_clock::now();
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    binning_stencil.apply(data.view(), res, stream);
    cudaStreamEndCapture(stream, &graph);
    auto capture_end = std::chrono::high_resolution_clock::now();

    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    std::chrono::duration<double> capture_diff = capture_end - capture_start;

    double total_capture_time = capture_diff.count();

    std::cout << "  Graph creation took: "
              << total_capture_time * 1e6 << " microseconds" << std::endl;
    for (ssize_t i = 0; i < n_iter; ++i) {
      auto exec_start = std::chrono::high_resolution_clock::now();
      cudaGraphLaunch(graphExec, stream);
      // Because graph recording does not allow synchronization inside the capture
      // the apply API when accepting a stream, does not synchronize. Must do that
      // here instead.
      cudaStreamSynchronize(stream);
      auto exec_end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> exec_diff = exec_end - exec_start;
      total_exec_time += exec_diff.count();
    }
  } else {
    for (ssize_t i = 0; i < n_iter; ++i) {

      auto exec_start = std::chrono::high_resolution_clock::now();
      binning_stencil.apply(data.view(), res);

      auto exec_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> exec_diff = exec_end - exec_start;

      total_exec_time += exec_diff.count();
    }
  }

  res.copy_into_astype<float>(reinterpret_cast<float*>(host_res.data()));
  std::cout << "Result array allocation took: "
            << total_alloc_time * 1e6 << " microseconds" << std::endl
            << "Binning stencil creation phase took: "
            << total_stencil_time * 1e6 << " microseconds" << std::endl
            << "Binning stencil execution phase took: "
            << (total_exec_time / n_iter) * 1e6 << " microseconds" << std::endl;

  auto first_proxy = host_res[{0, 0, 0}];
  float& first_val = first_proxy;
  auto p_end { static_cast<std::uint64_t>(res_shape[0] - 1) };
  auto r_end { static_cast<std::uint64_t>(res_shape[1] - 1) };
  auto c_end { static_cast<std::uint64_t>(res_shape[2] - 1) };
  auto last_proxy = host_res[{ p_end, r_end, c_end }];
  float& last_val = last_proxy;

  float expected { (npix * npix_rest * npix_rest) * scale_factor };

  std::cout << "Result[0, 0, 0]: " << first_val << " (Expected: " << expected << ")" << std::endl
            << "Result[" << p_end << ", " << r_end << ", " << c_end << "]: " << last_val
            << " (Expected: " << expected << ")" << std::endl;

  if (std::abs(first_val - expected) < 1e-5) {
    std::cout << "SUCCESS: Values are correct." << std::endl;
  } else {
    std::cout << "FAILURE: Value mismatch." << std::endl;
  }
}


void test_reshape_sum_bin(ncarray::NCDevArrayView data,
                          std::vector<ssize_t>& res_shape,
                          ssize_t n_iter,
                          ssize_t n_panels,
                          ssize_t n_rows,
                          ssize_t n_cols,
                          ssize_t npix,
                          ssize_t npix_rest) {
  std::cout << "Will test reshaping and summing algorithm." << std::endl;
  std::cout << "Initializing "
            << "(" << n_panels << ", " << n_rows << ", " << n_cols << ")" << std::endl;

  ncarray::NCArray host_res(res_shape, ncarray::DType::float32);
  double total_time { 0.0 };

  std::vector<ssize_t> sum_shape {
    n_panels / npix,
    npix,               // Sum this dimension
    n_rows / npix_rest,
    npix_rest,          // Sum this dimension
    n_cols / npix_rest,
    npix_rest           // Sum this dimension
  };
  auto new_view = data.reshape(sum_shape.data(), sum_shape.size());

  std::vector<ssize_t> sum_axes { 1, 3, 5 };
  for (ssize_t i = 0; i < n_iter; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    ncarray::NCDevArray res = new_view.sum(sum_axes);
    cudaDeviceSynchronize(); // superfluous since done internally
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    total_time += diff.count();

    res.copy_into_astype<float>(reinterpret_cast<float*>(host_res.data()));
  }
  std::cout << "Binning took: " << (total_time / n_iter) << " seconds" << std::endl;

  float first_val = *static_cast<float*>(host_res(0, 0, 0).data());
  std::cout << "Result[0,0,0]: " << first_val << " (Expected: 8.0)" << std::endl;

  if (std::abs(first_val - 8.0f) < 1e-5) {
    std::cout << "SUCCESS: Values are correct." << std::endl;
  } else {
    std::cout << "FAILURE: Value mismatch." << std::endl;
  }
}

void test_copy_reshape_sum_bin(ncarray::NCDevArrayView data,
                               std::vector<ssize_t>& res_shape,
                               ssize_t n_iter,
                               ssize_t n_panels,
                               ssize_t n_rows,
                               ssize_t n_cols,
                               ssize_t npix,
                               ssize_t npix_rest) {
  std::cout << "Will test reshaping and summing with a copy first algorithm." << std::endl;
  std::cout << "Initializing "
            << "(" << n_panels << ", " << n_rows << ", " << n_cols << ")" << std::endl;

  ncarray::NCArray host_res(res_shape, ncarray::DType::float32);
  double total_time { 0.0 };

  std::vector<ssize_t> sum_shape {
    n_panels / npix,
    npix,               // Sum this dimension
    n_rows / npix_rest,
    npix_rest,          // Sum this dimension
    n_cols / npix_rest,
    npix_rest           // Sum this dimension
  };

  std::vector<ssize_t> sum_axes { 1, 3, 5 };
  for (ssize_t i = 0; i < n_iter; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    // This emulates the case where due to pointer layouts reshape cannot be
    // done immediately
    auto new_data = data.copy_as_shape(sum_shape.data(), sum_shape.size());
    auto new_view = new_data.view();

    ncarray::NCDevArray res = new_view.sum(sum_axes);
    cudaDeviceSynchronize(); // superfluous since done internally
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    total_time += diff.count();

    res.copy_into_astype<float>(reinterpret_cast<float*>(host_res.data()));
  }
  std::cout << "Binning took: " << (total_time / n_iter) << " seconds" << std::endl;

  float first_val = *static_cast<float*>(host_res(0, 0, 0).data());
  std::cout << "Result[0,0,0]: " << first_val << " (Expected: 8.0)" << std::endl;

  if (std::abs(first_val - 8.0f) < 1e-5) {
    std::cout << "SUCCESS: Values are correct." << std::endl;
  } else {
    std::cout << "FAILURE: Value mismatch." << std::endl;
  }
}

template <unsigned NPanels = 32, unsigned NRows = 512, unsigned NCols = 1024, unsigned Bin = 2>
__global__ void max_binning_kernel(const float*  __restrict__ input,
                                   float* __restrict__ output) {

  constexpr unsigned out_panels { NPanels / Bin };
  constexpr unsigned out_rows { NRows / Bin };
  constexpr unsigned out_cols { NCols / Bin };
  constexpr unsigned NPixOut { out_panels * out_rows * out_cols };

  unsigned idx { blockIdx.x * blockDim.x + threadIdx.x };

  if (idx >= NPixOut) {
    return;
  }

  unsigned i_c { idx % out_cols };
  unsigned tmp { idx / out_cols };

  unsigned i_r { tmp % out_rows };

  unsigned i_p { tmp / out_rows };

  unsigned base_p { i_p * Bin };
  unsigned base_r { i_r * Bin };
  unsigned base_c { i_c * Bin };

  float bin_sum { 0.0f };

  for (unsigned dp = 0; dp < Bin; ++dp) {
    unsigned panel_offset { (base_p + dp) * (NRows * NCols) };
    for (unsigned dr = 0; dr < Bin; ++dr) {
      unsigned row_offset { (base_r + dr) * NCols };
      unsigned base_input_idx { panel_offset + row_offset + base_c };

      for (unsigned dc = 0; dc < Bin; ++dc) {
        bin_sum += input[base_input_idx + dc];
      }
    }
  }

  output[idx] = bin_sum;
}

void test_idealized_binning(ncarray::NCDevArrayView data,
                            std::vector<ssize_t>& res_shape,
                            ssize_t n_iter,
                            ssize_t n_panels,
                            ssize_t n_rows,
                            ssize_t n_cols,
                            ssize_t npix) {
  std::cout << "Will test idealized binning." << std::endl;
  std::cout << "Initializing "
            << "(" << n_panels << ", " << n_rows << ", " << n_cols << ")" << std::endl;

  ncarray::NCArray host_res(res_shape, ncarray::DType::float32);
  double total_time { 0.0 };

  ncarray::NCDevArray data_out(res_shape, ncarray::DType::float32);
  data_out.fill(1.0f); // Fill with 1.0

  for (ssize_t i = 0; i < n_iter; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    max_binning_kernel<<<8192, 256>>>(reinterpret_cast<float*>(data.data()),
                                      reinterpret_cast<float*>(data_out.data()));

    cudaDeviceSynchronize(); // superfluous since done internally
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    total_time += diff.count();

    data_out.copy_into_astype<float>(reinterpret_cast<float*>(host_res.data()));
  }
  std::cout << "Binning took: " << (total_time / n_iter) << " seconds" << std::endl;

  float first_val = *static_cast<float*>(host_res(0, 0, 0).data());
  std::cout << "Result[0,0,0]: " << first_val << " (Expected: 8.0)" << std::endl;

  if (std::abs(first_val - 8.0f) < 1e-5) {
    std::cout << "SUCCESS: Values are correct." << std::endl;
  } else {
    std::cout << "FAILURE: Value mismatch." << std::endl;
  }
}

int main(int argc, char* argv[]) {
  int c;

  ssize_t n_panels { 32 };
  ssize_t n_rows { 512 };
  ssize_t n_cols { 1024 };

  ssize_t npix { 2 };
  ssize_t npix_rest { 2 };

  ssize_t n_iter { 100 };

  while ((c = getopt(argc, argv, "p:r:c:n:b:t:")) != -1) {
    switch (c) {
    case 'h': {
      usage(argv[0]);
      exit(0);
    }
    case 'p': {
      n_panels = static_cast<ssize_t>(std::atoi(optarg));
      break;
    }
    case 'r': {
      n_rows = static_cast<ssize_t>(std::atoi(optarg));
      break;
    }
    case 'c': {
      n_cols = static_cast<ssize_t>(std::atoi(optarg));
      break;
    }
    case 'n': {
      n_iter = static_cast<ssize_t>(std::atoi(optarg));
      break;
    }
    case 't': {
      npix = static_cast<ssize_t>(std::atoi(optarg));
      break;
    }
    case 'b': {
      npix_rest = static_cast<ssize_t>(std::atoi(optarg));
      break;
    }
    default: {
      usage(argv[0]);
      exit(-1);
    }
    }
  }

  std::vector<ssize_t> shape { n_panels, n_rows, n_cols };

  // Result shape is dim/npix
  std::vector<ssize_t> res_shape {
    n_panels / npix,
    n_rows / npix_rest,
    n_cols / npix_rest
  };

  std::cout << "Starting binning (npix=" << npix << ")..." << std::endl;

  std::cout << "Testing idealized case." << std::endl
            << "----------------------------------------------------------" << std::endl;

  ncarray::NCDevArray ideal_data(shape, ncarray::DType::float32);
  ideal_data.fill(1.0f); // Fill with 1.0

  //test_idealized_binning(ideal_data.view(), res_shape, n_iter, n_panels, n_rows, n_cols, npix);

  // ------------------------------------------------------------------------------------

  std::cout << "----------------------------------------------------------" << std::endl
            << "Testing with contiguous data (float32)." << std::endl
            << "----------------------------------------------------------" << std::endl;

  ncarray::NCDevArray data(shape, ncarray::DType::float32);
  data = data.iota();

  test_fused_generic_add_bin(data.view(),
                             res_shape,
                             n_iter,
                             n_panels,
                             n_rows,
                             n_cols,
                             npix,
                             npix_rest);

  test_fused_generic_add_bin_decomposed(data.view(),
                                        res_shape,
                                        n_iter,
                                        n_panels,
                                        n_rows,
                                        n_cols,
                                        npix,
                                        npix_rest);

  test_fused_stencil_binning(data.view(),
                             res_shape,
                             n_iter,
                             n_panels,
                             n_rows,
                             n_cols,
                             npix,
                             npix_rest);

  test_fused_stencil_binning(data.view(),
                             res_shape,
                             n_iter,
                             n_panels,
                             n_rows,
                             n_cols,
                             npix,
                             npix_rest,
                             /*use_graph=*/true);


  test_reshape_sum_bin(data.view(),
                       res_shape,
                       n_iter,
                       n_panels,
                       n_rows,
                       n_cols,
                       npix,
                       npix_rest);

  test_copy_reshape_sum_bin(data.view(),
                            res_shape,
                            n_iter,
                            n_panels,
                            n_rows,
                            n_cols,
                            npix,
                            npix_rest);

  // ------------------------------------------------------------------------------------

  std::cout << "----------------------------------------------------------" << std::endl
            << "Testing with non-contiguous data (float32)." << std::endl
            << "----------------------------------------------------------" << std::endl;

  std::vector<void*> d_vecs(n_panels);

  for (ssize_t i = 0; i < n_panels; ++i) {
    cudaMalloc(&d_vecs[i], data.nbytes()/n_panels);
  }

  void** d_vecs_table;
  cudaMalloc(&d_vecs_table, n_panels * sizeof(void*));

  cudaMemcpy(d_vecs_table,
             d_vecs.data(),
             n_panels * sizeof(void*),
             cudaMemcpyHostToDevice);

  std::vector<ssize_t> strides(3, data.itemsize());
  strides[1] = shape[2] * strides[2];
  strides[0] = 1;

  ncarray::NCDevArrayView data_view(d_vecs_table,
                                    shape.size(),
                                    shape.data(),
                                    strides.data(),
                                    ncarray::DType::float32,
                                    0,
                                    false);
  data_view = data_view.iota();

  test_fused_generic_add_bin(data_view,
                             res_shape,
                             n_iter,
                             n_panels,
                             n_rows,
                             n_cols,
                             npix,
                             npix_rest);

  test_fused_generic_add_bin_decomposed(data_view,
                                        res_shape,
                                        n_iter,
                                        n_panels,
                                        n_rows,
                                        n_cols,
                                        npix,
                                        npix_rest);

  test_fused_stencil_binning(data_view,
                             res_shape,
                             n_iter,
                             n_panels,
                             n_rows,
                             n_cols,
                             npix,
                             npix_rest);

  test_fused_stencil_binning(data_view,
                             res_shape,
                             n_iter,
                             n_panels,
                             n_rows,
                             n_cols,
                             npix,
                             npix_rest,
                             /*use_graph=*/true);

  // Cannot reshape for non-contiguous array
  //test_reshape_sum_bin(data_view, res_shape, n_iter, n_panels, n_rows, n_cols, npix);

  test_copy_reshape_sum_bin(data_view,
                            res_shape,
                            n_iter,
                            n_panels,
                            n_rows,
                            n_cols,
                            npix,
                            npix_rest);

  return 0;
}
