#include "ncarray/ncarrays.hh"
#include "ncarray/jit/host/rtcompiler.hh"

#include <asmjit/asmjit.h>

#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace asmjit;
using namespace ncarray;

void usage(char* progname) {
  std::cerr << "Usage: " << progname
            << " -p <n_panels> -r <n_rows> -c <n_cols> -t <npix_bin> -b <npix_bin> -n <niter>"
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

void test_vm(ncarray::NCArrayView data,
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

  double total_build_time { 0.0 };
  double total_exec_time { 0.0 };

  using sl = ncarray::Slice;

  float scale_factor { 1.0f / (npix * npix_rest * npix_rest) };

  ncarray::NCArray res(res_shape, ncarray::DType::float32);

  for (ssize_t i = 0; i < n_iter; ++i) {
    auto build_start = std::chrono::high_resolution_clock::now();

    ncarray::ExprMVNode<ncarray::HostTag> expr;
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
    total_build_time += build_diff.count();
    total_exec_time += exec_diff.count();
  }

  std::cout << "Binning build phase took: "
            << (total_build_time / (n_iter - 1)) * 1e6 << " microseconds" << std::endl
            << "Binning exec phase took: "
            << (total_exec_time / (n_iter - 1)) * 1e6 << " microseconds" << std::endl;

  auto first_proxy = res[{0, 0, 0}];
  float& first_val = first_proxy;
  auto p_end { static_cast<std::uint64_t>(res_shape[0] - 1) };
  auto r_end { static_cast<std::uint64_t>(res_shape[1] - 1) };
  auto c_end { static_cast<std::uint64_t>(res_shape[2] - 1) };
  auto last_proxy = res[{ p_end, r_end, c_end }];
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

typedef void (*BinningKernelFunc)(const float** src_ptrs, float* dest_data);
void test_asmjit(ncarray::NCArrayView data,
                 std::vector<ssize_t>& res_shape,
                 ssize_t n_iter,
                 ssize_t n_panels,
                 ssize_t n_rows,
                 ssize_t n_cols,
                 ssize_t npix,
                 ssize_t npix_rest)
{
  std::cout << "\n----------------------------------------------------------" << std::endl;
  std::cout << "Testing fused sum of views with JIT (asmjit)." << std::endl;
  std::cout << "----------------------------------------------------------" << std::endl;
  JitRuntime rt;
  CodeHolder code;
  code.init(rt.environment());
  x86::Compiler cc(&code);
  // Define function signature: void func(const float** src_ptrs, float* dest_data)
  FuncSignature sig;
  sig.set_ret(TypeId::kVoid);
  sig.add_arg(TypeId::kUIntPtr); // src_ptrs
  sig.add_arg(TypeId::kUIntPtr); // dest_data
  FuncNode* func = cc.add_func(sig);
  x86::Gp src_ptrs_reg = cc.new_gp_ptr("src_ptrs");
  x86::Gp dest_data_reg = cc.new_gp_ptr("dest_data");
  func->set_arg(0, src_ptrs_reg);
  func->set_arg(1, dest_data_reg);

  // Generate the 3D loop parameters
  ssize_t out_panels = res_shape[0];
  ssize_t out_rows = res_shape[1];
  ssize_t out_cols = res_shape[2];
  ssize_t num_views = npix * npix_rest * npix_rest;
  float scale_factor = 1.0f / (npix * npix_rest * npix_rest);

  // Generate layouts for all the sliced views to compute offsets

  std::vector<ncarray::NCOffsetsPolicy> layouts;
  using sl = ncarray::Slice;
  for (ssize_t p = 0; p < npix; ++p) {
    for (ssize_t r = 0; r < npix_rest; ++r) {
      for (ssize_t c = 0; c < npix_rest; ++c) {
        auto view = data(sl(p, n_panels, npix), sl(r, n_rows, npix_rest), sl(c, n_cols, npix_rest));
        layouts.push_back(static_cast<const ncarray::NCOffsetsPolicy&>(view));
      }
    }
  }

  // Load loop limits into registers (since cmp r64, imm64 is not valid)
  x86::Gp limit_p = cc.new_gp(TypeId::kInt64, "limit_p");
  x86::Gp limit_r = cc.new_gp(TypeId::kInt64, "limit_r");
  x86::Gp limit_c = cc.new_gp(TypeId::kInt64, "limit_c");
  cc.mov(limit_p, out_panels);
  cc.mov(limit_r, out_rows);
  cc.mov(limit_c, out_cols);

  ncarray::NCArray res(res_shape, ncarray::DType::float32);
  auto dest_layout = static_cast<ncarray::NCOffsetsPolicy>(res.view());
  // Loop registers
  x86::Gp p_idx = cc.new_gp(TypeId::kInt64, "p_idx");
  x86::Gp r_idx = cc.new_gp(TypeId::kInt64, "r_idx");
  x86::Gp c_idx = cc.new_gp(TypeId::kInt64, "c_idx");
  x86::Gp addr = cc.new_gp_ptr("addr");
  x86::Gp stride = cc.new_gp(TypeId::kInt64, "stride");
  x86::Gp term = cc.new_gp(TypeId::kInt64, "term");
  x86::Vec val = cc.new_xmm("val");
  Label loop_p = cc.new_label();
  Label end_p = cc.new_label();
  Label loop_r = cc.new_label();
  Label end_r = cc.new_label();
  Label loop_c = cc.new_label();
  Label end_c = cc.new_label();

  // Nested Loop: Panel -> Row -> Column
  cc.xor_(p_idx, p_idx);
  cc.bind(loop_p);
  cc.cmp(p_idx, limit_p);
  cc.jge(end_p);
  cc.xor_(r_idx, r_idx);
  cc.bind(loop_r);
  cc.cmp(r_idx, limit_r);
  cc.jge(end_r);
  cc.xor_(c_idx, c_idx);
  cc.bind(loop_c);
  cc.cmp(c_idx, limit_c);
  cc.jge(end_c);

  // --- Innermost Loop Body ---
  // Load and sum all views for the current coordinate {p_idx, r_idx, c_idx}
  x86::Vec accum = cc.new_xmm("accum");
  cc.xorps(accum, accum); // accum = 0.0f
  // Define coords here so it is in scope for both src and dest pointer resolution
  std::vector<x86::Gp> coords = {p_idx, r_idx, c_idx};
  for (int v = 0; v < num_views; ++v) {
    // Load view base: addr = src_ptrs[v]
    cc.mov(addr, x86::ptr(src_ptrs_reg, v * 8));
    // Resolve offset
    for (int dim = 0; dim < 3; ++dim) {
      cc.mov(stride, layouts[v].stride(dim));
      cc.mov(term, coords[dim]); // Copy register for multiplication
      cc.imul(term, stride);     // term = coords[dim] * stride (2-operand imul)
      cc.add(addr, term);

      cc.mov(term, layouts[v].offset(dim)); // Load 64-bit offset into register
      cc.add(addr, term);                   // Register-register add
    }
    cc.movss(val, x86::ptr(addr)); // Load float value
    cc.addss(accum, val);          // accum += val
  }
  // Scale: accum *= scale_factor
  x86::Vec scale_reg = cc.new_xmm("scale");
  union {
    float f;
    uint32_t u;
  } u;
  u.f = scale_factor;
  x86::Gp tmp_gp = cc.new_gp(TypeId::kInt32, "tmp_gp");
  cc.mov(tmp_gp, u.u);
  cc.movd(scale_reg, tmp_gp);
  cc.mulss(accum, scale_reg);
  // Resolve dest pointer

  cc.mov(addr, dest_data_reg);
  for (int dim = 0; dim < 3; ++dim) {
    cc.mov(stride, dest_layout.stride(dim));
    cc.mov(term, coords[dim]); // Copy register for multiplication
    cc.imul(term, stride);     // term = coords[dim] * stride
    cc.add(addr, term);
  }
  // Store result

  cc.movss(x86::ptr(addr), accum);
  // Loop increments and jumps
  cc.add(c_idx, 1);
  cc.jmp(loop_c);
  cc.bind(end_c);
  cc.add(r_idx, 1);
  cc.jmp(loop_r);
  cc.bind(end_r);
  cc.add(p_idx, 1);
  cc.jmp(loop_p);
  cc.bind(end_p);
  cc.ret();
  cc.end_func();
  cc.finalize();
  BinningKernelFunc kernel_fn = nullptr;
  rt.add(&kernel_fn, &code);
  // Setup input pointers for each view
  std::vector<const float*> src_ptrs(num_views);
  for (int v = 0; v < num_views; ++v) {
    // Collect the advanced base data pointer for this slice
    ssize_t p = v / (npix_rest * npix_rest);
    ssize_t r = (v % (npix_rest * npix_rest)) / npix_rest;
    ssize_t c = v % npix_rest;
    auto view = data(sl(p, n_panels, npix), sl(r, n_rows, npix_rest), sl(c, n_cols, npix_rest));
    src_ptrs[v] = reinterpret_cast<const float*>(view.data());
  }
  float* dest_data = reinterpret_cast<float*>(res.data());
  double total_time { 0.0 };
  for (ssize_t i = 0; i < n_iter; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    kernel_fn(src_ptrs.data(), dest_data);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    if (i > 0) total_time += diff.count();
  }
  std::cout << "Asmjit JIT phase took: "
            << (total_time / (n_iter - 1)) * 1e6 << " microseconds" << std::endl;
  rt.release(kernel_fn);
  auto first_proxy = res[{0, 0, 0}];
  float& first_val = first_proxy;
  auto p_end { static_cast<std::uint64_t>(res_shape[0] - 1) };
  auto r_end { static_cast<std::uint64_t>(res_shape[1] - 1) };
  auto c_end { static_cast<std::uint64_t>(res_shape[2] - 1) };
  auto last_proxy = res[{p_end, r_end, c_end}];
  float& last_val = last_proxy;

  float expected{(npix * npix_rest * npix_rest) * scale_factor};

  std::cout << "Result[0, 0, 0]: " << first_val << " (Expected: " << expected << ")" << std::endl
            << "Result[" << p_end << ", " << r_end << ", " << c_end << "]: " << last_val
            << " (Expected: " << expected << ")" << std::endl;
}

typedef void (*BinningKernelFunc)(const float** src_ptrs, float* dest_data);
void test_asmjit_opt(ncarray::NCArrayView data,
                     std::vector<ssize_t>& res_shape,
                     ssize_t n_iter,
                     ssize_t n_panels,
                     ssize_t n_rows,
                     ssize_t n_cols,
                     ssize_t npix,
                     ssize_t npix_rest)
{
  std::cout << "\n----------------------------------------------------------" << std::endl;
  std::cout << "Testing fused sum of views with JIT (asmjit)." << std::endl;
  std::cout << "----------------------------------------------------------" << std::endl;
  JitRuntime rt;
  CodeHolder code;
  code.init(rt.environment());
  x86::Compiler cc(&code);
  // Define function signature: void func(const float** src_ptrs, float* dest_data)
  FuncSignature sig;
  sig.set_ret(TypeId::kVoid);
  sig.add_arg(TypeId::kUIntPtr); // src_ptrs
  sig.add_arg(TypeId::kUIntPtr); // dest_data
  FuncNode* func = cc.add_func(sig);
  x86::Gp src_ptrs_reg = cc.new_gp_ptr("src_ptrs");
  x86::Gp dest_data_reg = cc.new_gp_ptr("dest_data");
  func->set_arg(0, src_ptrs_reg);
  func->set_arg(1, dest_data_reg);
  // Generate the 3D loop parameters
  ssize_t out_panels = res_shape[0];
  ssize_t out_rows = res_shape[1];
  ssize_t out_cols = res_shape[2];
  ssize_t num_views = npix * npix_rest * npix_rest;
  float scale_factor = 1.0f / (npix * npix_rest * npix_rest);
  // Generate layouts for all the sliced views to compute offsets
  std::vector<ncarray::NCOffsetsPolicy> layouts;
  using sl = ncarray::Slice;
  for (ssize_t p = 0; p < npix; ++p) {
    for (ssize_t r = 0; r < npix_rest; ++r) {
      for (ssize_t c = 0; c < npix_rest; ++c) {
        auto view = data(sl(p, n_panels, npix), sl(r, n_rows, npix_rest), sl(c, n_cols, npix_rest));
        layouts.push_back(static_cast<const ncarray::NCOffsetsPolicy&>(view));
      }
    }
  }
  // Load loop limits into registers (since cmp r64, imm64 is not valid)
  x86::Gp limit_p = cc.new_gp(TypeId::kInt64, "limit_p");
  x86::Gp limit_r = cc.new_gp(TypeId::kInt64, "limit_r");
  x86::Gp limit_c = cc.new_gp(TypeId::kInt64, "limit_c");
  cc.mov(limit_p, out_panels);
  cc.mov(limit_r, out_rows);
  cc.mov(limit_c, out_cols);
  ncarray::NCArray res(res_shape, ncarray::DType::float32);
  auto dest_layout = static_cast<ncarray::NCOffsetsPolicy>(res.view());
  // Loop registers
  x86::Gp p_idx = cc.new_gp(TypeId::kInt64, "p_idx");
  x86::Gp r_idx = cc.new_gp(TypeId::kInt64, "r_idx");
  x86::Gp c_idx = cc.new_gp(TypeId::kInt64, "c_idx");
  x86::Gp addr = cc.new_gp_ptr("addr");
  x86::Gp stride = cc.new_gp(TypeId::kInt64, "stride");
  x86::Gp term = cc.new_gp(TypeId::kInt64, "term");
  x86::Vec val = cc.new_xmm("val");
  Label loop_p = cc.new_label();
  Label end_p = cc.new_label();
  Label loop_r = cc.new_label();
  Label end_r = cc.new_label();
  Label loop_c = cc.new_label();
  Label end_c = cc.new_label();
  // Nested Loop: Panel -> Row -> Column
  cc.xor_(p_idx, p_idx);
  cc.bind(loop_p);
  cc.cmp(p_idx, limit_p);
  cc.jge(end_p);
  cc.xor_(r_idx, r_idx);
  cc.bind(loop_r);
  cc.cmp(r_idx, limit_r);
  cc.jge(end_r);
  cc.xor_(c_idx, c_idx);
  cc.bind(loop_c);
  cc.cmp(c_idx, limit_c);
  cc.jge(end_c);
  // --- Innermost Loop Body ---
  // Load and sum all views for the current coordinate {p_idx, r_idx, c_idx}
  x86::Vec accum = cc.new_xmm("accum");
  cc.xorps(accum, accum); // accum = 0.0f

  // coord_offset = p_idx * stride(0) + r_idx * stride(1) + c_idx * stride(2)
  x86::Gp coord_offset = cc.new_gp(TypeId::kInt64, "coord_offset");

  // dim = 0
  cc.mov(coord_offset, p_idx);
  cc.imul(coord_offset, layouts[0].stride(0));

  // dim = 1
  cc.mov(term, r_idx);
  cc.imul(term, layouts[0].stride(1));
  cc.add(coord_offset, term);

  // dim = 2
  cc.mov(term, c_idx);
  cc.imul(term, layouts[0].stride(2));
  cc.add(coord_offset, term);
  for (int v = 0; v < num_views; ++v) {
    // Load view base: addr = src_ptrs[v]
    cc.mov(addr, x86::ptr(src_ptrs_reg, v * 8));
    cc.add(addr, coord_offset);
    // Pre-calculate the total offset of the slice at compile time
    ssize_t total_slice_offset = layouts[v].offset(0) + layouts[v].offset(1) + layouts[v].offset(2);
    if (total_slice_offset != 0) {
      cc.mov(term, total_slice_offset);
      cc.add(addr, term);
    }
    cc.movss(val, x86::ptr(addr)); // Load float value
    cc.addss(accum, val);          // accum += val
  }
  // Scale: accum *= scale_factor
  x86::Vec scale_reg = cc.new_xmm("scale");
  union {
    float f;
    uint32_t u;
  } u;
  u.f = scale_factor;
  x86::Gp tmp_gp = cc.new_gp(TypeId::kInt32, "tmp_gp");
  cc.mov(tmp_gp, u.u);
  cc.movd(scale_reg, tmp_gp);
  cc.mulss(accum, scale_reg);
  // Resolve dest pointer
  cc.mov(addr, dest_data_reg);
  for (int dim = 0; dim < 3; ++dim) {
    cc.mov(stride, dest_layout.stride(dim));
    cc.mov(term, c_idx); // Copy coordinate register (using c_idx for dim 2, r_idx for dim 1, p_idx for dim 0)
    if (dim == 0) cc.mov(term, p_idx);
    else if (dim == 1) cc.mov(term, r_idx);
    cc.imul(term, stride);     // term = coords[dim] * stride
    cc.add(addr, term);
  }
  // Store result
  cc.movss(x86::ptr(addr), accum);
  // Loop increments and jumps
  cc.add(c_idx, 1);
  cc.jmp(loop_c);
  cc.bind(end_c);
  cc.add(r_idx, 1);
  cc.jmp(loop_r);
  cc.bind(end_r);
  cc.add(p_idx, 1);
  cc.jmp(loop_p);
  cc.bind(end_p);
  cc.ret();
  cc.end_func();
  cc.finalize();
  BinningKernelFunc kernel_fn = nullptr;
  rt.add(&kernel_fn, &code);
  // Setup input pointers for each view
  std::vector<const float*> src_ptrs(num_views);
  for (int v = 0; v < num_views; ++v) {
    // Collect the advanced base data pointer for this slice
    ssize_t p = v / (npix_rest * npix_rest);
    ssize_t r = (v % (npix_rest * npix_rest)) / npix_rest;
    ssize_t c = v % npix_rest;
    auto view = data(sl(p, n_panels, npix), sl(r, n_rows, npix_rest), sl(c, n_cols, npix_rest));
    src_ptrs[v] = reinterpret_cast<const float*>(view.data());
  }
  float* dest_data = reinterpret_cast<float*>(res.data());
  double total_time { 0.0 };
  for (ssize_t i = 0; i < n_iter; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    kernel_fn(src_ptrs.data(), dest_data);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    if (i > 0) total_time += diff.count();
  }
  std::cout << "Asmjit JIT phase took: "
            << (total_time / (n_iter - 1)) * 1e6 << " microseconds" << std::endl;
  rt.release(kernel_fn);
  auto first_proxy = res[{0, 0, 0}];
  float& first_val = first_proxy;
  auto p_end { static_cast<std::uint64_t>(res_shape[0] - 1) };
  auto r_end { static_cast<std::uint64_t>(res_shape[1] - 1) };
  auto c_end { static_cast<std::uint64_t>(res_shape[2] - 1) };
  auto last_proxy = res[{p_end, r_end, c_end}];
  float& last_val = last_proxy;
  float expected{(npix * npix_rest * npix_rest) * scale_factor};
  std::cout << "Result[0, 0, 0]: " << first_val << " (Expected: " << expected << ")" << std::endl
            << "Result[" << p_end << ", " << r_end << ", " << c_end << "]: " << last_val
            << " (Expected: " << expected << ")" << std::endl;
}

void test_runtime_compiler(ncarray::NCArrayView data,
                           std::vector<ssize_t>& res_shape,
                           ssize_t n_iter,
                           ssize_t n_panels,
                           ssize_t n_rows,
                           ssize_t n_cols,
                           ssize_t npix,
                           ssize_t npix_rest)
{
  std::cout << "\n----------------------------------------------------------" << std::endl;
  std::cout << "Testing sum of views with RuntimeCompiler JIT Implementation" << std::endl;
  std::cout << "----------------------------------------------------------" << std::endl;

  ncarray::NCArray res(res_shape, ncarray::DType::float32);

  using sl = ncarray::Slice;

  float scale_factor { 1.0f / (npix * npix_rest * npix_rest) };

  auto build_start = std::chrono::high_resolution_clock::now();

  ncarray::ExprMVNode<ncarray::HostTag> expr;
  for (ssize_t p = 0; p < npix; ++p) {
    for (ssize_t r = 0; r < npix_rest; ++r) {
      for (ssize_t c = 0; c < npix_rest; ++c) {
        if (p == 0 && r == 0 && c == 0) {
          expr.build_node(data(sl(p, n_panels, npix), sl(r, n_rows, npix_rest), sl(c, n_cols, npix_rest)));
        } else {
          expr = expr +
            data(sl(p, n_panels, npix), sl(r, n_rows, npix_rest), sl(c, n_cols, npix_rest));
        }
      }
    }
  }
  expr = expr * scale_factor;

  DType src_dtype { expr.dtypes.empty() ? expr.work_dtype : expr.dtypes[0] };
  DType work_dtype { expr.work_dtype };
  auto kernel =
    ncarray::host::RuntimeCompiler::instance().get_expr_kernel(res.dtype(),
                                                               src_dtype,
                                                               work_dtype,
                                                               expr.ndim(),
                                                               expr.shape(),
                                                               expr.instrs,
                                                               expr.layouts,
                                                               expr.scalars,
                                                               expr.soarray);
  // Setup input pointers for each view
  std::size_t num_views { expr.layouts.size() };
  std::vector<const float*> src_ptrs(num_views);
  for (int v = 0; v < num_views; ++v) {
    // Collect the advanced base data pointer for this slice
    ssize_t p = v / (npix_rest * npix_rest);
    ssize_t r = (v % (npix_rest * npix_rest)) / npix_rest;
    ssize_t c = v % npix_rest;
    auto view = data(sl(p, n_panels, npix), sl(r, n_rows, npix_rest), sl(c, n_cols, npix_rest));
    src_ptrs[v] = reinterpret_cast<const float*>(view.data());
  }
  float* dest_data = reinterpret_cast<float*>(res.data());
  double total_time { 0.0 };
  for (ssize_t i = 0; i < n_iter; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    kernel(reinterpret_cast<const void**>(src_ptrs.data()), dest_data);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    if (i > 0)
      total_time += diff.count();
  }
  std::cout << "RuntimeCompiler took: " << (total_time / (n_iter - 1)) * 1e6 << " microseconds"
            << std::endl;
  auto first_proxy = res[{0, 0, 0}];
  float& first_val = first_proxy;
  auto p_end { static_cast<std::uint64_t>(res_shape[0] - 1) };
  auto r_end { static_cast<std::uint64_t>(res_shape[1] - 1) };
  auto c_end { static_cast<std::uint64_t>(res_shape[2] - 1) };
  auto last_proxy = res[{p_end, r_end, c_end}];
  float& last_val = last_proxy;
  float expected { (npix * npix_rest * npix_rest) * scale_factor };
  std::cout << "Result[0, 0, 0]: " << first_val << " (Expected: " << expected << ")" << std::endl
            << "Result[" << p_end << ", " << r_end << ", " << c_end << "]: " << last_val
            << " (Expected: " << expected << ")" << std::endl;
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
  std::vector<ssize_t> res_shape {
    n_panels / npix,
    n_rows / npix_rest,
    n_cols / npix_rest
  };

  std::cout << "Starting binning (npix=" << npix << ")..." << std::endl;

  std::cout << "----------------------------------------------------------" << std::endl
            << "Testing with contiguous data (float32)." << std::endl
            << "----------------------------------------------------------" << std::endl;

  ncarray::NCArray data(shape, ncarray::DType::float32);
  data = data.iota();

  test_vm(data.view(),
          res_shape,
          n_iter,
          n_panels,
          n_rows,
          n_cols,
          npix,
          npix_rest);

  test_asmjit(data.view(),
              res_shape,
              n_iter,
              n_panels,
              n_rows,
              n_cols,
              npix,
              npix_rest);

  test_asmjit_opt(data.view(),
                  res_shape,
                  n_iter,
                  n_panels,
                  n_rows,
                  n_cols,
                  npix,
                  npix_rest);

  test_runtime_compiler(data.view(),
                        res_shape,
                        n_iter,
                        n_panels,
                        n_rows,
                        n_cols,
                        npix,
                        npix_rest);
}
