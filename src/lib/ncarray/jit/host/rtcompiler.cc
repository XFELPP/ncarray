/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/jit/host/rtcompiler.hh"

#include "ncarray/dtype.hh"
#include "ncarray/jit/host/x86.hh"
#include "ncarray/jit/jit_utils.hh"
#include "ncarray/jit/path_utils.hh"
#include "ncarray/layout.hh"
#include "ncarray/op_code.hh"
#include "ncarray/op_traits.hh"

#include <asmjit/core.h>
#include <asmjit/x86.h>

#ifdef _WIN32
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>

#ifdef __linux__
#include <malloc.h> // For mallopt performance tuning
#endif
#endif

#include <cstdlib>
#include <iomanip>
#include <filesystem>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;

namespace ncarray {
  namespace host {
    /**
     * Compute a hash of the expression function/kernel based on types and metadata.
     */
    std::string compute_kernel_hash(DType dest_t,
                                    DType src_t,
                                    DType work_t,
                                    const SOArrayPolicy& dest_layout,
                                    const std::vector<Instruction>& instrs,
                                    const std::vector<SOArrayPolicy>& layouts,
                                    const std::vector<Scalar>& scalars,
                                    bool expr_is_soarr) {
      ssize_t ndim { dest_layout.ndim() };
      const ssize_t* final_shape { dest_layout.shape() };
      std::stringstream ss(std::ios::binary | std::ios::out | std::ios::in);
      // Serialize all the data types and array dimensions
      ss.write(reinterpret_cast<const char*>(&dest_t), sizeof(dest_t));
      ss.write(reinterpret_cast<const char*>(&src_t), sizeof(src_t));
      ss.write(reinterpret_cast<const char*>(&work_t), sizeof(work_t));
      ss.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
      ss.write(reinterpret_cast<const char*>(&expr_is_soarr), sizeof(expr_is_soarr));
      // Serialize the shape, strides, pointer info of the result/destination
      for (ssize_t d = 0; d < ndim; ++d) {
        ssize_t stride { dest_layout.stride(d) };
        ssize_t offset { 0 };
        if (expr_is_soarr) {
          offset = dest_layout.suboffset(d);
        } else {
          auto& ncl = reinterpret_cast<const NCOffsetsPolicy&>(dest_layout);
          offset = ncl.offset(d);
        }
        ss.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
        ss.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
      }
      ss.write(reinterpret_cast<const char*>(final_shape), ndim * sizeof(ssize_t));
      // Serialize the VM operations/instructions
      std::size_t n_instrs = instrs.size();
      ss.write(reinterpret_cast<const char*>(&n_instrs), sizeof(n_instrs));
      ss.write(reinterpret_cast<const char*>(instrs.data()), n_instrs * sizeof(Instruction));
      // Serialize the layout
      size_t n_layouts = layouts.size();
      ss.write(reinterpret_cast<const char*>(&n_layouts), sizeof(n_layouts));
      for (const auto& l : layouts) {
        for (ssize_t d = 0; d < ndim; ++d) {
          ssize_t stride = l.stride(d);
          ssize_t offset { 0 };
          if (expr_is_soarr) {
            offset = l.suboffset(d);
          } else {
            auto& ncl = reinterpret_cast<const NCOffsetsPolicy&>(l);
            offset = ncl.offset(d);
          }
          ss.write(reinterpret_cast<const char*>(&stride), sizeof(stride));
          ss.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
        }
      }
      // Serialize the constant types
      std::size_t n_scalars = scalars.size();
      ss.write(reinterpret_cast<const char*>(&n_scalars), sizeof(n_scalars));
      for (const auto& s : scalars) {
        std::size_t variant_idx = s.index();
        ss.write(reinterpret_cast<const char*>(&variant_idx), sizeof(variant_idx));
        auto serialize_op = [&](auto&& val) {
          ss.write(reinterpret_cast<const char*>(&val), sizeof(val));
        };
        std::visit(serialize_op, s);
      }
      // Return the hash
      return hash_to_hex(ss.str());
    }

    RuntimeCompiler::RuntimeCompiler() {}

    RuntimeCompiler& RuntimeCompiler::instance() {
#ifdef __linux__
      // Help performance by avoiding page faults for free/allocate cycles
      // On macOS, libsystem_malloc is more aggressive when reusing pages
      // On Windows, the NT heap is totally different. Both macOS and Windows shouldn't
      // need this change
      mallopt(M_MMAP_MAX, 0);
      mallopt(M_MMAP_THRESHOLD, 1024 * 1024 * 1024); // 1 GB
      mallopt(M_TRIM_THRESHOLD, -1);
#endif
      static RuntimeCompiler inst;
      return inst;
    }

    asmjit::TypeId RuntimeCompiler::dtype_to_typeid(DType dtype) {
      switch (dtype) {
      case DType::char_:
      case DType::int8: {
        return asmjit::TypeId::kInt8;
      }
      case DType::int16: {
        return asmjit::TypeId::kInt16;
      }
      case DType::int32: {
        return asmjit::TypeId::kInt32;
      }
      case DType::int64: {
        return asmjit::TypeId::kInt64;
      }
      case DType::bool_:
      case DType::uint8: {
        return asmjit::TypeId::kUInt8;
      }
      case DType::uint16: {
        return asmjit::TypeId::kUInt16;
      }
      case DType::uint32: {
        return asmjit::TypeId::kUInt32;
      }
      case DType::uint64: {
        return asmjit::TypeId::kUInt64;
      }
      case DType::float32: {
        return asmjit::TypeId::kFloat32;
      }
      case DType::float64: {
        return asmjit::TypeId::kFloat64;
      }
      case DType::float128: {
        return asmjit::TypeId::kFloat80;
      }
      // asmjit has packed SIMD types for SSE/AVX packed instructions
      case DType::complex64:
      case DType::vfloat2: {
        return asmjit::TypeId::kFloat32x2; // xmm (lower)
      }
      case DType::vfloat3: // No native 3 float types, use the 4 version and ignore 4th channel
      case DType::vfloat4: {
        return asmjit::TypeId::kFloat32x4; // xmm (all)
      }
      case DType::complex128:
      case DType::vdouble2: {
        return asmjit::TypeId::kFloat64x2; // xmm
      }
      case DType::vdouble3:
      case DType::vdouble4: {
        return asmjit::TypeId::kFloat64x4; // ymm
      }
      default: {
        return asmjit::TypeId::kVoid;
      }
      }
    }

    ExprKernelFunc RuntimeCompiler::get_expr_kernel(DType dest_t,
                                                    DType src_t,
                                                    DType work_t,
                                                    const SOArrayPolicy& dest_layout,
                                                    const std::vector<Instruction>& instrs,
                                                    const std::vector<SOArrayPolicy>& layouts,
                                                    const std::vector<Scalar>& scalars,
                                                    bool expr_is_soarr) {
      std::string k_id = compute_kernel_hash(dest_t,
                                             src_t,
                                             work_t,
                                             dest_layout,
                                             instrs,
                                             layouts,
                                             scalars,
                                             expr_is_soarr);
      if (m_kernel_cache.count(k_id)) {
        return m_kernel_cache[k_id];
      }

      fs::path cache_path = get_cache_dir() / (k_id + ".bin");

      if (!fs::exists(cache_path)) {
        // If it doesn't exist in the cache on disk, compile and store on disk
        // We then re-read it from disk afterwards
        compile_x86_expr_kernel(cache_path,
                                dest_t,
                                src_t,
                                work_t,
                                dest_layout,
                                instrs,
                                layouts,
                                scalars,
                                expr_is_soarr);
      }
      // Load function from disk
      ExprKernelFunc k_func =
        reinterpret_cast<ExprKernelFunc>(read_bin_file_to_exec_mem(cache_path));
      m_kernel_cache[k_id] = k_func;

      return m_kernel_cache[k_id];
    }

    void RuntimeCompiler::compile_x86_expr_kernel(fs::path cache_path,
                                                  DType dest_t,
                                                  DType src_t,
                                                  DType work_t,
                                                  const SOArrayPolicy& dest_layout,
                                                  const std::vector<Instruction>& instrs,
                                                  const std::vector<SOArrayPolicy>& layouts,
                                                  const std::vector<Scalar>& scalars,
                                                  bool expr_is_soarr) {
      asmjit::CodeHolder code;
      code.init(asmjit::Environment::host());
      asmjit::x86::Compiler cc(&code);

      // Function signature: void func(const SrcT** src_ptrs, DestT* dest_data)
      asmjit::FuncSignature func_sig;
      func_sig.set_ret(asmjit::TypeId::kVoid);
      func_sig.add_arg(asmjit::TypeId::kUIntPtr); // Source pointers
      func_sig.add_arg(asmjit::TypeId::kUIntPtr); // Destination pointer

      asmjit::FuncNode* func { cc.add_func(func_sig) };

      func->frame().set_avx_enabled();
      func->frame().set_avx512_enabled();

      asmjit::x86::Gp src_ptrs_reg { cc.new_gp_ptr("src_ptrs") };
      asmjit::x86::Gp dest_ptr_reg { cc.new_gp_ptr("dest_ptr") };

      func->set_arg(0, src_ptrs_reg);
      func->set_arg(1, dest_ptr_reg);

      ssize_t ndim { dest_layout.ndim() };
      const ssize_t* final_shape { dest_layout.shape() };

      // For IDX index generator, record the number of element strides
      std::vector<ssize_t> cum_nelem_strides(ndim);
      ssize_t current_stride { 1 };
      for (ssize_t dim = ndim - 1; dim >= 0; --dim) {
        cum_nelem_strides[dim] = current_stride;
        current_stride *= final_shape[dim];
      }

      // NOTE: The strategy here is to reduce redundant calculations and dereferencing
      //       in the innermost loop
      //       - Pull out the base pointer for each array view
      //       - Dereference and propagate offset at each level of the loop as you go
      // NOTE: Finally, in the innermost loop, x86 supports hardware scaling when
      //       you address a pointer like [base + index * scale] and scale is 1,2,4,8
      std::vector<asmjit::x86::Gp> base_ptrs(instrs.size());
      for (const auto& instr : instrs) {
        OpCode op { get_op(instr) };
        if (op != OpCode::LOAD_NCARR && op != OpCode::LOAD_SOARR) {
          continue;
        }
        int idx { get_index(instr) };
        base_ptrs[idx] = cc.new_gp_ptr();
        cc.mov(base_ptrs[idx], asmjit::x86::ptr(src_ptrs_reg, idx * sizeof(void*)));
      }

      // Hold the pointer for the view up to dimension `dim`: op_ptrs[view_idx][dim]
      std::vector<std::vector<asmjit::x86::Gp>> op_ptrs(instrs.size(),
                                                        std::vector<asmjit::x86::Gp>(ndim));

      // Likewise, track the destination pointer accumulation
      std::vector<asmjit::x86::Gp> dest_ptrs(ndim);

      // Setup loop limits and load into registers
      std::vector<asmjit::x86::Gp> loop_limits;
      // Setup loop indexing registers
      std::vector<asmjit::x86::Gp> loop_regs;
      // Setup loop labels and end points
      std::vector<asmjit::Label> loop_labels;
      std::vector<asmjit::Label> loop_ends;
      for (ssize_t dim = 0; dim < ndim; ++dim) {
        std::string id = "limit_" + std::to_string(dim);
        loop_limits.push_back(cc.new_gp(asmjit::TypeId::kInt64, id.c_str()));
        cc.mov(loop_limits[dim], final_shape[dim]);

        id = "idx_" + std::to_string(dim);
        loop_regs.push_back(cc.new_gp(asmjit::TypeId::kInt64, id.c_str()));

        loop_labels.push_back(cc.new_label());
        loop_ends.push_back(cc.new_label());

        // Setup a nested loop: Outer Dim --> Inner Dim --> ... --> Innermost
        cc.xor_(loop_regs[dim], loop_regs[dim]);
        if (dim == ndim - 1) {
          cc.align(asmjit::AlignMode::kCode, 16);
        }
        cc.bind(loop_labels[dim]);
        cc.cmp(loop_regs[dim], loop_limits[dim]);
        cc.jge(loop_ends[dim]);

        if (dim < ndim - 1) {
          // Propagate the destination pointer offset/strides etc.
          // Pickup where the last dimension left off (or base_ptr if dim == 0)
          dest_ptrs[dim] = cc.new_gp_ptr();
          if (dim == 0) {
            cc.mov(dest_ptrs[dim], dest_ptr_reg);
          } else {
            cc.mov(dest_ptrs[dim], dest_ptrs[dim - 1]);
          }

          if (!expr_is_soarr) {
            auto& ncl = reinterpret_cast<const NCOffsetsPolicy&>(dest_layout);
            if (ncl.is_pointer_axis(dim)) {
              // For NCOffsetsPolicy for pointer axes we derefence with [index + offset]
              x86::advance_ncoffsets_pointer_axis(cc,
                                                  dest_ptrs[dim],
                                                  loop_regs[dim],
                                                  ncl.offset(dim));
            } else {
              // Normal strided traversal (index * stride + offset)
              x86::advance_ncoffsets_strided_axis(cc,
                                                  dest_ptrs[dim],
                                                  loop_regs[dim],
                                                  ncl.stride(dim),
                                                  ncl.offset(dim));
            }
          } else {
            // For SOArrayPolicy we first do normal strided traversal (index * stride)
            x86::advance_soarray_axis(cc,
                                      dest_ptrs[dim],
                                      loop_regs[dim],
                                      dest_layout.stride(dim),
                                      dest_layout.suboffset(dim));
          }

          for (const auto& instr : instrs) {
            OpCode op { get_op(instr) };
            if (op != OpCode::LOAD_NCARR && op != OpCode::LOAD_SOARR) {
              continue;
            }
            int idx { get_index(instr) };
            op_ptrs[idx][dim] = cc.new_gp_ptr();

            // Pickup where the last dimension left off (or base_ptr if dim == 0)
            if (dim == 0) {
              cc.mov(op_ptrs[idx][dim], base_ptrs[idx]);
            } else {
              cc.mov(op_ptrs[idx][dim], op_ptrs[idx][dim - 1]);
            }

            if (!expr_is_soarr) {
              auto& ncl = reinterpret_cast<const NCOffsetsPolicy&>(layouts[idx]);
              if (ncl.is_pointer_axis(dim)) {
                // For NCOffsetsPolicy for pointer axes we derefence with [index + offset]
                x86::advance_ncoffsets_pointer_axis(cc,
                                                    op_ptrs[idx][dim],
                                                    loop_regs[dim],
                                                    ncl.offset(dim));
              } else {
                // Normal strided traversal (index * stride + offset)
                x86::advance_ncoffsets_strided_axis(cc,
                                                    op_ptrs[idx][dim],
                                                    loop_regs[dim],
                                                    ncl.stride(dim),
                                                    ncl.offset(dim));
              }
            } else {
              auto& sol = layouts[idx];
              // For SOArrayPolicy we first do normal strided traversal (index * stride)
              x86::advance_soarray_axis(cc,
                                        op_ptrs[idx][dim],
                                        loop_regs[dim],
                                        sol.stride(dim),
                                        sol.suboffset(dim));
            }
          }
        }
      }

      // Run the innermost loop body for evaluating the expression
      // ---------------------------------------------------------
      std::vector<asmjit::Reg> reg_stack;
      asmjit::TypeId type_id { dtype_to_typeid(work_t) };
      asmjit::TypeId src_type_id { dtype_to_typeid(src_t) };
      for (const auto& instr : instrs) {
        OpCode op { get_op(instr) };
        int idx { get_index(instr) };
        if (op == OpCode::LOAD_NCARR || op == OpCode::LOAD_SOARR) {
          asmjit::Reg v_reg { cc.new_reg<asmjit::Reg>(type_id) };
          asmjit::x86::Gp val_addr { cc.new_gp_ptr() };

          // Start from the pointer that was propagated with offsets up to ndim - 2
          // Or, if there's only 1 dimension, just use the base pointer
          if (ndim > 1) {
            cc.mov(val_addr, op_ptrs[idx][ndim - 2]);
          } else {
            cc.mov(val_addr, base_ptrs[idx]);
          }

          ssize_t inner_dim { ndim - 1 };
          x86::scaled_address_array(cc,
                                    val_addr,
                                    loop_regs[inner_dim],
                                    v_reg,
                                    inner_dim,
                                    layouts[idx],
                                    src_type_id,
                                    /*addr_is_sink=*/false,
                                    expr_is_soarr);

          // Load using the source type, but cast to our working type
          asmjit::Reg casted_reg {
            x86::cast_register(cc, v_reg, src_type_id, type_id)
          };

          // Push to stack and advance
          reg_stack.push_back(casted_reg);
        } else if (op == OpCode::LOAD_CONST) {
          asmjit::Reg s_reg { x86::load_constant(cc, scalars[idx], type_id) };
          reg_stack.push_back(s_reg);
        } else if (op == OpCode::IDX) {
          asmjit::x86::Gp lin_idx_reg { cc.new_gp(asmjit::TypeId::kInt64) };
          cc.mov(lin_idx_reg, 0);
          for (ssize_t dim = 0; dim < ndim; ++dim) {
            ssize_t stride { cum_nelem_strides[dim] };
            if (stride == 1) {
              cc.add(lin_idx_reg, loop_regs[dim]);
            } else {
              asmjit::x86::Gp term { cc.new_gp(asmjit::TypeId::kInt64) };
              cc.mov(term, loop_regs[dim]);
              cc.imul(term, stride);
              cc.add(lin_idx_reg, term);
            }
          }

          // Convert the IDX ravel index to the working type and push to stack
          asmjit::Reg v_reg {
            x86::cast_register(cc, lin_idx_reg.as<asmjit::Reg>(), asmjit::TypeId::kInt64, type_id)
          };
          reg_stack.push_back(v_reg);
        } else if (static_cast<int>(op) < static_cast<int>(OpCode::ADD)) {
          // NOTE: OpCodes are ordered, so comparisons of value work.
          //       - First are loads
          //       - Unary Ops
          //       - Arithemtic (Binary)
          //       - Comparisons (Binary)
          //       - Logical (Binary)

          // Unary Operations
          // -----------------
          // Get the operand
          asmjit::Reg operand { reg_stack.back() };
          reg_stack.pop_back();

          // Allocate new register for result and perform operation
          asmjit::Reg res { cc.new_reg<asmjit::Reg>(type_id) };
          x86::unary_operation(cc, op, type_id, operand, res);

          // Push result back onto stack
          reg_stack.push_back(res);
        } else {
          // Binary operations
          // -----------------
          // Get left and right operands
          asmjit::Reg r { reg_stack.back() };
          reg_stack.pop_back();

          asmjit::Reg l { reg_stack.back() };
          reg_stack.pop_back();

          // Allocate a new virtual register for the result of the operation
          asmjit::Reg res { cc.new_reg<asmjit::Reg>(type_id) };

          if (static_cast<int>(op) < static_cast<int>(OpCode::EQ)) {
            // Arithmetic
            if (asmjit::TypeUtils::is_int(type_id) && (op == OpCode::DIV || op == OpCode::MOD)) {
              // Integer division and modulo has to be done in specific-registers
              x86::integer_div_mod(cc, op, type_id, l, r, res);
            } else {
              cc.emit(x86::get_move_inst(type_id), res, l);                  // res = l
              cc.emit(x86::get_binary_arithmetic_inst(op, type_id), res, r); // res = res op r
            }
          } else if (static_cast<int>(op) < static_cast<int>(OpCode::LAND)) {
            // Comparisons
            x86::binary_compare(cc, op, type_id, l, r, res);
          }
          reg_stack.push_back(res);
        }
      }
      asmjit::Reg final_res { reg_stack.back() };

      asmjit::TypeId dest_type_id { dtype_to_typeid(dest_t) };
      asmjit::Reg casted_res { x86::cast_register(cc, final_res, type_id, dest_type_id) };

      // Resolve the location for storing the result in the destination
      asmjit::x86::Gp dest_addr { cc.new_gp_ptr("dest_addr") };
      if (ndim > 1) {
        cc.mov(dest_addr, dest_ptrs[ndim - 2]);
      } else {
        cc.mov(dest_addr, dest_ptr_reg);
      }

      ssize_t inner_dim { ndim - 1 };
      x86::scaled_address_array(cc,
                                dest_addr,
                                loop_regs[inner_dim],
                                casted_res,
                                inner_dim,
                                dest_layout,
                                dest_type_id,
                                /*addr_is_sink=*/true,
                                expr_is_soarr);

      // ---------------------------------------------------------
      // End inner loop body

      // Loop increments and jumps
      // NOTE: Go from the innermost --> out!
      for (ssize_t dim = ndim - 1; dim >= 0; --dim) {
        cc.add(loop_regs[dim], 1);
        cc.jmp(loop_labels[dim]);
        cc.bind(loop_ends[dim]);
      }

      cc.ret();
      cc.end_func();
      cc.finalize();

      // Write the function to the cache
      asmjit::CodeBuffer& buffer { code.section_by_id(0)->buffer() };
      std::size_t k_size { buffer.size() };
      const std::uint8_t* k_data { buffer.data() };
      write_file(cache_path, k_size, k_data);
    }
  } // namespace host
} // namespace ncarray
