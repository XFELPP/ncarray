/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/jit/host/rtcompiler.hh"

#include "ncarray/dtype.hh"
#include "ncarray/jit/host/jit_utils.hh"
#include "ncarray/jit/jit_utils.hh"
#include "ncarray/jit/path_utils.hh"
#include "ncarray/layout.hh"
#include "ncarray/op_code.hh"
#include "ncarray/op_traits.hh"

#include <asmjit/asmjit.h>

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
                                    ssize_t ndim,
                                    const ssize_t* final_shape,
                                    const std::vector<Instruction>& instrs,
                                    const std::vector<SOArrayPolicy>& layouts,
                                    const std::vector<Scalar>& scalars,
                                    bool expr_is_soarr) {
    std::stringstream ss(std::ios::binary | std::ios::out | std::ios::in);
    // Serialize all the data types and array dimensions
    ss.write(reinterpret_cast<const char*>(&dest_t), sizeof(dest_t));
    ss.write(reinterpret_cast<const char*>(&src_t), sizeof(src_t));
    ss.write(reinterpret_cast<const char*>(&work_t), sizeof(work_t));
    ss.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
    ss.write(reinterpret_cast<const char*>(&expr_is_soarr), sizeof(expr_is_soarr));
    // Serialize the shape
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
      default: {
        return asmjit::TypeId::kVoid;
      }
      }
    }

    ExprKernelFunc RuntimeCompiler::get_expr_kernel(DType dest_t,
                                                    DType src_t,
                                                    DType work_t,
                                                    ssize_t ndim,
                                                    const ssize_t* final_shape,
                                                    const std::vector<Instruction>& instrs,
                                                    const std::vector<SOArrayPolicy>& layouts,
                                                    const std::vector<Scalar>& scalars,
                                                    bool expr_is_soarr) {
      std::string k_id = compute_kernel_hash(dest_t,
                                             src_t,
                                             work_t,
                                             ndim,
                                             final_shape,
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
                                ndim,
                                final_shape,
                                instrs,
                                layouts,
                                scalars,
                                expr_is_soarr);
      }
      // Load function from disk
      ExprKernelFunc k_func = load_kernel_from_disk(cache_path);
      m_kernel_cache[k_id] = k_func;

      return m_kernel_cache[k_id];
    }

    void RuntimeCompiler::compile_x86_expr_kernel(fs::path cache_path,
                                                  DType dest_t,
                                                  DType src_t,
                                                  DType work_t,
                                                  ssize_t ndim,
                                                  const ssize_t* final_shape,
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

      asmjit::FuncNode* func = cc.add_func(func_sig);
      asmjit::x86::Gp src_ptrs_reg = cc.new_gp_ptr("src_ptrs");
      asmjit::x86::Gp dest_ptr_reg = cc.new_gp_ptr("dest_ptr");

      func->set_arg(0, src_ptrs_reg);
      func->set_arg(1, dest_ptr_reg);

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
        cc.bind(loop_labels[dim]);
        cc.cmp(loop_regs[dim], loop_limits[dim]);
        cc.jge(loop_ends[dim]);
      }

      asmjit::x86::Gp term = cc.new_gp_ptr("term");

      // Run the innermost loop body for evaluating the expression
      // ---------------------------------------------------------
      std::vector<asmjit::Reg> reg_stack;
      asmjit::TypeId type_id = dtype_to_typeid(work_t);
      int v_ptr = 0;
      for (const auto& instr : instrs) {
        OpCode op = get_op(instr);
        int idx = get_index(instr);
        if (op == OpCode::LOAD_NCARR || op == OpCode::LOAD_SOARR) {
          // Allocate virtual vector register for the array element
          asmjit::Reg v_reg = cc.new_reg<asmjit::Reg>(type_id);

          // Get the address (stride and offset calculation hoisted outside this loop)
          asmjit::x86::Gp val_addr = cc.new_gp_ptr();
          cc.mov(val_addr, asmjit::x86::ptr(src_ptrs_reg, idx * sizeof(void*)));
          asmjit::x86::Gp coord_offset = cc.new_gp(asmjit::TypeId::kInt64);
          cc.xor_(coord_offset, coord_offset);
          for (ssize_t dim = 0; dim < ndim; ++dim) {
            asmjit::x86::Gp term_dim = cc.new_gp(asmjit::TypeId::kInt64);
            cc.mov(term_dim, loop_regs[dim]);
            cc.imul(term_dim, layouts[idx].stride(dim));
            cc.add(coord_offset, term_dim);
          }
          cc.add(val_addr, coord_offset);

          // Get the dynamic offset from summation over dimensions
          ssize_t total_slice_offset { 0 };
          for (ssize_t dim = 0; dim < ndim; ++dim) {
            if (!expr_is_soarr) {
              auto& ncl = reinterpret_cast<const NCOffsetsPolicy&>(layouts[idx]);
              total_slice_offset += ncl.offset(dim);
            } else {
              total_slice_offset += layouts[v_ptr].suboffset(dim);
            }
          }

          if (total_slice_offset != 0) {
            cc.mov(term, total_slice_offset);
            cc.add(val_addr, term);
          }

          // Setup appropriate load for type
          cc.emit(get_move_x86_inst(type_id), v_reg, asmjit::x86::ptr(val_addr));

          // Push to stack and advance
          reg_stack.push_back(v_reg);
          v_ptr++;
        } else if (op == OpCode::LOAD_CONST) {
          asmjit::Reg s_reg = load_constant_x86(cc, scalars[idx], type_id);
          reg_stack.push_back(s_reg);
        } else {
          // Perform an operation from the virtual stack

          // Binary operations
          // -----------------
          // Get left and right operands
          asmjit::Reg r = reg_stack.back();
          reg_stack.pop_back();

          asmjit::Reg l = reg_stack.back();
          reg_stack.pop_back();

          // Allocate a new virtual register for the result of the operation
          asmjit::Reg res = cc.new_reg<asmjit::Reg>(type_id);

          // Perform linearized math using emit
          cc.emit(get_move_x86_inst(type_id), res, l);       // res = l
          cc.emit(get_binary_x86_inst(op, type_id), res, r); // res = res op r

          reg_stack.push_back(res);
        }
      }
      asmjit::Reg final_res = reg_stack.back();

      // Resolve the location for storing the result in the destination
      asmjit::x86::Gp dest_addr = cc.new_gp_ptr("dest_addr");
      cc.mov(dest_addr, dest_ptr_reg);

      std::vector<ssize_t> dest_strides(ndim);
      asmjit::TypeId dest_type_id { dtype_to_typeid(dest_t) };
      ssize_t current_stride { asmjit::TypeUtils::size_of(dest_type_id) };
      for (ssize_t dim = ndim - 1; dim >= 0; --dim) {
        dest_strides[dim] = current_stride;
        current_stride *= final_shape[dim];
      }

      asmjit::x86::Gp dest_offset = cc.new_gp(asmjit::TypeId::kInt64);
      cc.xor_(dest_offset, dest_offset);
      for (ssize_t dim = 0; dim < ndim; ++dim) {
        asmjit::x86::Gp term_dim = cc.new_gp(asmjit::TypeId::kInt64);
        cc.mov(term_dim, loop_regs[dim]);
        cc.imul(term_dim, dest_strides[dim]);
        cc.add(dest_offset, term_dim);
      }
      cc.add(dest_addr, dest_offset);

      // Store the result
      cc.emit(get_move_x86_inst(dest_type_id), asmjit::x86::ptr(dest_addr), final_res);

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
      asmjit::CodeBuffer& buffer = code.section_by_id(0)->buffer();
      std::size_t k_size { buffer.size() };
      const std::uint8_t* k_data = buffer.data();
      write_kernel_to_disk(cache_path, k_size, k_data);
    }

    asmjit::InstId RuntimeCompiler::get_binary_x86_inst(OpCode op, asmjit::TypeId type_id) {
      bool is_double { (type_id == asmjit::TypeId::kFloat64) };
      bool is_float { (type_id == asmjit::TypeId::kFloat32) };
      switch (op) {
      case OpCode::ADD: {
        if (is_double) {
          return asmjit::x86::Inst::kIdAddsd;
        } else if (is_float) {
          return asmjit::x86::Inst::kIdAddss;
        } else {
          return asmjit::x86::Inst::kIdAdd;
        }
      }
      case OpCode::SUB: {
        if (is_double) {
          return asmjit::x86::Inst::kIdSubsd;
        } else if (is_float) {
          return asmjit::x86::Inst::kIdSubss;
        } else {
          return asmjit::x86::Inst::kIdSub;
        }
      }
      case OpCode::MUL: {
        if (is_double) {
          return asmjit::x86::Inst::kIdMulsd;
        } else if (is_float) {
          return asmjit::x86::Inst::kIdMulss;
        } else {
          return asmjit::x86::Inst::kIdImul; // Signed integer multiplication
        }
      }
      case OpCode::DIV: {
        if (is_double) {
          return asmjit::x86::Inst::kIdDivsd;
        } else if (is_float) {
          return asmjit::x86::Inst::kIdDivss;
        } else {
          return asmjit::x86::Inst::kIdIdiv;
        }
      }
      default: {
        return asmjit::BaseInst::kIdNone;
        }
      }
    }

    asmjit::Reg RuntimeCompiler::load_constant_x86(asmjit::x86::Compiler& cc,
                                                   Scalar scalar,
                                                   asmjit::TypeId type_id) {
      // cc.new_reg automatically returns a Gp (for ints) or Vec (for floats)
      asmjit::Reg reg = cc.new_reg<asmjit::Reg>(type_id);
      if (asmjit::TypeUtils::is_int(type_id)) {
        std::int64_t val = 0;
        auto cast_op = [&](auto&& v) {
          using T = std::decay_t<decltype(v)>;
          val = op_traits<T>::template cast<std::int64_t>(v);
        };
        std::visit(cast_op, scalar);
        cc.mov(reg.as<asmjit::x86::Gp>(), val);
      } else if (type_id == asmjit::TypeId::kFloat32) {
        float val { 0.0f };
        auto cast_op = [&](auto&& v) {
          using T = std::decay_t<decltype(v)>;
          val = op_traits<T>::template cast<float>(v);
        };
        std::visit(cast_op, scalar);

        union {
          float f;
          std::uint32_t u;
        } u;
        u.f = val;
        asmjit::x86::Gp tmp_gp = cc.new_gp(asmjit::TypeId::kInt32);
        cc.mov(tmp_gp, u.u);
        cc.movd(reg.as<asmjit::x86::Vec>(), tmp_gp);
      } else if (type_id == asmjit::TypeId::kFloat64) {
        double val { 0.0 };
        auto cast_op = [&](auto&& v) {
          using T = std::decay_t<decltype(v)>;
          val = op_traits<T>::template cast<double>(v);
        };
        std::visit(cast_op, scalar);

        union {
          double d;
          std::uint64_t u;
        } u;
        u.d = val;
        asmjit::x86::Gp tmp_gp = cc.new_gp(asmjit::TypeId::kInt64);
        cc.mov(tmp_gp, u.u);
        cc.movq(reg.as<asmjit::x86::Vec>(), tmp_gp);
      }
      return reg;
    }

    asmjit::InstId RuntimeCompiler::get_move_x86_inst(asmjit::TypeId type_id) {
      if (asmjit::TypeUtils::is_int(type_id)) {
        return asmjit::x86::Inst::kIdMov;
      } else if (type_id == asmjit::TypeId::kFloat32) {
        return asmjit::x86::Inst::kIdMovss;
      } else if (type_id == asmjit::TypeId::kFloat64) {
        return asmjit::x86::Inst::kIdMovsd;
      }
      return asmjit::BaseInst::kIdNone;
    }
  } // namespace host
} // namespace ncarray
