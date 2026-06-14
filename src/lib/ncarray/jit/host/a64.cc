/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/jit/host/a64.hh"

#include "ncarray/dtype.hh"
#include "ncarray/layout.hh"
#include "ncarray/op_code.hh"
#include "ncarray/op_traits.hh"

#ifdef __CUDACC__
// Silence warning: reduction in alignment ignored
#pragma nv_diag_suppress 1286
#endif

#include <asmjit/a64.h>
#include <asmjit/core.h>

#ifdef __CUDACC__
#pragma nv_diag_default 1286
#endif

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <variant>

namespace ncarray {
  namespace host::a64 {
    void unary_operation(asmjit::a64::Compiler& cc,
                         OpCode op,
                         asmjit::TypeId type_id,
                         asmjit::Reg& operand,
                         asmjit::Reg& res) {
      bool is_int { asmjit::TypeUtils::is_int(type_id) };
      switch (op) {
      case OpCode::NEG: {
        if (is_int) {
          cc.neg(res.as<asmjit::a64::Gp>(), operand.as<asmjit::a64::Gp>());
        } else {
          // ARM has a floating point negative, unlike x86!
          cc.fneg(res.as<asmjit::a64::Vec>(), operand.as<asmjit::a64::Vec>());
        }
        break;
      }
      case OpCode::INC: {
        if (is_int) {
          cc.add(res.as<asmjit::a64::Gp>(), operand.as<asmjit::a64::Gp>(), 1);
        } else {
          asmjit::Reg one { load_constant(cc, Scalar(1.0), type_id) };
          cc.fadd(res.as<asmjit::a64::Vec>(),
                  operand.as<asmjit::a64::Vec>(),
                  one.as<asmjit::a64::Vec>());
        }
        break;
      }
      case OpCode::DEC: {
        if (is_int) {
          cc.sub(res.as<asmjit::a64::Gp>(), operand.as<asmjit::a64::Gp>(), 1);
        } else {
          asmjit::Reg one { load_constant(cc, Scalar(1.0), type_id) };
          cc.fsub(res.as<asmjit::a64::Vec>(),
                  operand.as<asmjit::a64::Vec>(),
                  one.as<asmjit::a64::Vec>());
        }
        break;
      }
      default: break;
      }
    }

    asmjit::InstId get_binary_arithmetic_inst(OpCode op, asmjit::TypeId type_id) {
      // Floating point (both single and double precision) use the same instruction
      bool is_fp {
        (type_id == asmjit::TypeId::kFloat32 || type_id == asmjit::TypeId::kFloat64)
      };
      bool is_vec { asmjit::TypeUtils::is_vec(type_id) };

      bool is_unsigned { false };
      if (type_id == asmjit::TypeId::kUInt8  ||
          type_id == asmjit::TypeId::kUInt16 ||
          type_id == asmjit::TypeId::kUInt32 ||
          type_id == asmjit::TypeId::kUInt64) {
        is_unsigned = true;
      }
      switch (op) {
      case OpCode::ADD: {
        if (is_fp) {
          return asmjit::a64::Inst::kIdFadd_v;
        } else if (is_vec) {
          return asmjit::a64::Inst::kIdAdd_v;
        } else {
          return asmjit::a64::Inst::kIdAdd;
        }
      }
      case OpCode::SUB: {
        if (is_fp) {
          return asmjit::a64::Inst::kIdFsub_v;
        } else if (is_vec) {
          return asmjit::a64::Inst::kIdSub_v;
        } else {
          return asmjit::a64::Inst::kIdSub;
        }
      }
      case OpCode::MUL: {
        // NOTE: Vec ops NOT appropriate for complex numbers!
        if (is_fp) {
          return asmjit::a64::Inst::kIdFmul_v;
        } else if (is_vec) {
          return asmjit::a64::Inst::kIdMul_v;
        } else {
          return asmjit::a64::Inst::kIdMul;
        }
      }
      case OpCode::DIV: {
        // NOTE: Vec ops NOT appropriate for complex numbers!
        if (is_fp) {
          return asmjit::a64::Inst::kIdFdiv_v;
        } else if (is_unsigned) {
          return asmjit::a64::Inst::kIdUdiv;
        } else {
          return asmjit::a64::Inst::kIdSdiv;
        }
      }
      default: {
        return asmjit::BaseInst::kIdNone;
      }
      }
    }

    void integer_div_mod(asmjit::a64::Compiler& cc,
                         OpCode op,
                         asmjit::TypeId type_id,
                         asmjit::Reg& left,
                         asmjit::Reg& right,
                         asmjit::Reg& res) {
      bool is_unsigned { false };
      if (type_id == asmjit::TypeId::kUInt8  ||
          type_id == asmjit::TypeId::kUInt16 ||
          type_id == asmjit::TypeId::kUInt32 ||
          type_id == asmjit::TypeId::kUInt64) {
        is_unsigned = true;
      }

      asmjit::InstId div_inst { asmjit::a64::Inst::kIdSdiv };
      if (is_unsigned) {
        div_inst = asmjit::a64::Inst::kIdUdiv;
      }

      asmjit::a64::Gp left_gp { left.as<asmjit::a64::Gp>() };
      asmjit::a64::Gp right_gp { right.as<asmjit::a64::Gp>() };
      asmjit::a64::Gp res_gp { res.as<asmjit::a64::Gp>() };

      if (op == OpCode::DIV) {
        cc.emit(div_inst, res_gp, left_gp, right_gp);
      } else {
        // As far as I can tell, unlike with x86, on ARM, the remainder (for modulo)
        // is not computed and stored automatically. Instead calculate the quotient
        // then use it to calculate modulo

        // Get quotient
        asmjit::a64::Gp quotient { cc.new_gp(type_id) };
        cc.emit(div_inst, quotient, left_gp, right_gp);

        // Calculate modulo as: res = left - (quotient * right)
        // This is actually a fast single instruction on ARM
        cc.emit(asmjit::a64::Inst::kIdMsub, res_gp, quotient, right_gp, left_gp);
      }
    }

    void binary_compare(asmjit::a64::Compiler& cc,
                        OpCode op,
                        asmjit::TypeId type_id,
                        asmjit::Reg& left,
                        asmjit::Reg& right,
                        asmjit::Reg& res) {
      bool is_unsigned { false };
      if (type_id == asmjit::TypeId::kUInt8  ||
          type_id == asmjit::TypeId::kUInt16 ||
          type_id == asmjit::TypeId::kUInt32 ||
          type_id == asmjit::TypeId::kUInt64) {
        is_unsigned = true;
      }

      // Select correct condition code based on the OpCode
      asmjit::a64::CondCode cond { asmjit::a64::CondCode::kEQ };
      switch (op) {
      case OpCode::EQ: {
        cond = asmjit::a64::CondCode::kEQ;
        break;
      }
      case OpCode::NE: {
        cond = asmjit::a64::CondCode::kNE;
        break;
      }
      case OpCode::LT: {
        if (is_unsigned) {
          cond = asmjit::a64::CondCode::kLO;
        } else {
          cond = asmjit::a64::CondCode::kLT;
        }
        break;
      }
      case OpCode::LE: {
        if (is_unsigned) {
          cond = asmjit::a64::CondCode::kLS;
        } else {
          cond = asmjit::a64::CondCode::kLE;
        }
        break;
      }
      case OpCode::GT: {
        if (is_unsigned) {
          cond = asmjit::a64::CondCode::kHI;
        } else {
          cond = asmjit::a64::CondCode::kGT;
        }
        break;
      }
      case OpCode::GE: {
        if (is_unsigned) {
          cond = asmjit::a64::CondCode::kHS;
        } else {
          cond = asmjit::a64::CondCode::kGE;
        }
        break;
      }
      default: break;
      }
      if (asmjit::TypeUtils::is_int(type_id)) {
        asmjit::a64::Gp left_gp  { left.as<asmjit::a64::Gp>() };
        asmjit::a64::Gp right_gp { right.as<asmjit::a64::Gp>() };
        asmjit::a64::Gp res_gp   { res.as<asmjit::a64::Gp>() };

        // Compare and set 0/1 based on code
        cc.cmp(left_gp, right_gp);
        cc.cset(res_gp, cond);
      } else {
        asmjit::a64::Vec left_vec  { left.as<asmjit::a64::Vec>() };
        asmjit::a64::Vec right_vec { right.as<asmjit::a64::Vec>() };
        asmjit::a64::Vec res_vec   { res.as<asmjit::a64::Vec>() };

        // Compare
        cc.fcmp(left_vec, right_vec);

        // Convert to temp int register to read result and then move to float reg
        if (type_id == asmjit::TypeId::kFloat32) {
          asmjit::a64::Gp tmp_gp { cc.new_gp32() };
          cc.cset(tmp_gp, cond);
          cc.scvtf(res_vec, tmp_gp);
        } else {
          asmjit::a64::Gp tmp_gp64 { cc.new_gp64() };
          cc.cset(tmp_gp64, cond);
          cc.scvtf(res_vec, tmp_gp64);
        }
      }
    }

    asmjit::Reg cast_register(asmjit::a64::Compiler& cc,
                              asmjit::Reg src,
                              asmjit::TypeId src_type,
                              asmjit::TypeId dest_type) {
      if (src_type == dest_type) {
        return src;
      }
      asmjit::Reg dest { cc.new_reg<asmjit::Reg>(dest_type) };

      bool src_is_float { asmjit::TypeUtils::is_float(src_type) };
      bool dest_is_float { asmjit::TypeUtils::is_float(dest_type) };

      std::uint32_t src_size { asmjit::TypeUtils::size_of(src_type) };
      std::uint32_t dest_size { asmjit::TypeUtils::size_of(dest_type) };

      if (src_is_float && dest_is_float) {
        // Floating point width conversions (32 -> 64 and vice versa)
        cc.fcvt(dest.as<asmjit::a64::Vec>(), src.as<asmjit::a64::Vec>());
      } else if (src_is_float && !dest_is_float) {
        // Convert floating point to integer
        bool dest_is_unsigned { false };
        if (dest_type == asmjit::TypeId::kUInt8  ||
            dest_type == asmjit::TypeId::kUInt16 ||
            dest_type == asmjit::TypeId::kUInt32 ||
            dest_type == asmjit::TypeId::kUInt64) {
          dest_is_unsigned = true;
        }

        asmjit::InstId conv_inst { asmjit::a64::Inst::kIdFcvtzs_v };
        if (dest_is_unsigned) {
          conv_inst = asmjit::a64::Inst::kIdFcvtzu_v;
        }

        asmjit::a64::Gp dest_gp { dest.as<asmjit::a64::Gp>() };
        if (dest_size <= 4) {
          // Convert directly to a 32-bit integer register (W)
          cc.emit(conv_inst, dest_gp.r32(), src.as<asmjit::a64::Vec>());
        } else {
          // Convert directly to a 64-bit integer register (X)
          cc.emit(conv_inst, dest_gp.r64(), src.as<asmjit::a64::Vec>());
        }
      } else if (!src_is_float && dest_is_float) {
        // Convert integer to floating point
        asmjit::a64::Gp src_gp { src.as<asmjit::a64::Gp>() };
        asmjit::a64::Gp tmp { src_gp };
        if (src_size < 4) {
          // For smaller integers, sign/zero extend to 32-bit GP register first
          tmp = cc.new_gp32();
          bool src_is_unsigned { false };
          if (src_type == asmjit::TypeId::kUInt8  ||
              src_type == asmjit::TypeId::kUInt16) {
            src_is_unsigned = true;
          }

          if (src_is_unsigned) {
            if (src_size == 1) {
              cc.uxtb(tmp, src_gp.r32());
            } else {
              cc.uxth(tmp, src_gp.r32());
            }
          } else {
            if (src_size == 1) {
              cc.sxtb(tmp, src_gp.r32());
            } else {
              cc.sxth(tmp, src_gp.r32());
            }
          }
        }
        // Convert integer to floating point (scvtf / ucvtf)
        bool src_is_unsigned { false };
        if (src_type == asmjit::TypeId::kUInt8  ||
            src_type == asmjit::TypeId::kUInt16 ||
            src_type == asmjit::TypeId::kUInt32 ||
            src_type == asmjit::TypeId::kUInt64) {
          src_is_unsigned = true;
        }
        asmjit::InstId conv_inst { asmjit::a64::Inst::kIdScvtf_v };
        if (src_is_unsigned) {
          conv_inst = asmjit::a64::Inst::kIdUcvtf_v;
        }
        cc.emit(conv_inst, dest.as<asmjit::a64::Vec>(), tmp);
      } else {
        // Convert integers to other integers
        asmjit::a64::Gp src_gp { src.as<asmjit::a64::Gp>() };
        asmjit::a64::Gp dest_gp { dest.as<asmjit::a64::Gp>() };

        if (dest_size > src_size) {
          bool src_is_unsigned { false };
          if (src_type == asmjit::TypeId::kUInt8  ||
              src_type == asmjit::TypeId::kUInt16 ||
              src_type == asmjit::TypeId::kUInt32 ||
              src_type == asmjit::TypeId::kUInt64) {
            src_is_unsigned = true;
          }

          if (src_is_unsigned) {
            if (src_size == 1) {
              cc.uxtb(dest_gp.r32(), src_gp.r32());
            } else if (src_size == 2) {
              cc.uxth(dest_gp.r32(), src_gp.r32());
            } else {
              // 32-bit zero-extension to 64-bit is implicit in AArch64 when moving to W register
              cc.mov(dest_gp.r32(), src_gp.r32());
            }
          } else {
            if (src_size == 1) {
              // Sign-extend byte to 32-bit or 64-bit
              if (dest_size == 8) {
                cc.sxtb(dest_gp.r64(), src_gp.r32());
              } else {
                cc.sxtb(dest_gp.r32(), src_gp.r32());
              }
            } else if (src_size == 2) {
              // Sign-extend halfword to 32-bit or 64-bit
              if (dest_size == 8) {
                cc.sxth(dest_gp.r64(), src_gp.r32());
              } else {
                cc.sxth(dest_gp.r32(), src_gp.r32());
              }
            } else {
              // Sign-extend word (32-bit) to doubleword (64-bit)
              cc.sxtw(dest_gp.r64(), src_gp.r32());
            }
          }
        } else {
          // Destination is <= source width - either simple move, or sub-register move
          if (dest_size == 1) {
            cc.mov(dest_gp.r32(), src_gp.r32());
          } else if (dest_size == 2) {
            cc.mov(dest_gp.r32(), src_gp.r32());
          } else if (dest_size == 4) {
            cc.mov(dest_gp.r32(), src_gp.r32());
          } else {
            cc.mov(dest_gp.r64(), src_gp.r64());
          }
        }
      }
      return dest;
    }

    asmjit::Reg load_constant(asmjit::a64::Compiler& cc,
                              Scalar scalar,
                              asmjit::TypeId type_id) {
      // cc.new_reg automatically returns a Gp (for ints) or Vec (for floats)
      asmjit::Reg reg { cc.new_reg<asmjit::Reg>(type_id) };
      if (asmjit::TypeUtils::is_int(type_id)) {
        std::int64_t val { 0 };
        auto cast_op = [&](auto&& v) {
          using T = std::decay_t<decltype(v)>;
          val = op_traits<T>::template cast<std::int64_t>(v);
        };
        std::visit(cast_op, scalar);
        cc.mov(reg.as<asmjit::a64::Gp>(), val);
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
        asmjit::a64::Gp tmp_gp { cc.new_gp(asmjit::TypeId::kInt32) };
        cc.mov(tmp_gp, u.u);
        cc.fmov(reg.as<asmjit::a64::Vec>(), tmp_gp);
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
        asmjit::a64::Gp tmp_gp { cc.new_gp(asmjit::TypeId::kInt64) };
        cc.mov(tmp_gp, u.u);
        cc.fmov(reg.as<asmjit::a64::Vec>(), tmp_gp);
      }
      return reg;
    }

    asmjit::InstId get_move_inst(asmjit::TypeId type_id) {
      if (asmjit::TypeUtils::is_int(type_id)) {
        return asmjit::a64::Inst::kIdMov;
      } else {
        return asmjit::a64::Inst::kIdMov_v;
      }
    }

    void emit_load(asmjit::a64::Compiler& cc,
                   asmjit::Reg& dest,
                   const asmjit::a64::Mem& src) {
      if (dest.is_gp()) {
        cc.ldr(dest.as<asmjit::a64::Gp>(), src);
      } else {
        cc.ldr(dest.as<asmjit::a64::Vec>(), src);
      }
    }

    void emit_store(asmjit::a64::Compiler& cc,
                    const asmjit::a64::Mem& dest,
                    asmjit::Reg& src) {
      if (src.is_gp()) {
        cc.str(src.as<asmjit::a64::Gp>(), dest);
      } else {
        cc.str(src.as<asmjit::a64::Vec>(), dest);
      }
    }

    void scaled_address_array(asmjit::a64::Compiler& cc,
                              asmjit::a64::Gp& addr,
                              asmjit::a64::Gp& index,
                              asmjit::Reg& value,
                              ssize_t dim,
                              const SOArrayPolicy& arr_layout,
                              asmjit::TypeId type_id,
                              bool addr_is_sink,
                              bool expr_is_soarr) {
      if (!expr_is_soarr) {
        auto& ncl = reinterpret_cast<const NCOffsetsPolicy&>(arr_layout);
        if (ncl.is_pointer_axis(dim)) {
          advance_ncoffsets_pointer_axis(cc,
                                         addr,
                                         index,
                                         ncl.offset(dim));
          if (!addr_is_sink) {
            emit_load(cc, value, asmjit::a64::ptr(addr));
          } else {
            emit_store(cc, asmjit::a64::ptr(addr), value);
          }
        } else {
          // Normal strided traversal, with scaled loading
          ssize_t stride { ncl.stride(dim) };
          ssize_t offset { ncl.offset(dim) };
          if (offset != 0) {
            asmjit::a64::Gp term_off { cc.new_gp(asmjit::TypeId::kInt64) };
            cc.mov(term_off, offset);
            cc.add(addr, addr, term_off);
          }

          // ARM also has hardward support for register-shifted offsets in load/store
          int scale_shift { -1 };
          switch (stride) {
          case 1:  { scale_shift =  0; break; }
          case 2:  { scale_shift =  1; break; }
          case 4:  { scale_shift =  2; break; }
          case 8:  { scale_shift =  3; break; }
          default: { scale_shift = -1; break; }
          }
          // But you need to use a shift operand (lsl) instead of the raw int like x86
          if (scale_shift >= 0) {
            if (!addr_is_sink) {
              emit_load(cc,
                        value,
                        asmjit::a64::ptr(addr, index, asmjit::a64::lsl(scale_shift)));
            } else {
              emit_store(cc,
                         asmjit::a64::ptr(addr, index, asmjit::a64::lsl(scale_shift)),
                         value);
            }
          } else {
            // Offset handled up above, so pass -1
            advance_ncoffsets_strided_axis(cc, addr, index, stride, -1);
            if (!addr_is_sink) {
              emit_load(cc, value, asmjit::a64::ptr(addr));
            } else {
              emit_store(cc, asmjit::a64::ptr(addr), value);
            }
          }
        }
      } else {
        // For SOArrayPolicy we first do normal strided traversal (index * stride)
        ssize_t stride { arr_layout.stride(dim) };
        ssize_t suboffset { arr_layout.suboffset(dim) };
        if (suboffset >= 0) {
          // Then, if it has a suboffset, dereference and add it
          advance_soarray_axis(cc, addr, index, stride, suboffset);
          if (!addr_is_sink) {
            emit_load(cc, value, asmjit::a64::ptr(addr));
          } else {
            emit_store(cc, asmjit::a64::ptr(addr), value);
          }
        } else {
          // Without pointer axis, use a64 scaled hardware loading
          int scale_shift { -1 };
          switch (stride) {
          case 1:  { scale_shift =  0; break; }
          case 2:  { scale_shift =  1; break; }
          case 4:  { scale_shift =  2; break; }
          case 8:  { scale_shift =  3; break; }
          default: { scale_shift = -1; break; }
          }
          // But you need to use a shift operand (lsl) instead of the raw int like x86
          if (scale_shift >= 0) {
            if (!addr_is_sink) {
              emit_load(cc,
                        value,
                        asmjit::a64::ptr(addr, index, asmjit::a64::lsl(scale_shift)));
            } else {
              emit_store(cc,
                         asmjit::a64::ptr(addr, index, asmjit::a64::lsl(scale_shift)),
                         value);
            }
          } else {
            // suboffset should be negative, but we otherwise determined it should
            // not be used, so pass -1 to emit instructions
            advance_soarray_axis(cc, addr, index, stride, -1);
            if (!addr_is_sink) {
              emit_load(cc, value, asmjit::a64::ptr(addr));
            } else {
              emit_store(cc, asmjit::a64::ptr(addr), value);
            }
          }
        }
      }
    }

    void advance_ncoffsets_pointer_axis(asmjit::a64::Compiler& cc,
                                        asmjit::a64::Gp& addr,
                                        asmjit::a64::Gp& index,
                                        ssize_t offset) {
      // For NCOffsetsPolicy for pointer axis dereference and load
      asmjit::a64::Gp ptr_idx { cc.new_gp(asmjit::TypeId::kInt64) };
      cc.mov(ptr_idx, index);
      if (offset != 0) {
        cc.add(ptr_idx, ptr_idx, offset);
      }
      cc.lsl(ptr_idx, ptr_idx, 3); // General purpose register shift is lsl
      cc.add(addr, addr, ptr_idx);
      // Now dereference
      cc.ldr(addr, asmjit::a64::ptr(addr)); // Use ldr instead of mov
    }

    void advance_ncoffsets_strided_axis(asmjit::a64::Compiler& cc,
                                        asmjit::a64::Gp& addr,
                                        asmjit::a64::Gp& index,
                                        ssize_t stride,
                                        ssize_t offset) {
      asmjit::a64::Gp term { cc.new_gp(asmjit::TypeId::kInt64) };
      cc.mov(term, index);

      // Load stride into register since mul doesn't take immediates on ARM64
      asmjit::a64::Gp stride_reg { cc.new_gp(asmjit::TypeId::kInt64) };
      cc.mov(stride_reg, stride);
      cc.mul(term, term, stride_reg);

      cc.add(addr, addr, term);
      if (offset > 0) {
        asmjit::a64::Gp term_off { cc.new_gp(asmjit::TypeId::kInt64) };
        cc.mov(term_off, offset);
        cc.add(addr, addr, term_off);
      }
    }

    void advance_soarray_axis(asmjit::a64::Compiler& cc,
                              asmjit::a64::Gp& addr,
                              asmjit::a64::Gp& index,
                              ssize_t stride,
                              ssize_t suboffset) {
      // For SOArrayPolicy we first do normal strided traversal (index * stride)
      asmjit::a64::Gp term { cc.new_gp(asmjit::TypeId::kInt64) };
      cc.mov(term, index);

      // Load stride into register since mul doesn't take immediates on ARM64
      asmjit::a64::Gp stride_reg { cc.new_gp(asmjit::TypeId::kInt64) };
      cc.mov(stride_reg, stride);
      cc.mul(term, term, stride_reg);

      cc.add(addr, addr, term);
      if (suboffset >= 0) {
        // Then, if it has a suboffset, dereference and add it
        cc.ldr(addr, asmjit::a64::ptr(addr));
        if (suboffset != 0) {
          asmjit::a64::Gp term_sub { cc.new_gp(asmjit::TypeId::kInt64) };
          cc.mov(term_sub, suboffset);
          cc.add(addr, addr, term_sub);
        }
      }
    }
  } // namespace host::a64
} // namespace ncarray
