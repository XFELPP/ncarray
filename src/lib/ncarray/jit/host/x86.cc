/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/jit/host/x86.hh"

#include "ncarray/dtype.hh"
#include "ncarray/layout.hh"
#include "ncarray/op_code.hh"
#include "ncarray/op_traits.hh"

#include <asmjit/core.h>
#include <asmjit/x86.h>

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
  namespace host::x86 {
    void unary_operation(asmjit::x86::Compiler& cc,
                         OpCode op,
                         asmjit::TypeId type_id,
                         asmjit::Reg& operand,
                         asmjit::Reg& res) {
      switch (op) {
      case OpCode::NEG: {
        if (asmjit::TypeUtils::is_int(type_id)) {
          cc.mov(res.as<asmjit::x86::Gp>(), operand.as<asmjit::x86::Gp>());
          cc.neg(res.as<asmjit::x86::Gp>());
        } else {
          asmjit::x86::Vec op_vec { operand.as<asmjit::x86::Vec>() };
          asmjit::x86::Vec res_vec { res.as<asmjit::x86::Vec>() };

          if (type_id == asmjit::TypeId::kFloat32) {
            // Create mask for bit 31 (sign bit)
            asmjit::x86::Gp tmp { cc.new_gp32() };
            cc.mov(tmp, 0x80000000u);

            asmjit::x86::Vec mask { cc.new_xmm_ss() };
            cc.movd(mask, tmp);

            // Negate with XOR
            cc.vxorps(res_vec, op_vec, mask);
          } else {
            // Create mask for bit 63 (sign bit)
            asmjit::x86::Gp tmp { cc.new_gp64() };
            cc.mov(tmp, 0x8000000000000000uLL);

            asmjit::x86::Vec mask { cc.new_xmm_sd() };
            cc.movd(mask, tmp);

            // Negate with XOR
            cc.vxorps(res_vec, op_vec, mask);
          }
        }
        break;
      }
      case OpCode::INC: {
        if (asmjit::TypeUtils::is_int(type_id)) {
          cc.mov(res.as<asmjit::x86::Gp>(), operand.as<asmjit::x86::Gp>());
          cc.inc(res.as<asmjit::x86::Gp>());
        } else {
          asmjit::Reg one { load_constant(cc, Scalar(1.0), type_id) };
          cc.emit(x86::get_move_inst(type_id), res, operand);                       // res = operand
          cc.emit(x86::get_binary_arithmetic_inst(OpCode::ADD, type_id), res, one); // res = res + 1
        }
        break;
      }
      case OpCode::DEC: {
        if (asmjit::TypeUtils::is_int(type_id)) {
          cc.mov(res.as<asmjit::x86::Gp>(), operand.as<asmjit::x86::Gp>());
          cc.dec(res.as<asmjit::x86::Gp>());
        } else {
          asmjit::Reg one { load_constant(cc, Scalar(1.0), type_id) };
          cc.emit(x86::get_move_inst(type_id), res, operand);                       // res = operand
          cc.emit(x86::get_binary_arithmetic_inst(OpCode::SUB, type_id), res, one); // res = res - 1
        }
        break;
      }
      default: break;
      }
    }

    asmjit::InstId get_binary_arithmetic_inst(OpCode op, asmjit::TypeId type_id) {
      bool is_double { (type_id == asmjit::TypeId::kFloat64) };
      bool is_double_vec {
        (type_id == asmjit::TypeId::kFloat64x2 || type_id == asmjit::TypeId::kFloat64x4)
      };
      bool is_float { (type_id == asmjit::TypeId::kFloat32) };
      bool is_float_vec {
        (type_id == asmjit::TypeId::kFloat32x2 ||
         type_id == asmjit::TypeId::kFloat32x4 ||
         type_id == asmjit::TypeId::kFloat32x8)
      };
      switch (op) {
      case OpCode::ADD: {
        if (is_double) {
          return asmjit::x86::Inst::kIdAddsd;
        } else if (is_double_vec) {
          return asmjit::x86::Inst::kIdAddpd;
        } else if (is_float) {
          return asmjit::x86::Inst::kIdAddss;
        } else if (is_float_vec) {
          return asmjit::x86::Inst::kIdAddps;
        } else {
          return asmjit::x86::Inst::kIdAdd;
        }
      }
      case OpCode::SUB: {
        if (is_double) {
          return asmjit::x86::Inst::kIdSubsd;
        } else if (is_double_vec) {
          return asmjit::x86::Inst::kIdSubpd;
        } else if (is_float) {
          return asmjit::x86::Inst::kIdSubss;
        } else if (is_float_vec) {
          return asmjit::x86::Inst::kIdSubps;
        } else {
          return asmjit::x86::Inst::kIdSub;
        }
      }
      case OpCode::MUL: {
        // NOTE: Vec ops NOT appropriate for complex numbers!
        if (is_double) {
          return asmjit::x86::Inst::kIdMulsd;
        } else if (is_double_vec) {
          return asmjit::x86::Inst::kIdMulpd;
        } else if (is_float) {
          return asmjit::x86::Inst::kIdMulss;
        } else if (is_float_vec) {
          return asmjit::x86::Inst::kIdMulps;
        } else {
          return asmjit::x86::Inst::kIdImul; // Signed integer multiplication
        }
      }
      case OpCode::DIV: {
        // NOTE: Vec ops NOT appropriate for complex numbers!
        if (is_double) {
          return asmjit::x86::Inst::kIdDivsd;
        } else if (is_double_vec) {
          return asmjit::x86::Inst::kIdDivpd;
        } else if (is_float) {
          return asmjit::x86::Inst::kIdDivss;
        } else if (is_float_vec) {
          return asmjit::x86::Inst::kIdDivps;
        } else {
          return asmjit::x86::Inst::kIdIdiv;
        }
      }
      default: {
        return asmjit::BaseInst::kIdNone;
      }
      }
    }

    void integer_div_mod(asmjit::x86::Compiler& cc,
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

      std::uint32_t size { asmjit::TypeUtils::size_of(type_id) };

      // Despite requiring the use of EAX, RAX etc. since we've been using all
      // virtual register allocation we have to make sure to continue, otherwise
      // the allocator in the Compiler won't know that we've selected physical
      // registers and overwrite them. This can lead to corruption or all sorts
      // of problems.
      if (size == 1) {
        // For 8-bit division must place dividend in AX
        // Divisor can be in any register or memory
        asmjit::x86::Gp ax_reg { cc.new_gp16("ax_reg") };

        // Perform zero or sign extension to fill the 16-bits
        if (is_unsigned) {
          cc.movzx(ax_reg, left.as<asmjit::x86::Gp>());
          cc.div(ax_reg, right.as<asmjit::x86::Gp>());
        } else {
          cc.movsx(ax_reg, left.as<asmjit::x86::Gp>());
          cc.idiv(ax_reg, right.as<asmjit::x86::Gp>());
        }

        // Lower half will have the quotient, upper half the remainder
        if (op == OpCode::DIV) {
          cc.mov(res.as<asmjit::x86::Gp>(), ax_reg.r8_lo());
        } else {
          cc.mov(res.as<asmjit::x86::Gp>(), ax_reg.r8_hi());
        }

      } else if (size == 2) {
        // For 16-bit division, the quotient ends up in DX, and the remainder in AX
        asmjit::x86::Gp quot { cc.new_gp16("quot") };
        asmjit::x86::Gp rem { cc.new_gp16("rem") };

        cc.mov(quot, left.as<asmjit::x86::Gp>());
        if (is_unsigned) {
          cc.xor_(rem, rem);
          cc.div(rem, quot, right.as<asmjit::x86::Gp>());
        } else {
          cc.cwd(rem, quot);
          cc.idiv(rem, quot, right.as<asmjit::x86::Gp>());
        }

        // Quotient in DX, and remainder in AX
        if (op == OpCode::DIV) {
          cc.mov(res.as<asmjit::x86::Gp>(), quot);
        } else {
          cc.mov(res.as<asmjit::x86::Gp>(), rem);
        }

      } else if (size == 4) {
        // For 32-bit division, the quotient ends up in EDX, and the remainder in EAX
        asmjit::x86::Gp quot { cc.new_gp32("quot") };
        asmjit::x86::Gp rem { cc.new_gp32("rem") };

        cc.mov(quot, left.as<asmjit::x86::Gp>());
        if (is_unsigned) {
          cc.xor_(rem, rem);
          cc.div(rem, quot, right.as<asmjit::x86::Gp>());
        } else {
          cc.cdq(rem, quot);
          cc.idiv(rem, quot, right.as<asmjit::x86::Gp>());
        }

        if (op == OpCode::DIV) {
          cc.mov(res.as<asmjit::x86::Gp>(), quot);
        } else {
          cc.mov(res.as<asmjit::x86::Gp>(), rem);
        }

      } else {
        // For 64-bit division, the quotient ends up in RDX, and the remainder in RAX
        asmjit::x86::Gp quot { cc.new_gp64("quot") };
        asmjit::x86::Gp rem { cc.new_gp64("rem") };

        cc.mov(quot, left.as<asmjit::x86::Gp>());
        if (is_unsigned) {
          cc.xor_(rem, rem);
          cc.div(rem, quot, right.as<asmjit::x86::Gp>());
        } else {
          cc.cqo(rem, quot);
          cc.idiv(rem, quot, right.as<asmjit::x86::Gp>());
        }

        if (op == OpCode::DIV) {
          cc.mov(res.as<asmjit::x86::Gp>(), quot);
        } else {
          cc.mov(res.as<asmjit::x86::Gp>(), rem);
        }
      }
    }

    void binary_compare(asmjit::x86::Compiler& cc,
                        OpCode op,
                        asmjit::TypeId type_id,
                        asmjit::Reg& left,
                        asmjit::Reg& right,
                        asmjit::Reg& res) {
      if (asmjit::TypeUtils::is_int(type_id)) {
        asmjit::x86::Gp left_addr { left.as<asmjit::x86::Gp>() };
        asmjit::x86::Gp right_addr { right.as<asmjit::x86::Gp>() };
        asmjit::x86::Gp res_addr { res.as<asmjit::x86::Gp>() };

        cc.xor_(res_addr, res_addr);
        cc.cmp(left_addr, right_addr);

        bool is_unsigned { false };
        if (type_id == asmjit::TypeId::kUInt8  ||
            type_id == asmjit::TypeId::kUInt16 ||
            type_id == asmjit::TypeId::kUInt32 ||
            type_id == asmjit::TypeId::kUInt64) {
          is_unsigned = true;
        }

        switch (op) {
        case OpCode::EQ: {
          cc.sete(res_addr.r8());
          break;
        }
        case OpCode::NE: {
          cc.setne(res_addr.r8());
          break;
        }
        case OpCode::LT: {
          if (is_unsigned) {
            cc.setb(res_addr.r8());
          } else {
            cc.setl(res_addr.r8());
          }
          break;
        }
        case OpCode::LE: {
          if (is_unsigned) {
            cc.setbe(res_addr.r8());
          } else {
            cc.setle(res_addr.r8());
          }
          break;
        }
        case OpCode::GT: {
          if (is_unsigned) {
            cc.seta(res_addr.r8());
          } else {
            cc.setg(res_addr.r8());
          }
          break;
        }
        case OpCode::GE: {
          if (is_unsigned) {
            cc.setae(res_addr.r8());
          } else {
            cc.setge(res_addr.r8());
          }
          break;
        }
        default: break;
        }
      } else {
        asmjit::x86::Vec left_vec { left.as<asmjit::x86::Vec>() };
        asmjit::x86::Vec right_vec { right.as<asmjit::x86::Vec>() };

        asmjit::x86::Vec res_vec { res.as<asmjit::x86::Vec>() };

        // Select correct predicate for the FP comparisons
        asmjit::x86::VCmpImm imm { asmjit::x86::VCmpImm::kEQ_OQ };
        switch (op) {
        case OpCode::EQ: {
          imm = asmjit::x86::VCmpImm::kEQ_OQ;
          break;
        }
        case OpCode::NE: {
          imm = asmjit::x86::VCmpImm::kNEQ_OQ;
          break;
        }
        case OpCode::LT: {
          imm = asmjit::x86::VCmpImm::kLT_OQ;
          break;
        }
        case OpCode::LE: {
          imm = asmjit::x86::VCmpImm::kLE_OQ;
          break;
        }
        case OpCode::GT: {
          imm = asmjit::x86::VCmpImm::kGT_OQ;
          break;
        }
        case OpCode::GE: {
          imm = asmjit::x86::VCmpImm::kGE_OQ;
          break;
        }
        default: break;
        }

        if (type_id == asmjit::TypeId::kFloat32) {
          // Produces mask in res
          cc.vcmpss(res_vec, left_vec, right_vec, imm);
          // Convert mask to 0 or 1 integer
          cc.vpsrld(res_vec, res_vec, 31);

          // Now reconvert to float
          cc.vcvtdq2ps(res_vec, res_vec);
        } else {
          asmjit::x86::Vec one_vec { load_constant(cc, Scalar(1.0), type_id).as<asmjit::x86::Vec>() };

          // Produces mask in res
          cc.vcmpsd(res_vec, left_vec, right_vec, imm);

          // Run bitwise and (res = res & 1.0)
          cc.vandpd(res_vec, res_vec, one_vec);
        }

      }
    }

    asmjit::Reg cast_register(asmjit::x86::Compiler& cc,
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
        if (src_type == asmjit::TypeId::kFloat32 && dest_type == asmjit::TypeId::kFloat64) {
          cc.cvtss2sd(dest.as<asmjit::x86::Vec>(), src.as<asmjit::x86::Vec>());
        } else if (src_type == asmjit::TypeId::kFloat64 && dest_type == asmjit::TypeId::kFloat32) {
          cc.cvtsd2ss(dest.as<asmjit::x86::Vec>(), src.as<asmjit::x86::Vec>());
        }
      } else if (src_is_float && !dest_is_float) {
        // Convert floating point to integer
        asmjit::x86::Gp tmp { cc.new_gp(asmjit::TypeId::kInt32) };
        if (src_type == asmjit::TypeId::kFloat32) {
          cc.cvttss2si(tmp, src.as<asmjit::x86::Vec>());
        } else {
          cc.cvttsd2si(tmp, src.as<asmjit::x86::Vec>());
        }

        // Truncate or extend the temporary into correct destination width
        asmjit::x86::Gp dest_gp { dest.as<asmjit::x86::Gp>() };
        if (dest_size == 1) {
          cc.mov(dest_gp.r8(), tmp.r8());
        } else if (dest_size == 2) {
          cc.mov(dest_gp.r16(), tmp.r16());
        } else if (dest_size == 4) {
          cc.mov(dest_gp.r32(), tmp); // tmp is already 32 bit
        } else {
          // Extend 32-bit to 64-bit
          cc.movsxd(dest_gp, tmp);
        }
      } else if (!src_is_float && dest_is_float) {
        // Convert integer to floating point
        asmjit::x86::Gp tmp { src.as<asmjit::x86::Gp>() };

        if (src_size < 4) {
          tmp = cc.new_gp(asmjit::TypeId::kInt32);

          bool src_is_unsigned { false };
          if (src_type == asmjit::TypeId::kUInt8  ||
              src_type == asmjit::TypeId::kUInt16) {
            src_is_unsigned = true;
          }

          if (src_is_unsigned) {
            if (src_size == 1) {
              cc.movzx(tmp, src.as<asmjit::x86::Gp>().r8());
            } else {
              cc.movzx(tmp, src.as<asmjit::x86::Gp>().r16());
            }
          } else {
            if (src_size == 1) {
              cc.movsx(tmp, src.as<asmjit::x86::Gp>().r8());
            } else {
              cc.movsx(tmp, src.as<asmjit::x86::Gp>().r16());
            }
          }
        }
        if (dest_type == asmjit::TypeId::kFloat32) {
          cc.cvtsi2ss(dest.as<asmjit::x86::Vec>(), tmp);
        } else {
          cc.cvtsi2sd(dest.as<asmjit::x86::Vec>(), tmp);
        }
      } else {
        // Convert integers to other integers
        asmjit::x86::Gp src_gp { src.as<asmjit::x86::Gp>() };
        asmjit::x86::Gp dest_gp { dest.as<asmjit::x86::Gp>() };

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
              cc.movzx(dest_gp, src_gp.r8());
            } else if (src_size == 2) {
              cc.movzx(dest_gp, src_gp.r16());
            } else {
              // The zero-extension from 32-bit to 64-bit is implicit when using
              // 32b sub-register
              cc.mov(dest_gp.r32(), src_gp);
            }
          } else {
            if (src_size == 1) {
              cc.movsx(dest_gp, src_gp.r8());
            } else if (src_size == 2) {
              cc.movsx(dest_gp, src_gp.r16());
            } else {
              // 32-bit zero extension to 64
              cc.movsxd(dest_gp, src_gp);
            }
          }
        } else {
          // Destination is <= source width - either simple move, or just truncate
          if (dest_size == 1) {
            cc.mov(dest_gp.r8(), src_gp.r8());
          } else if (dest_size == 2) {
            cc.mov(dest_gp.r16(), src_gp.r16());
          } else if (dest_size == 4) {
            cc.mov(dest_gp.r32(), src_gp.r32());
          } else {
            cc.mov(dest_gp, src_gp);
          }
        }
      }

      return dest;
    }

    asmjit::Reg load_constant(asmjit::x86::Compiler& cc,
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
        asmjit::x86::Gp tmp_gp { cc.new_gp(asmjit::TypeId::kInt32) };
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
        asmjit::x86::Gp tmp_gp { cc.new_gp(asmjit::TypeId::kInt64) };
        cc.mov(tmp_gp, u.u);
        cc.movq(reg.as<asmjit::x86::Vec>(), tmp_gp);
      }
      return reg;
    }

    asmjit::InstId get_move_inst(asmjit::TypeId type_id) {
      if (asmjit::TypeUtils::is_int(type_id)) {
        return asmjit::x86::Inst::kIdMov;
      } else if (type_id == asmjit::TypeId::kFloat32) {
        return asmjit::x86::Inst::kIdMovss;
      } else if (type_id == asmjit::TypeId::kFloat64) {
        return asmjit::x86::Inst::kIdMovsd;
      } else if (type_id == asmjit::TypeId::kFloat32x2) {
        return asmjit::x86::Inst::kIdMovq;
      } else if (type_id == asmjit::TypeId::kFloat32x4 ||
                 type_id == asmjit::TypeId::kFloat32x8) {
        return asmjit::x86::Inst::kIdMovups;
      } else if (type_id == asmjit::TypeId::kFloat64x2) {
        return asmjit::x86::Inst::kIdMovupd;
      } else if (type_id == asmjit::TypeId::kFloat64x4) {
        return asmjit::x86::Inst::kIdVmovupd;
      }
      return asmjit::BaseInst::kIdNone;
    }

    void scaled_address_array(asmjit::x86::Compiler& cc,
                              asmjit::x86::Gp& addr,
                              asmjit::x86::Gp& index,
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
            cc.emit(get_move_inst(type_id), value, asmjit::x86::ptr(addr));
          } else {
            cc.emit(get_move_inst(type_id), asmjit::x86::ptr(addr), value);
          }
        } else {
          // Normal strided traversal, with scaled loading
          ssize_t stride { ncl.stride(dim) };
          ssize_t offset { ncl.offset(dim) };
          if (offset != 0) {
            asmjit::x86::Gp term_off { cc.new_gp(asmjit::TypeId::kInt64) };
            cc.mov(term_off, offset);
            cc.add(addr, term_off);
          }

          // Use x86 scaled hardware loading
          int scale_shift { -1 };
          switch (stride) {
          case 1:  { scale_shift =  0; break; }
          case 2:  { scale_shift =  1; break; }
          case 4:  { scale_shift =  2; break; }
          case 8:  { scale_shift =  3; break; }
          default: { scale_shift = -1; break; }
          }

          if (scale_shift >= 0) {
            if (!addr_is_sink) {
              cc.emit(get_move_inst(type_id),
                      value,
                      asmjit::x86::ptr(addr, index, scale_shift));
            } else {
              cc.emit(get_move_inst(type_id),
                      asmjit::x86::ptr(addr, index, scale_shift),
                      value);
            }
          } else {
            // Offset handled up above, so pass -1
            advance_ncoffsets_strided_axis(cc, addr, index, stride, -1);
            if (!addr_is_sink) {
              cc.emit(get_move_inst(type_id), value, asmjit::x86::ptr(addr));
            } else {
              cc.emit(get_move_inst(type_id), asmjit::x86::ptr(addr), value);
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
            cc.emit(get_move_inst(type_id), value, asmjit::x86::ptr(addr));
          } else {
            cc.emit(get_move_inst(type_id), asmjit::x86::ptr(addr), value);
          }
        } else {
          // Without pointer axis, use x86 scaled hardware loading
          int scale_shift { -1 };

          switch (stride) {
          case 1:  { scale_shift =  0; break; }
          case 2:  { scale_shift =  1; break; }
          case 4:  { scale_shift =  2; break; }
          case 8:  { scale_shift =  3; break; }
          default: { scale_shift = -1; break; }
          }

          if (scale_shift >= 0) {
            if (!addr_is_sink) {
              cc.emit(get_move_inst(type_id),
                      value,
                      asmjit::x86::ptr(addr, index, scale_shift));
            } else {
              cc.emit(get_move_inst(type_id),
                      asmjit::x86::ptr(addr, index, scale_shift),
                      value);
            }
          } else {
            // suboffset should be negative, but we otherwise determined it should
            // not be used, so pass -1 to emit instructions
            advance_soarray_axis(cc, addr, index, stride, -1);
            if (!addr_is_sink) {
              cc.emit(get_move_inst(type_id), value, asmjit::x86::ptr(addr));
            } else {
              cc.emit(get_move_inst(type_id), asmjit::x86::ptr(addr), value);
            }
          }
        }
      }
    }

    void advance_ncoffsets_pointer_axis(asmjit::x86::Compiler& cc,
                                        asmjit::x86::Gp& addr,
                                        asmjit::x86::Gp& index,
                                        ssize_t offset) {
      // For NCOffsetsPolicy for pointer axis dereference and load
      asmjit::x86::Gp ptr_idx { cc.new_gp(asmjit::TypeId::kInt64) };
      cc.mov(ptr_idx, index);
      if (offset != 0) {
        cc.add(ptr_idx, offset);
      }
      cc.shl(ptr_idx, 3); // Multiply by sizeof(void*) = 8 bytes (2^3)
      cc.add(addr, ptr_idx);
      // Now dereference
      cc.mov(addr, asmjit::x86::ptr(addr));
    }

    void advance_ncoffsets_strided_axis(asmjit::x86::Compiler& cc,
                                        asmjit::x86::Gp& addr,
                                        asmjit::x86::Gp& index,
                                        ssize_t stride,
                                        ssize_t offset) {
      asmjit::x86::Gp term { cc.new_gp(asmjit::TypeId::kInt64) };
      cc.mov(term, index);
      cc.imul(term, stride);
      cc.add(addr, term);
      if (offset > 0) {
        asmjit::x86::Gp term_off { cc.new_gp(asmjit::TypeId::kInt64) };
        cc.mov(term_off, offset);
        cc.add(addr, term_off);
      }
    }

    void advance_soarray_axis(asmjit::x86::Compiler& cc,
                              asmjit::x86::Gp& addr,
                              asmjit::x86::Gp& index,
                              ssize_t stride,
                              ssize_t suboffset) {
      // For SOArrayPolicy we first do normal strided traversal (index * stride)
      asmjit::x86::Gp term { cc.new_gp(asmjit::TypeId::kInt64) };
      cc.mov(term, index);
      cc.imul(term, stride);
      cc.add(addr, term);
      if (suboffset >= 0) {
        // Then, if it has a suboffset, dereference and add it
        cc.mov(addr, asmjit::x86::ptr(addr));
        if (suboffset != 0) {
          asmjit::x86::Gp term_sub { cc.new_gp(asmjit::TypeId::kInt64) };
          cc.mov(term_sub, suboffset);
          cc.add(addr, term_sub);
        }
      }
    }
  } // namespace host::x86
} // namespace ncarray
