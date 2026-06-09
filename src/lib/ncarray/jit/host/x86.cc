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
    asmjit::InstId get_binary_arithmetic_inst(OpCode op, asmjit::TypeId type_id) {
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
        }
      } else {
        asmjit::x86::Vec left_vec { left.as<asmjit::x86::Vec>() };
        asmjit::x86::Vec right_vec { right.as<asmjit::x86::Vec>() };

        if (type_id == asmjit::TypeId::kFloat32) {
          cc.ucomiss(left_vec, right_vec);
        } else {
          cc.ucomisd(left_vec, right_vec);
        }

        asmjit::x86::Gp tmp { cc.new_gp(asmjit::TypeId::kInt32) };
        cc.xor_(tmp, tmp);

        switch (op) {
        case OpCode::EQ: {
          cc.sete(tmp.r8());
          break;
        }
        case OpCode::NE: {
          cc.setne(tmp.r8());
          break;
        }
        case OpCode::LT: {
          cc.setb(tmp.r8());
          break;
        }
        case OpCode::LE: {
          cc.setbe(tmp.r8());
          break;
        }
        case OpCode::GT: {
          cc.seta(tmp.r8());
          break;
        }
        case OpCode::GE: {
          cc.setae(tmp.r8());
          break;
        }
        }

        asmjit::x86::Vec res_vec { res.as<asmjit::x86::Vec>() };

        if (type_id == asmjit::TypeId::kFloat32) {
          cc.cvtsi2ss(res_vec, tmp);
        } else {
          cc.cvtsi2sd(res_vec, tmp);
        }
      }
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
  } // namespace host
} // namespace ncarray
