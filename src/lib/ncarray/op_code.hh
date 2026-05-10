/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_OP_CODE_HH
#define NCARRAY_OP_CODE_HH

#include "ncarray/dtype.hh"
#include "ncarray/op_traits.hh"

#ifdef __CUDACC_RTC__
#include <cuda/std/cstdint>

using cuda::std::uint8_t;
using cuda::std::uint32_t;

#else

#include <cstdint>

using std::uint8_t;
using std::uint32_t;

#endif

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
  /**
   * OpCodes indicate types of operations or expressions.
   *
   * When interfacing with Python these are used for the virtual machine. In normal
   * C++ usage, they are used by the various BinaryExpr/UnaryExpr and so on.
   *
   * All supported operations (add, subtract, etc.) are included here, in addition
   * to the necessary codes to indicate loading/retrieval of the operands (arrays,
   * or constant scalars)
   */
  enum class OpCode: uint8_t {
    NOOP = 0,   ///< Null op
    IDX,        ///< Index generator (Like APL)
    LOAD_NCARR, ///< Load an NCArray (VM only)
    LOAD_SOARR, ///< Load an SOArray (VM only)
    LOAD_CONST, ///< Load a constant (VM only)
    // --- Unary ops --- //
    NEG,        ///< Negative
    INC,        ///< Increment
    DEC,        ///< Decrement
    SZOF,       ///< Size of
    ADDR,       ///< Address of
    INDR,       ///< Indirection/dereference
    CAST,       ///< Cast
    LNOT,       ///< Logical not
    BNOT,       ///< Bitwise not
    // --- Binary ops --- //
    // Arithmetic
    ADD,        ///< Addition
    SUB,        ///< Subtraction
    MUL,        ///< Multiplication
    DIV,        ///< True division
    MOD,        ///< Modulo
    FDIV,       ///< Floor (integer) division
    // Comparisons
    EQ,         ///< Equal to
    NE,         ///< Not equal
    LT,         ///< Less than
    LE,         ///< Less than or equal
    GT,         ///< Greater than
    GE,         ///< Greater than or equal
    // Logical
    LAND,       ///< Logical and
    LOR,        ///< Logical or
    // Bitwise
    BAND,       ///< Bitwise and (&)
    BOR,        ///< Bitwise or (|)
    XOR,        ///< Bitwise XOR (^)
    LSHFT,      ///< Left shift (<<)
    RSHFT       ///< Right shift (>>)
  };

  /**
   * For a specific operation and input datatype determine the correct result dtype.
   *
   * `op_traits` define certain semantics for type promotion or conversion
   * depending on the operation being performed. E.g. to avoid frequent overflows
   * very small integers (uint8, uint16) will be promoted to larger ones. The output
   * datatype for an ADD operation for uint8 will then be uint64 (int8 -> int64).
   *
   * @param[in] code The op code for this operation.
   * @param[in] left The datatype of the leftmost operand.
   * @param[in] res_dtype The appropriate output datatype for the operation.
   */
  NCA_HD inline DType determine_dtype_for_op(OpCode code, DType left) {
#ifndef __CUDACC_RTC__
    auto dtype_op = [&] <typename SrcT> () {
      switch (code) {
      // Unary operations
      case OpCode::NEG: {
        // NOTE: Do we want to auto-convert unsigned to signed?
        // TODO: Consider unsigned behaviour for negation
        return dtype_traits<SrcT>::value;
      }
      case OpCode::INC:
      case OpCode::DEC: {
        return dtype_traits<SrcT>::value;
      }
      case OpCode::LNOT: {
        return dtype_traits<bool>::value;
      }
      case OpCode::BNOT: {
        return dtype_traits<SrcT>::value;
      }
      // Arithmetic
      case OpCode::ADD: {
        using AccumT = typename op_traits<SrcT>::sum_type;
        return dtype_traits<AccumT>::value;
      }
      case OpCode::SUB: {
        using DiffT = typename op_traits<SrcT>::diff_type;
        return dtype_traits<DiffT>::value;
      }
      case OpCode::MUL: {
        return dtype_traits<SrcT>::value;
      }
      case OpCode::DIV: {
        using DivT = typename op_traits<SrcT>::truediv_type;
        return dtype_traits<DivT>::value;
      }
      case OpCode::MOD: {
        // TODO: Use truediv_type for modulo?
        return dtype_traits<SrcT>::value;
      }
      // Comparisons
      case OpCode::EQ:
      case OpCode::NE:
      case OpCode::LT:
      case OpCode::LE:
      case OpCode::GT:
      case OpCode::GE:
      // Logical
      case OpCode::LAND:
      case OpCode::LOR:
        return dtype_traits<bool>::value;
      default: {
        return dtype_traits<SrcT>::value;
      }
      }
    };
    return dispatch(left, dtype_op);
#else
    return left;
#endif
  }

  /**
   * Instructions for the virutal machine are stored in a packed format in 32 bit ints.
   *
   * The op code (e.g. ADD, LOAD_ARRAY etc) is stored in the lower 8 bits. The index of
   * the operand in the VM's stack is in the remaining bits.
   */
  using Instruction = uint32_t;

  /**
   * Pack an op code and an operand index into a single VM instruction.
   */
  NCA_HD inline uint32_t pack_instruction(OpCode op, int idx) {
    return (static_cast<uint32_t>(op) << 24) | (static_cast<uint32_t>(idx) & 0xFFFFFF);
  }

  /**
   * From a packed instruction retrieve the op code.
   */
  NCA_HD inline OpCode get_op(Instruction instr) {
    return static_cast<OpCode>(instr >> 24);
  }

  /**
   * From a packed instruction retrieve the operand stack index.
   */
  NCA_HD inline int get_index(Instruction instr) {
    return static_cast<int>(instr & 0xFFFFFF);
  }
} // namespace ncarray

#endif // NCARRAY_OP_CODE_HH
