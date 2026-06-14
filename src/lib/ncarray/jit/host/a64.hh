/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_JIT_HOST_A64_HH
#define NCARRAY_JIT_HOST_A64_HH

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
typedef SSIZE_T ssize_t;

// On Windows need to export symbols for DLLs
#ifdef NCA_BUILD_JIT_API
#define NCA_JIT_API __declspec(dllexport)
#else
#define NCA_JIT_API __declspec(dllimport)
#endif
#else
#include <dlfcn.h>
#include <sys/types.h>

#define NCA_JIT_API
#endif

namespace ncarray {
  namespace host::a64 {
    /**
     * Determine the appropriate a64 instruction for the unary operation.
     *
     * This function will incorporate the correct a64 instruction to perform the unary
     * operation considering signed/unsignededness, width, and floating-point/integer
     * type of the operand.
     *
     * @param[in] cc The compiler building the function.
     * @param[in] op The OpCode for the unary operation.
     * @param[in] type_id The asmjit type being operated on (kUInt8, kFloat32, etc.).
     * @param[in] operand The register for the unary operand.
     * @param[in] res The register for the result.
     */
    NCA_JIT_API void unary_operation(asmjit::a64::Compiler& cc,
                                     OpCode op,
                                     asmjit::TypeId type_id,
                                     asmjit::Reg& operand,
                                     asmjit::Reg& res);

    /**
     * Determine the appropriate a64 arithmetic instruction given the operation and working type.
     *
     * This function will provide the correct instruction to emit given the type under
     * consideration. This accounts for signed/unsigned, width, and floating-point/integer
     * differences.
     *
     * @param[in] op The OpCode to emit an instruction for (ADD, SUB, etc.)
     * @param[in] type_id The asmjit type being operated on (kUInt8, kFloat32, etc.).
     * @returns instr The correct arithmetic instruction to incorporate into the function.
     */
    NCA_JIT_API asmjit::InstId get_binary_arithmetic_inst(OpCode op,
                                                          asmjit::TypeId type_id);

    /**
     * Perform integer division or modulo/remainder operations.
     *
     * For a64 these are calculated simultaneously, but must be calculated in specific
     * registers. Additionally, since the rest of the compiler setup has been setup
     * using virtual register allocation via the Compiler class, the collisions between
     * those virtual allocations and the references to the specific physical registers
     * must be handled. This function does this for all widths of integers, and signed/
     * unsigned handling.
     *
     * @param[in] cc The compiler building the function.
     * @param[in] op The DIV or MOD OpCode to place the result of in res.
     * @param[in] type_id The asmjit type being operated on (one of the integer types).
     * @param[in] left The register for the left-side operand.
     * @param[in] right The register for the right-side operand.
     * @param[in] res The register for the result.
     */
    NCA_JIT_API void integer_div_mod(asmjit::a64::Compiler& cc,
                                     OpCode op,
                                     asmjit::TypeId type_id,
                                     asmjit::Reg& left,
                                     asmjit::Reg& right,
                                     asmjit::Reg& res);

    /**
     * Perform the appropriate binary comparison.
     *
     * This function will emit the correct set of instructions accounting for signed or
     * unsigned values, different widths, and integer/floating-point differences.
     *
     * @param[in] cc The compiler building the function.
     * @param[in] op The OpCode to emit an instruction for (ADD, SUB, etc.)
     * @param[in] type_id The asmjit type being operated on (kUInt8, kFloat32, etc.).
     * @param[in] left The register for the left-side operand.
     * @param[in] right The register for the right-side operand.
     * @param[in] res The register for the result.
     */
    NCA_JIT_API void binary_compare(asmjit::a64::Compiler& cc,
                                    OpCode op,
                                    asmjit::TypeId type_id,
                                    asmjit::Reg& left,
                                    asmjit::Reg& right,
                                    asmjit::Reg& res);

    /**
     * Convert a register of one type to another.
     *
     * @param[in] cc The compiler building the function.
     * @param[in] src The register for the input source to be cast.
     * @param[in] src_type The asmjit type of the source register.
     * @param[in] dest_type The asmjit type of the requested destination register.
     * @returns casted The casted register
     */
    NCA_JIT_API asmjit::Reg cast_register(asmjit::a64::Compiler& cc,
                                          asmjit::Reg src,
                                          asmjit::TypeId src_type,
                                          asmjit::TypeId dest_type);

    /**
     * Emit the correct a64 load instruction for a scalar of the specified type.
     *
     * This function will provide the correct instruction to emit given the type under
     * consideration. This accounts for signed/unsigned, width, and floating-point/integer
     * differences.
     *
     * @param[in] cc The compiler building the function.
     * @param[in] scalar The scalar to load.
     * @param[in] type_id The asmjit type being operated on (kUInt8, kFloat32, etc.).
     * @returns reg The virtual register with the correct load instruction.
     */
    NCA_JIT_API asmjit::Reg load_constant(asmjit::a64::Compiler& cc,
                                          Scalar scalar,
                                          asmjit::TypeId type_id);

    /**
     * Emit the correct a64 move instruction for the specified type.
     *
     * This function will provide the correct instruction to emit for integers, or
     * floating point types of different precision.
     *
     * @param[in] type_id The asmjit type being operated on (kUInt8, kFloat32, etc.).
     * @returns inst The a64 move instruction appropriate for the operand type.
     */
    NCA_JIT_API asmjit::InstId get_move_inst(asmjit::TypeId type_id);

    /**
     * Emit the correct A64 instruction to load a value from memory into a register.
     *
     * This function will perform the correct conversion to general purpose or
     * vector register of the destination register as necessary.
     *
     * @param[in] cc The compiler constructing the code.
     * @param[in] dest The virtual register that is the destination for the load.
     * @param[in] src The memory location that is the source of the load.
     */
    NCA_JIT_API void emit_load(asmjit::a64::Compiler& cc,
                               asmjit::Reg& dest,
                               const asmjit::a64::Mem& src);

    /**
     * Emit the correct A64 instruction to store a value from a register into memory.
     *
     * This function will perform the correct conversion to general purpose or
     * vector register of the source register as necessary.
     *
     * (NOTE: The fact that the asmjit mem location dest is const does not mean
     * that it cannot be stored to.)
     *
     * @param[in] cc The compiler constructing the code.
     * @param[in] dest The memory location that is the destination of the store.
     * @param[in] src The virtual register that is the source of the store.
     */
    NCA_JIT_API void emit_store(asmjit::a64::Compiler& cc,
                                const asmjit::a64::Mem& dest,
                                asmjit::Reg& src);


    /**
     * Advance the pointer for the underlying data of an NCArray* or SOArray* and
     * index into it using a64 scaled hardware addressing.
     *
     * This function takes two registers, allowing the caller to determine whether
     * the value of the non-array register should get stored in the array, or whether
     * the value from the array gets stored in the non-array register.
     *
     * The function will deal with correct dereferencing and offset handling for both
     * SOArrayPolicy and NCOffsetsPolicy type arrays. The scaled addressing makes use
     * of an a64 feature reducing instruction count when addressing a pointer using
     * the form [base + index * scale] and the scale multiplier is 1, 2, 4 or 8.
     * This is only applied for those cases, and for the non-pointer axis case. In all
     * other cases, the full set of multiply/add instructions are used as a fallback.
     *
     * @param[in] cc The compiler constructing the code.
     * @param[in] addr The address pointer for the array.
     * @param[in] index The index (linearized) for the element to index from the array.
     * @param[in] value The additional register to either store into the array, or hold
     *            the value from the array.
     * @param[in] dim The dimension of the array being accessed (for stride, offset lookup, etc.)
     * @param[in] arr_layout The array's layout.
     * @param[in] type_id The array's data type (in the asmjit type system).
     * @param[in] addr_is_sink If true, after correct addressing, store `value` into
     *            addr. If false, after addressing, store `addr` into `value`.
     * @param[in] expr_is_soarr As layout is always passed as `SOArrayPolicy`, if false,
     *            convert to `NCOffsetsPolicy`.
     */
    NCA_JIT_API void scaled_address_array(asmjit::a64::Compiler& cc,
                                          asmjit::a64::Gp& addr,
                                          asmjit::a64::Gp& index,
                                          asmjit::Reg& value,
                                          ssize_t dim,
                                          const SOArrayPolicy& arr_layout,
                                          asmjit::TypeId type_id,
                                          bool addr_is_sink = false,
                                          bool expr_is_soarr = false);

    /**
     * Advance the pointer for the underlying data of an NCArray*.
     *
     * This function expects the pointer to have already been advanced along the
     * prior dimensions. It will advance it further for the current axis, using
     * the provided index, stride and offset.
     *
     * This function moves along a POINTER axis of an NCArray*. This must be
     * checked by the caller. For pointer axes, use the advance_ncoffsets_strided_axis
     * function.
     *
     * @param[in] cc The compiler constructing the code.
     * @param[in] addr The address pointer for the array.
     * @param[in] index The index (linearized) for the element to index from the array.
     * @param[in] offset The offset for the pointer axis. Should be >= 0. This is not checked.
     */
    NCA_JIT_API void advance_ncoffsets_pointer_axis(asmjit::a64::Compiler& cc,
                                                    asmjit::a64::Gp& addr,
                                                    asmjit::a64::Gp& index,
                                                    ssize_t offset);

    /**
     * Advance the pointer for the underlying data of an NCArray*.
     *
     * This function expects the pointer to have already been advanced along the
     * prior dimensions. It will advance it further for the current axis, using
     * the provided index, stride and offset.
     *
     * This function moves along a NON-pointer axis of an NCArray*. This must be
     * checked by the caller. For pointer axes, use the advance_ncoffsets_pointer_axis
     * function.
     *
     * @param[in] cc The compiler constructing the code.
     * @param[in] addr The address pointer for the array.
     * @param[in] index The index (linearized) for the element to index from the array.
     * @param[in] stride The stride for the selected axis.
     * @param[in] offset The offset for the non-pointer axis.
     */
    NCA_JIT_API void advance_ncoffsets_strided_axis(asmjit::a64::Compiler& cc,
                                                    asmjit::a64::Gp& addr,
                                                    asmjit::a64::Gp& index,
                                                    ssize_t stride,
                                                    ssize_t offset);

    /**
     * Advance the pointer for the underlying data of an SOArray*.
     *
     * This function expects the pointer to have already been advanced along the
     * prior dimensions. It will advance it further for the current axis, using
     * the provided index, stride and suboffset.
     *
     * @param[in] cc The compiler constructing the code.
     * @param[in] addr The address pointer for the array.
     * @param[in] index The index for the element to index from the array.
     * @param[in] stride The stride for the current axis.
     * @param[in] offset The suboffset for the current axis.
     */
    NCA_JIT_API void advance_soarray_axis(asmjit::a64::Compiler& cc,
                                          asmjit::a64::Gp& addr,
                                          asmjit::a64::Gp& index,
                                          ssize_t stride,
                                          ssize_t suboffset);
  } // namespace host::a64
} // namespace ncarray

#endif // NCARRAY_JIT_HOST_A64_HH
