/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_JIT_HOST_X86_HH
#define NCARRAY_JIT_HOST_X86_HH

#include "ncarray/dtype.hh"
#include "ncarray/layout.hh"
#include "ncarray/op_code.hh"
#include "ncarray/op_traits.hh"

#include <asmjit/core.h>
#include <asmjit/x86.h>

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
  namespace host::x86 {
    /**
     * Determine the appropriate x86 binary instruction given the operation and working type.
     *
     * This function will provide the correct instruction to emit given the type under
     * consideration. This accounts for signed/unsigned, width, and floating-point/integer
     * differences.
     *
     * @param[in] op The OpCode to emit an instruction for (ADD, SUB, etc.)
     * @param[in] type_id The asmjit type being operated on (kUInt8, kFloat32, etc.).
     * @returns instr The correct instruction to incorporate into the function.
     */
    NCA_JIT_API asmjit::InstId get_binary_inst(OpCode op, asmjit::TypeId type_id);

    /**
     * Emit the correct x86 load instruction for a scalar of the specified type.
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
    NCA_JIT_API asmjit::Reg load_constant(asmjit::x86::Compiler& cc,
                                          Scalar scalar,
                                          asmjit::TypeId type_id);

    /**
     * Emit the correct x86 move instruction for the specified type.
     *
     * This function will provide the correct instruction to emit for integers, or
     * floating point types of different precision.
     *
     * @param[in] type_id The asmjit type being operated on (kUInt8, kFloat32, etc.).
     * @returns inst The x86 move instruction appropriate for the operand type.
     */
    NCA_JIT_API asmjit::InstId get_move_inst(asmjit::TypeId type_id);

    /**
     * Advance the pointer for the underlying data of an NCArray* or SOArray* and
     * index into it using x86 scaled hardware addressing.
     *
     * This function takes two registers, allowing the caller to determine whether
     * the value of the non-array register should get stored in the array, or whether
     * the value from the array gets stored in the non-array register.
     *
     * The function will deal with correct dereferencing and offset handling for both
     * SOArrayPolicy and NCOffsetsPolicy type arrays. The scaled addressing makes use
     * of an x86 feature reducing instruction count when addressing a pointer using
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
    NCA_JIT_API void scaled_address_array(asmjit::x86::Compiler& cc,
                                          asmjit::x86::Gp& addr,
                                          asmjit::x86::Gp& index,
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
    NCA_JIT_API void advance_ncoffsets_pointer_axis(asmjit::x86::Compiler& cc,
                                                    asmjit::x86::Gp& addr,
                                                    asmjit::x86::Gp& index,
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
    NCA_JIT_API void advance_ncoffsets_strided_axis(asmjit::x86::Compiler& cc,
                                                    asmjit::x86::Gp& addr,
                                                    asmjit::x86::Gp& index,
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
    NCA_JIT_API void advance_soarray_axis(asmjit::x86::Compiler& cc,
                                          asmjit::x86::Gp& addr,
                                          asmjit::x86::Gp& index,
                                          ssize_t stride,
                                          ssize_t suboffset);
  } // namespace host
} // namespace ncarray

#endif // NCARRAY_JIT_HOST_X86_HH
