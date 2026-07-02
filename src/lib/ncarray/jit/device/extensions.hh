/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_JIT_DEVICE_EXTENSIONS_HH
#define NCARRAY_JIT_DEVICE_EXTENSIONS_HH

#ifndef __CUDACC_RTC__
#include <string>

namespace ncarray {
  namespace device {
    /**
     * A stencil extension provides a mechanism to incorporate extra code for the Stencil.
     *
     * When using a StencilJITExtensions object, the user can include prologue and epilogue
     * code which will be run before and/or after the main stencil expression's evaluation.
     * The extra parameters must also be included if making use of this facility.
     *
     * Parameters MUST be provided in a specific format begun by a comma. This is
     * because they will be inserted into a string of C++ code that will be compiled.
     * The prologue code will be run AFTER the thread index is determined. If the
     * thread index goes beyond the size of the array, then the kernel returns.
     * The epilogue code will be run following the expression evaluation.
     *
     * A trivial example setup may look like:
     * @code
     * // Inclusion of newlines and formatting is just for ease of viewing.
     * // It is not required.
     * ncarray::device::StencilJITExtensions ext;
     * ext.extra_params =
     *   ", unsigned* extra_ptr"; // NOTE: See the starting comma!
     *
     * ext.prologue_code =
     *   "if ((extra_ptr != nullptr) && (*extra_ptr == 42)) {\n"
     *   "  return;\n"
     *   "}\n";
     *
     * ext.epilogue_code =
     *   "if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
     *   "  *extra_ptr = 42;\n"
     *   "}\n";
     * @endcode
     *
     * This code will be inserted as follows into the compiled Stencil kernel:
     * @code
     * __global__ void stencil_kernel(const void* data,
     *                                //other stencil params,
     *                                //ext.extra_params here!!!
     *                                , int s0) { // Scalars at the end
     *   unsigned b_idx { 42 }; // Get index
     *   if (b_idx >= dest_l.size()) {
     *     return; // This will return WITHOUT running prologue code!
     *   }
     *
     *   // <--- ext.prologue_code --->
     *
     *   // <--- run normal Stencil expression evaluation --->
     *
     *   // <--- ext.prologue_code --->
     *  }
     * @endcode
     */
    struct StencilJITExtensions {
      std::string extra_params;  ///< Parameters needed to run the pro/epilogue code.
      std::string prologue_code; ///< Code to run before the stencil expression.
      std::string epilogue_code; ///< Code to run after the stencil expression.
    };
  } // namespace device
} // namespace ncarray
#endif // __CUDACC_RTC__

#endif // NCARRAY_JIT_DEVICE_EXTENSIONS_HH
