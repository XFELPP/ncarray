/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "python/utilities.hh"

#include "ncarray/custom_types.hh"
#include "ncarray/ncarrays.hh"
#include "ncarray/soarrays.hh"
#ifdef NCA_HAS_CUDA
// This code doesn't actually have anything GPU-specific (__device__ or kernels)
// They are the specializations for GPU arrays
#include "ncarray/ncdevarrays.cuh"
#include "ncarray/sodevarrays.cuh"
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <vector>

namespace py = pybind11;

using namespace pyncarray;

// NOTE: For faster/more efficient builds, each array specialization is in its own TU
// They're all combined here into a single module
void register_ncdevarray_view(py::module_& m);
void register_ncdevarray_ref(py::module_& m);
void register_ncdevarray_owner(py::module_& m);
void register_sodevarray_view(py::module_& m);
void register_sodevarray_ref(py::module_& m);
void register_sodevarray_owner(py::module_& m);

PYBIND11_MODULE(_pyncdevarray, ncdevarray_module, py::mod_gil_not_used()) {
  ncdevarray_module.doc() = "Non-contiguous (NC) GPU Array Classes.";
#ifdef NCA_HAS_CUDA
  register_ncdevarray_view(ncdevarray_module);
  register_ncdevarray_ref(ncdevarray_module);
  register_ncdevarray_owner(ncdevarray_module);
  register_sodevarray_view(ncdevarray_module);
  register_sodevarray_ref(ncdevarray_module);
  register_sodevarray_owner(ncdevarray_module);
#endif
} // ncdevarray_module
