/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "pyncarray/binding_builder.hh"

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

void register_sodevarray_view(py::module_& m) {
#ifdef NCA_HAS_CUDA
  auto sodevview_cls = py::classh<ncarray::SODevArrayView>(m, "SODevArrayView")
    // These constructors, plus the implicitly_convertible on SODevArrayRef and
    // SODevArray allow view interconversion
    .def(py::init<const ncarray::SODevArrayRef&>())
    .def(py::init<const ncarray::SODevArray&>());
  register_common_array_methods(sodevview_cls);
#endif
}
