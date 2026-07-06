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

void register_sodevarray_ref(py::module_& m) {
#ifdef NCA_HAS_CUDA
  auto sodevref_cls = py::classh<ncarray::SODevArrayRef>(m, "SODevArrayRef")
    .def(py::init([](const py::array& arr, const bool read_only = false) {
      return
        pyarray_to_ref<ncarray::SOArrayPolicy, ncarray::DevRefPolicy>(arr, read_only);
    }),
      py::arg("data"),
      py::arg("read_only") = py::cast(false))
    .def(py::init([](const py::list& list, const bool read_only = false) {
      return
        pylist_to_ref<ncarray::SOArrayPolicy, ncarray::DevRefPolicy>(list, read_only);
    }),
      py::arg("data"),
      py::arg("read_only") = py::cast(false));
  register_common_array_methods(sodevref_cls);

  py::implicitly_convertible<ncarray::SODevArrayRef, ncarray::SODevArrayView>();
#endif
}
