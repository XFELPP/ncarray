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

#include <pybind11/native_enum.h>
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

PYBIND11_MODULE(_pyncarray, ncarray_module, py::mod_gil_not_used()) {
  ncarray_module.doc() = "Non-contiguous (NC) Array Classes.";

  py::native_enum<ncarray::DType>(ncarray_module,
                                  "DType",
                                  "enum.Enum",
                                  "Type enumerators supported by ncarray.")
    .value("bool", ncarray::DType::bool_)
    .value("char", ncarray::DType::char_)
    .value("uint8", ncarray::DType::uint8)
    .value("uint16", ncarray::DType::uint16)
    .value("uint32", ncarray::DType::uint32)
    .value("uint64", ncarray::DType::uint64)
    .value("int8", ncarray::DType::int8)
    .value("int16", ncarray::DType::int16)
    .value("int32", ncarray::DType::int32)
    .value("int64", ncarray::DType::int64)
    .value("float32", ncarray::DType::float32)
    .value("float64", ncarray::DType::float64)
    .value("complex64", ncarray::DType::complex64)
    .value("complex128", ncarray::DType::complex128)
    .value("complex256", ncarray::DType::complex256)
    .value("vfloat2", ncarray::DType::vfloat2)
    .value("vfloat3", ncarray::DType::vfloat3)
    .value("vfloat4", ncarray::DType::vfloat4)
    .value("vdouble2", ncarray::DType::vdouble2)
    .value("vdouble3", ncarray::DType::vdouble3)
    .value("vdouble4", ncarray::DType::vdouble4)
    .export_values()
    .finalize();

  // --- Simple 2,3,4 float vectors (for CPU/GPU similarity) --- //
  py::classh<ncarray::Float2>(ncarray_module, "Float2", "A simple 2 float vector.")
    .def(py::init<float, float>())
    .def_readwrite("x", &ncarray::Float2::x)
    .def_readwrite("y", &ncarray::Float2::y);

  py::classh<ncarray::Float3>(ncarray_module, "Float3", "A simple 3 float vector.")
    .def(py::init<float, float, float>())
    .def_readwrite("x", &ncarray::Float3::x)
    .def_readwrite("y", &ncarray::Float3::y)
    .def_readwrite("z", &ncarray::Float3::z);

  py::classh<ncarray::Float4>(ncarray_module, "Float4", "A simple 4 float vector.")
    .def(py::init<float, float, float, float>())
    .def_readwrite("x", &ncarray::Float4::x)
    .def_readwrite("y", &ncarray::Float4::y)
    .def_readwrite("z", &ncarray::Float4::z)
    .def_readwrite("w", &ncarray::Float4::w);

  py::classh<ncarray::Double2>(ncarray_module, "Double2", "A simple 2 double vector.")
    .def(py::init<double, double>())
    .def_readwrite("x", &ncarray::Double2::x)
    .def_readwrite("y", &ncarray::Double2::y);

  py::classh<ncarray::Double3>(ncarray_module, "Double3", "A simple 3 double vector.")
    .def(py::init<double, double, double>())
    .def_readwrite("x", &ncarray::Double3::x)
    .def_readwrite("y", &ncarray::Double3::y)
    .def_readwrite("z", &ncarray::Double3::z);

  py::classh<ncarray::Double4>(ncarray_module, "Double4", "A simple 4 double vector.")
    .def(py::init<double, double, double, double>())
    .def_readwrite("x", &ncarray::Double4::x)
    .def_readwrite("y", &ncarray::Double4::y)
    .def_readwrite("z", &ncarray::Double4::z)
    .def_readwrite("w", &ncarray::Double4::w);

  // --- Namesake NCArray* array classes --- //
  auto ncview_cls = py::classh<ncarray::NCArrayView>(ncarray_module, "NCArrayView");
  register_common_array_methods(ncview_cls);

  auto ncref_cls = py::classh<ncarray::NCArrayRef>(ncarray_module, "NCArrayRef")
    .def(py::init([](const py::array& arr, const bool read_only = false) {
      return pyarray_to_ref<ncarray::NCOffsetsPolicy>(arr, read_only);
    }),
      py::arg("data"),
      py::arg("read_only") = py::cast(false))
    .def(py::init([](const py::list& list, const bool read_only = false) {
      return pylist_to_ref<ncarray::NCOffsetsPolicy>(list, read_only);
    }),
      py::arg("data"),
      py::arg("read_only") = py::cast(false));
  register_common_array_methods(ncref_cls);

  auto ncowner_cls = py::classh<ncarray::NCArray>(ncarray_module, "NCArray")
    .def(py::init([](const std::vector<ssize_t>& shape, const ncarray::DType& dtype) {
      return new ncarray::NCArray(shape, dtype);
    }),
      py::arg("shape"),
      py::arg("dtype") = py::cast(ncarray::DType::float32));
  register_common_array_methods(ncowner_cls);

  // --- Suboffsets support --- //
  auto soview_cls = py::classh<ncarray::SOArrayView>(ncarray_module, "SOArrayView");
  register_common_array_methods(soview_cls);

  auto soref_cls = py::classh<ncarray::SOArrayRef>(ncarray_module, "SOArrayRef")
    .def(py::init([](const py::array& arr, const bool read_only = false) {
      return pyarray_to_ref<ncarray::SOArrayPolicy>(arr, read_only);
    }),
      py::arg("data"),
      py::arg("read_only") = py::cast(false))
    .def(py::init([](const py::list& list, const bool read_only = false) {
      return pylist_to_ref<ncarray::SOArrayPolicy>(list, read_only);
    }),
      py::arg("data"),
      py::arg("read_only") = py::cast(false));
  register_common_array_methods(soref_cls);

  auto soowner_cls = py::classh<ncarray::SOArray>(ncarray_module, "SOArray")
    .def(py::init([](const std::vector<ssize_t>& shape, const ncarray::DType& dtype) {
      return new ncarray::SOArray(shape, dtype);
    }),
      py::arg("shape"),
      py::arg("dtype") = py::cast(ncarray::DType::float32));
  register_common_array_methods(soowner_cls);
} // ncarray_module
