/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "python/binding_builder.hh"

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

namespace {
  py::object materialize(const py::object& expr) {
    if (py::isinstance<ncarray::ExprMVNode<ncarray::HostTag>>(expr) ||
        py::isinstance<ncarray::ExprMVNode<ncarray::DevTag>>(expr)) {
#ifdef NCA_HAS_CUDA
      if (needs_device_vm(expr)) {
        return pyncarray::eval_python_expr<ncarray::DevTag>(expr);
      }
#endif
      return pyncarray::eval_python_expr<ncarray::HostTag>(expr);
    }
    // If not an expression just pass through, essentially a no-op
    return expr;
  }
} // anonymous namespace

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

  py::native_enum<ncarray::OpCode>(ncarray_module,
                                   "OpCode",
                                   "enum.Enum",
                                   "Operations supported by ncarray.")
    .value("NOOP", ncarray::OpCode::NOOP)   ///< Null Op
    .value("IDX", ncarray::OpCode::IDX)     ///< Index generator (APL)
    // Load operations (VM only)
    .value("LOAD_NCARR", ncarray::OpCode::LOAD_NCARR) ///< Load NCArray
    .value("LOAD_SOARR", ncarray::OpCode::LOAD_SOARR) ///< Load SOArray
    .value("LOAD_CONST", ncarray::OpCode::LOAD_CONST) ///< Load a constant
    // --- Unary Ops --- //
    .value("NEG", ncarray::OpCode::NEG)     ///< Negative of the value
    .value("INC", ncarray::OpCode::INC)     ///< Increment
    .value("DEC", ncarray::OpCode::DEC)     ///< Decrement
    .value("SZOF", ncarray::OpCode::SZOF)   ///< Sizeof
    .value("ADDR", ncarray::OpCode::ADDR)   ///< Address of
    .value("INDR", ncarray::OpCode::INDR)   ///< Indirection/dereference
    .value("CAST", ncarray::OpCode::CAST)   ///< Cast
    .value("LNOT", ncarray::OpCode::LNOT)   ///< Logical not
    .value("BNOT", ncarray::OpCode::BNOT)   ///< Bitwise not
    // --- Binary Ops --- //
    // Arithmetic
    .value("ADD", ncarray::OpCode::ADD)     ///< Addition
    .value("SUB", ncarray::OpCode::SUB)     ///< Subtraction
    .value("MUL", ncarray::OpCode::MUL)     ///< Multiplication
    .value("DIV", ncarray::OpCode::DIV)     ///< (True) division
    .value("FDIV", ncarray::OpCode::FDIV)   ///< Floor/integer division
    // Comparisons
    .value("EQ", ncarray::OpCode::EQ)       ///< Is equal
    .value("NE", ncarray::OpCode::NE)       ///< Not equal
    .value("LT", ncarray::OpCode::LT)       ///< Less than
    .value("LE", ncarray::OpCode::LE)       ///< Less than or equal
    .value("GT", ncarray::OpCode::GT)       ///< Greater than
    .value("GE", ncarray::OpCode::GE)       ///< Greater than or equal
    // (Binary) Logical
    .value("LAND", ncarray::OpCode::LAND)   ///< Logical and
    .value("LOR", ncarray::OpCode::LOR)     ///< Logical or
    // Bitwise
    .value("BAND", ncarray::OpCode::BAND)   ///< Bitwise and (&)
    .value("BOR", ncarray::OpCode::BOR)     ///< Bitwise or (|)
    .value("XOR", ncarray::OpCode::XOR)     ///< Bitwise XOR (^)
    .value("LSHFT", ncarray::OpCode::LSHFT) ///< Left shift (<<)
    .value("RSHFT", ncarray::OpCode::RSHFT) ///< Right shift (>>)
    .export_values()
    .finalize();

  // --- Simple 2,3,4 float vectors (for CPU/GPU similarity) --- //
#define BIND_VEC_TYPE(SNAME, BASE_TYPE, BINDNAME)        \
  py::classh<ncarray::SNAME##2>(ncarray_module, #BINDNAME"2", "A simple 2 " #BASE_TYPE " vector.") \
    .def(py::init<BASE_TYPE, BASE_TYPE>())                                                         \
    .def_readwrite("x", &ncarray::SNAME##2::x)                                                     \
    .def_readwrite("y", &ncarray::SNAME##2::y);                                                    \
                                                                                                   \
  py::classh<ncarray::SNAME##3>(ncarray_module, #BINDNAME"3", "A simple 3 " #BASE_TYPE " vector.") \
    .def(py::init<BASE_TYPE, BASE_TYPE, BASE_TYPE>())                                              \
    .def_readwrite("x", &ncarray::SNAME##3::x)                                                     \
    .def_readwrite("y", &ncarray::SNAME##3::y)                                                     \
    .def_readwrite("z", &ncarray::SNAME##3::z);                                                    \
                                                                                                   \
  py::classh<ncarray::SNAME##4>(ncarray_module, #BINDNAME"4", "A simple 4 " #BASE_TYPE " vector.") \
    .def(py::init<BASE_TYPE, BASE_TYPE, BASE_TYPE, BASE_TYPE>())                                   \
    .def_readwrite("x", &ncarray::SNAME##4::x)                                                     \
    .def_readwrite("y", &ncarray::SNAME##4::y)                                                     \
    .def_readwrite("z", &ncarray::SNAME##4::z)                                                     \
    .def_readwrite("w", &ncarray::SNAME##4::w);

  BIND_VEC_TYPE(Float, float, Float)
  BIND_VEC_TYPE(Double, double, Double)

#undef BIND_VEC_TYPE


  pyncarray::register_expr_class<ncarray::HostTag>(ncarray_module, "HostExpr");

  ncarray_module.def("materialize", &materialize, py::arg("expr"));
  ncarray_module.def("set_eager",
                     &pyncarray::set_eager,
                     py::arg("is_eager"),
                     "Toggle whether expressions are eagerily evaluated to arrays or not.");
  ncarray_module.def("is_eager",
                     &pyncarray::is_eager,
                     "Whether expressions are currently being eagerily evaluated.");

  // --- Namesake NCArray* array classes --- //
  auto ncview_cls = py::classh<ncarray::NCArrayView>(ncarray_module, "NCArrayView")
    // These constructors, plus the implicitly_convertible below allow view interconversion
    .def(py::init<const ncarray::NCArrayRef&>())
    .def(py::init<const ncarray::NCArray&>());
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
  auto soview_cls = py::classh<ncarray::SOArrayView>(ncarray_module, "SOArrayView")
    // These constructors, plus the implicitly_convertible below allow view interconversion
    .def(py::init<const ncarray::SOArrayRef&>())
    .def(py::init<const ncarray::SOArray&>());
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

  // Make the non-view classes implicitly convertible to views
  py::implicitly_convertible<ncarray::NCArrayRef, ncarray::NCArrayView>();
  py::implicitly_convertible<ncarray::NCArray, ncarray::NCArrayView>();

  py::implicitly_convertible<ncarray::SOArrayRef, ncarray::SOArrayView>();
  py::implicitly_convertible<ncarray::SOArray, ncarray::SOArrayView>();
} // ncarray_module
