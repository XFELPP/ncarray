#define PYBIND11_DETAILED_ERROR_MESSAGES
#include "dtype.hh"
#include "indexing.hh"
#include "ncarrayref.hh"
#include "ncarrayview.hh"
#include "ncarray.hh"

#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

namespace py = pybind11;

namespace {
  ncarray::DType pydtype_to_dtype(py::dtype dtype) {
    if (dtype.is(py::dtype::of<std::uint8_t>())) {
      return ncarray::dtype_traits<std::uint8_t>::value;
    } else if (dtype.is(py::dtype::of<std::uint16_t>())) {
      return ncarray::dtype_traits<std::uint16_t>::value;
    } else if (dtype.is(py::dtype::of<std::uint32_t>())) {
      return ncarray::dtype_traits<std::uint32_t>::value;
    } else if (dtype.is(py::dtype::of<std::uint64_t>())) {
      return ncarray::dtype_traits<std::uint64_t>::value;
    } else if (dtype.is(py::dtype::of<std::int8_t>())) {
      return ncarray::dtype_traits<std::int8_t>::value;
    } else if (dtype.is(py::dtype::of<std::int16_t>())) {
      return ncarray::dtype_traits<std::int16_t>::value;
    } else if (dtype.is(py::dtype::of<std::int32_t>())) {
      return ncarray::dtype_traits<std::int32_t>::value;
    } else if (dtype.is(py::dtype::of<std::int64_t>())) {
      return ncarray::dtype_traits<std::int64_t>::value;
    } else if (dtype.is(py::dtype::of<bool>())) {
      return ncarray::dtype_traits<bool>::value;
    } else if (dtype.is(py::dtype::of<float>())) {
      return ncarray::dtype_traits<float>::value;
    } else if (dtype.is(py::dtype::of<double>())) {
      return ncarray::dtype_traits<double>::value;
    }
    std::ostringstream oss;
    oss << "Unsupported type: " << dtype << "!";
    throw py::type_error(oss.str());
  }

  ncarray::Slice pyslice_to_slice(ssize_t axis_shape, py::slice slice) {
    ssize_t start, stop, step, length;
    if (!slice.compute(axis_shape, &start, &stop, &step, &length)) {
      throw py::error_already_set();
    }
    return ncarray::Slice(start, stop, step);
  }

  auto pyarray_to_view(const py::array& arr) {
    py::buffer_info info = arr.request();
    auto buf_ptr = info.ptr;
    // This is temporary!! Careful how you use it!
    ssize_t ptr_axis { -1 }; // -1 means NO pointer axis
    return ncarray::NCArrayView(&buf_ptr,
                                arr.ndim(),
                                arr.shape(),
                                arr.strides(),
                                pydtype_to_dtype(arr.dtype()),
                                ptr_axis);
  }

  ncarray::NCArrayRef* pyarray_to_ref(const py::array& arr) {
    py::buffer_info arr_info = arr.request();
    ssize_t ndim { arr.ndim() };
    std::vector<ssize_t> shape(arr.shape(), arr.shape() + ndim);
    std::vector<ssize_t> strides(arr.strides(), arr.strides() + ndim);

    shape.insert(shape.begin(), 1);
    strides.insert(strides.begin(), 1);

    std::vector<void*> data_ptrs { arr_info.ptr };
    ncarray::DType dtype = pydtype_to_dtype(arr.dtype());
    return new ncarray::NCArrayRef(data_ptrs, shape, strides, dtype);
  }

  ncarray::NCArrayRef* pylist_to_ref(const py::list& list) {
    ssize_t len_ptr_axis = list.size();
    std::vector<ssize_t> strides { 1 };
    std::vector<ssize_t> shape { len_ptr_axis };
    std::vector<void*> data_ptrs;

    // Assume each array in list is same shape for now
    py::dtype pydtype;
    bool assigned { false };
    for (py::handle o : list) {
      py::array data = py::cast<py::array>(o);
      py::buffer_info arr_info = data.request();
      data_ptrs.push_back(arr_info.ptr);
      if (!assigned) {
        for (ssize_t i = 0; i < data.ndim(); ++i) {
          strides.push_back(data.strides()[i]);
          shape.push_back(data.shape()[i]);
        }
        assigned = true;
      }
      pydtype = data.dtype();
    }
    ncarray::DType dtype = pydtype_to_dtype(pydtype);
    return new ncarray::NCArrayRef(data_ptrs, shape, strides, dtype);
  }
} // anonymous namespace

PYBIND11_MODULE(ncarray, ncarray_module, py::mod_gil_not_used()) {
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
    .export_values()
    .finalize();

  py::classh<ncarray::NCArrayView>(ncarray_module, "NCArrayView")
    .def("__repr__", &ncarray::NCArrayView::repr)
    .def_property_readonly("shape", [](const ncarray::NCArrayView& self) -> py::tuple {
      auto* shape = self.shape();
      py::list l;
      for (ssize_t i = 0; i < self.ndim(); ++i) {
        l.append(shape[i]);
      }
      return l;
    })
    .def_property_readonly("strides", [](const ncarray::NCArrayView& self) -> py::tuple {
      auto* strides = self.strides();
      py::list l;
      for (ssize_t i = 0; i < self.ndim(); ++i) {
        l.append(strides[i]);
      }
      return l;
    })
    .def_property_readonly("size", &ncarray::NCArrayView::size)
    .def_property_readonly("ndim", &ncarray::NCArrayView::ndim)
    .def_property_readonly("itemsize", &ncarray::NCArrayView::itemsize)
    .def_property_readonly("nbytes", &ncarray::NCArrayView::nbytes)
    .def("__getitem__",
         [](const ncarray::NCArrayView& self, py::object idx) {
           if (py::isinstance<py::int_>(idx)) {
             return py::cast(self[idx.cast<ssize_t>()]);
           } else if (py::isinstance<py::slice>(idx)) {
             return py::cast(self[pyslice_to_slice(self.shape(0), idx.cast<py::slice>())]);
           } else if (py::isinstance<py::tuple>(idx)) {
             ncarray::ArrayIndices indices;
             ssize_t axis { 0 };
             for (auto& item : idx.cast<py::tuple>()) {
               if (py::isinstance<py::int_>(item)) {
                 indices.emplace_back(item.cast<ssize_t>());
               } else if (py::isinstance<py::slice>(item)) {
                 indices.emplace_back(pyslice_to_slice(self.shape(axis),
                                                       item.cast<py::slice>()));
               } else if (item.is(py::ellipsis())) {
                 indices.emplace_back(ncarray::Ellipsis{});
               } else {
                 throw py::type_error("Invalid indexing argument!");
               }
             }
             return py::cast(self[indices]);
           } else {
             throw py::type_error("Invalid indexing argument!");
           }
         },
         py::is_operator())
    .def("sum", &ncarray::NCArrayView::sum)
    .def("max", &ncarray::NCArrayView::max)
    .def("min", &ncarray::NCArrayView::min)
    .def("mean", &ncarray::NCArrayView::mean)
    // Conversion helpers for py::array shouldn't really be needed... TODO
    .def("__add__", [](const ncarray::NCArrayView& self, const ncarray::NCArrayView& other) {
      return py::cast(self + other);
    })
    .def("__add__", [](const ncarray::NCArrayView& self, const py::array& other) {
      return py::cast(self + pyarray_to_view(other));
    },
      py::is_operator())
    .def("__mul__", [](const ncarray::NCArrayView& self, const ncarray::NCArrayView& other) {
      return py::cast(self * other);
    },
      py::is_operator())
    .def("__mul__", [](const ncarray::NCArrayView& self, const py::array& other) {
      return py::cast (self * pyarray_to_view(other));
    },
      py::is_operator())
    .def("__truediv__", [](const ncarray::NCArrayView& self, const ncarray::NCArrayView& other) {
      return py::cast(self / other);
    },
      py::is_operator())
    .def("__truediv__", [](const ncarray::NCArrayView& self, const py::array& other) {
      return py::cast(self / pyarray_to_view(other));
    },
      py::is_operator());

  py::classh<ncarray::NCArrayRef, ncarray::NCArrayView>(ncarray_module, "NCArrayRef")
    .def(py::init([](const py::array& arr) {
      return pyarray_to_ref(arr);
    }))
    .def(py::init([](const py::list& list) {
      return pylist_to_ref(list);
    }));

  py::classh<ncarray::NCArray, ncarray::NCArrayView>(ncarray_module, "NCArray")
    .def(py::init([](const std::vector<ssize_t>& shape, const ncarray::DType& dtype) {
      return new ncarray::NCArray(shape, dtype);
    }),
      py::arg("shape"),
      py::arg("dtype") = py::cast(ncarray::DType::float32));
} // ncarray_module
