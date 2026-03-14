#include "dtype.hh"
#include "indexing.hh"
#include "ncarrayref.hh"
#include "ncarrayview.hh"
#include "ncarray.hh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
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
  }

  ncarray::Slice pyslice_to_slice(py::slice s) {
    return ncarray::Slice(s.attr("start").cast<ssize_t>(),
                          s.attr("stop").cast<ssize_t>(),
                          s.attr("step").is_none() ? 1 : s.attr("step").cast<ssize_t>());
  }
}

PYBIND11_MODULE(ncarray, ncarray_module, py::mod_gil_not_used()) {
  ncarray_module.doc() = "Non-contiguous (NC) Array Classes.";

  //py::classh<ncarray::NCArray>(ncarray_module, "NCArray");
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
             return py::cast(self[pyslice_to_slice(idx.cast<py::slice>())]);
           } else if (py::isinstance<py::tuple>(idx)) {
             ncarray::ArrayIndices indices;
             for (auto& item : idx.cast<py::tuple>()) {
               if (py::isinstance<py::int_>(item)) {
                 indices.emplace_back(item.cast<ssize_t>());
               } else if (py::isinstance<py::slice>(item)) {
                 indices.emplace_back(pyslice_to_slice(item.cast<py::slice>()));
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
    .def("__add__",
         [](const ncarray::NCArrayView& self, py::object other) {
           if (py::isinstance<py::array>(other)) {
             auto arr = py::reinterpret_borrow<py::array>(other);
             py::buffer_info info = arr.request();
             auto buf_ptr = info.ptr;
             ssize_t ptr_axis { -1 }; // -1 means NO pointer axis
             ncarray::NCArrayView view(&buf_ptr,
                                       arr.ndim(),
                                       arr.shape(),
                                       arr.strides(),
                                       pydtype_to_dtype(arr.dtype()),
                                       ptr_axis);
             return py::cast(self.add(view));
           }
           throw py::type_error("Only array types accepted!");
         },
         py::is_operator())
    .def("__mul__",
         [](const ncarray::NCArrayView& self, py::object other) {
           if (py::isinstance<py::array>(other)) {
             auto arr = py::reinterpret_borrow<py::array>(other);
             py::buffer_info info = arr.request();
             auto buf_ptr = info.ptr;
             ssize_t ptr_axis { -1 }; // -1 means NO pointer axis
             ncarray::NCArrayView view(&buf_ptr,
                                       arr.ndim(),
                                       arr.shape(),
                                       arr.strides(),
                                       pydtype_to_dtype(arr.dtype()),
                                       ptr_axis);
             return py::cast(self.mul(view));
           }
           throw py::type_error("Only array types accepted!");
         },
       py::is_operator())
    .def("__truediv__",
         [](const ncarray::NCArrayView& self, py::object other) {
           if (py::isinstance<py::array>(other)) {
             auto arr = py::reinterpret_borrow<py::array>(other);
             py::buffer_info info = arr.request();
             auto buf_ptr = info.ptr;
             ssize_t ptr_axis { -1 }; // -1 means NO pointer axis
             ncarray::NCArrayView view(&buf_ptr,
                                       arr.ndim(),
                                       arr.shape(),
                                       arr.strides(),
                                       pydtype_to_dtype(arr.dtype()),
                                       ptr_axis);
             return py::cast(self.truediv(view));
           }
           throw py::type_error("Only array types accepted!");
         },
       py::is_operator());

  /*
  py::classh<ncarray::NCArrayRef>(ncarray_module, "NCArrayRef")
    .def(py::init([](const py::array& data) {
      py::buffer_info arr_info = data.request();
      ssize_t ndim {data.ndim()};
      std::vector<ssize_t> shape(data.shape(), data.shape() + ndim);
      std::vector<ssize_t> strides(data.strides(), data.strides() + ndim);
      shape.insert(shape.begin(), 1);
      strides.insert(strides.begin(), 1);
      std::vector<void*> data_ptrs {arr_info.ptr};
      ncarray::DType dtype = pydtype_to_dtype(data.dtype());
      return new ncarray::NCArrayRef(data_ptrs, shape, strides, dtype);
    }))
    .def(py::init([](const py::list data_list) {
      ssize_t len_ptr_axis = data_list.size();
      std::vector<ssize_t> strides {1};
      std::vector<ssize_t> shape {len_ptr_axis};
      std::vector<void*> data_ptrs;
      // Assume each array in list is same shape for now
      py::dtype pydtype;
      bool assigned { false };
      for (py::handle o : data_list) {
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
    }))
    .def("__repr__", &ncarray::NCArrayRef::repr)
    .def_property_readonly("shape", [](const ncarray::NCArrayRef& self) -> py::tuple {
      auto* shape = self.shape();
      py::list l;
      for (ssize_t i = 0; i < self.ndim(); ++i) {
        l.append(shape[i]);
      }
      return l;
    })
    .def_property_readonly("strides", [](const ncarray::NCArrayRef& self) -> py::tuple {
      auto* strides = self.strides();
      py::list l;
      for (ssize_t i = 0; i < self.ndim(); ++i) {
        l.append(strides[i]);
      }
      return l;
    })
    .def_property_readonly("size", &ncarray::NCArrayRef::size)
    .def_property_readonly("ndim", &ncarray::NCArrayRef::ndim)
    .def_property_readonly("itemsize", &ncarray::NCArrayRef::itemsize)
    .def_property_readonly("nbytes", &ncarray::NCArrayRef::nbytes)
    .def("__getitem__", &ncarray::NCArrayRef::operator[], py::is_operator())
    .def("sum", &ncarray::NCArrayRef::sum)
    .def("max", &ncarray::NCArrayRef::max)
    .def("min", &ncarray::NCArrayRef::min)
    .def("mean", &ncarray::NCArrayRef::mean)
    .def("__add__", &ncarray::NCArrayRef::add, py::is_operator());

  py::classh<ncarray::NCArray>(ncarray_module, "NCArray")
    .def("__repr__", &ncarray::NCArray::repr)
    .def_property_readonly("shape", [](const ncarray::NCArray& self) -> py::tuple {
      auto* shape = self.shape();
      py::list l;
      for (ssize_t i = 0; i < self.ndim(); ++i) {
        l.append(shape[i]);
      }
      return l;
    })
    .def_property_readonly("strides", [](const ncarray::NCArray& self) -> py::tuple {
      auto* strides = self.strides();
      py::list l;
      for (ssize_t i = 0; i < self.ndim(); ++i) {
        l.append(strides[i]);
      }
      return l;
    })
    .def_property_readonly("size", &ncarray::NCArray::size)
    .def_property_readonly("ndim", &ncarray::NCArray::ndim)
    .def_property_readonly("itemsize", &ncarray::NCArray::itemsize)
    .def_property_readonly("nbytes", &ncarray::NCArray::nbytes)
    .def("__getitem__", &ncarray::NCArray::operator[], py::is_operator())
    .def("sum", &ncarray::NCArray::sum)
    .def("max", &ncarray::NCArray::max)
    .def("min", &ncarray::NCArray::min)
    .def("mean", &ncarray::NCArray::mean)
    .def("__add__", &ncarray::NCArray::add, py::is_operator());
  */
} // ncarray_module
