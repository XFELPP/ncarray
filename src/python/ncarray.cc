#include "ncarrayref.hh"
#include "ncarrayview.hh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(ncarray, ncarray_module, py::mod_gil_not_used()) {
  ncarray_module.doc() = "Non-contiguous (NC) Array Classes.";

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
    .def("__getitem__", &ncarray::NCArrayView::operator[], py::is_operator())
    .def("sum", &ncarray::NCArrayView::sum)
    .def("max", &ncarray::NCArrayView::max)
    .def("min", &ncarray::NCArrayView::min)
    .def("mean", &ncarray::NCArrayView::mean)
    .def("__add__", &ncarray::NCArrayView::add, py::is_operator())
    .def("__mul__", &ncarray::NCArrayView::mul, py::is_operator())
    .def("__truediv__", &ncarray::NCArrayView::truediv, py::is_operator());

  py::classh<ncarray::NCArrayRef>(ncarray_module, "NCArrayRef")
    .def(py::init([](const py::array& data) {
      py::buffer_info arr_info = data.request();
      ssize_t ndim {data.ndim()};
      std::vector<ssize_t> shape(data.shape(), data.shape() + ndim);
      std::vector<ssize_t> strides(data.strides(), data.strides() + ndim);
      shape.insert(shape.begin(), 1);
      strides.insert(strides.begin(), 1);
      std::vector<void*> data_ptrs {arr_info.ptr};
      return new ncarray::NCArrayRef(data_ptrs, shape, strides, data.dtype());
    }))
    .def(py::init([](const py::list data_list) {
      ssize_t len_ptr_axis = data_list.size();
      std::vector<ssize_t> strides {1};
      std::vector<ssize_t> shape {len_ptr_axis};
      std::vector<void*> data_ptrs;
      // Assume each array in list is same shape for now
      py::dtype dtype;
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
        dtype = data.dtype();
      }
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
} // ncarray_module
