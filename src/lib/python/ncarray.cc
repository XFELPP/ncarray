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
    bool read_only = !arr.writeable();
    // This is temporary!! Careful how you use it!
    ssize_t ptr_axis { -1 }; // -1 means NO pointer axis
    // When you have no pointer axis, just pass the raw pointer
    return ncarray::NCArrayView(reinterpret_cast<void**>(buf_ptr),
                                arr.ndim(),
                                arr.shape(),
                                arr.strides(),
                                pydtype_to_dtype(arr.dtype()),
                                ptr_axis,
                                read_only);
  }

  ncarray::NCArrayRef pyarray_to_ref(const py::array& arr, const bool read_only_ = false) {
    py::buffer_info arr_info = arr.request();
    ssize_t ndim { arr.ndim() };
    std::vector<ssize_t> shape(arr.shape(), arr.shape() + ndim);
    std::vector<ssize_t> strides(arr.strides(), arr.strides() + ndim);

    shape.insert(shape.begin(), 1);
    strides.insert(strides.begin(), 1);

    std::vector<void*> data_ptrs { arr_info.ptr };
    ncarray::DType dtype = pydtype_to_dtype(arr.dtype());
    bool read_only { read_only_ };
    if (!arr.writeable()) {
      read_only = true;
    }
    // A single array will have no pointer axis
    ssize_t ptr_axis { -1 };
    return ncarray::NCArrayRef(data_ptrs, shape, strides, dtype, ptr_axis, read_only);
  }

  ncarray::NCArrayRef pylist_to_ref(const py::list& list, const bool read_only_ = false) {
    ssize_t len_ptr_axis = list.size();
    std::vector<ssize_t> strides { 1 };
    std::vector<ssize_t> shape { len_ptr_axis };
    std::vector<void*> data_ptrs;

    // Assume each array in list is same shape for now
    py::dtype pydtype;
    bool assigned { false };
    bool read_only { read_only_ };
    for (py::handle o : list) {
      py::array data = py::cast<py::array>(o);
      py::buffer_info arr_info = data.request();
      data_ptrs.push_back(arr_info.ptr);
      if (!data.writeable()) {
        // If any array from the list is read_only, all of them in the NCArrayRef
        // will therefore be marked read only
        read_only = true;
      }
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
    ssize_t ptr_axis { 0 };
    return ncarray::NCArrayRef(data_ptrs, shape, strides, dtype, ptr_axis, read_only);
  }

  py::array ncarr_to_numpy(const ncarray::NCArrayView& ncarr,
                           std::optional<ncarray::DType> dtype = std::nullopt) {
    ncarray::DType out_dtype = dtype.value_or(ncarr.dtype());

    auto dispatched_copy = [&] <typename OutT> () {
      // TODO: Consider adding a vector accessor on NCArrayView (like ncarr.shape_vec)
      std::vector<ssize_t> shape_vec(ncarr.shape(), ncarr.shape() + ncarr.ndim());
      py::array_t<OutT> out_arr(shape_vec);

      auto* out_ptr  = reinterpret_cast<OutT*>(out_arr.mutable_data());
      ncarr.copy_into_astype<OutT>(out_ptr);

      return py::array(out_arr);
    };

    return ncarray::dispatch(out_dtype, dispatched_copy);
  }

  std::variant<ssize_t, ncarray::Slice,ncarray::ArrayIndices>
  pyindices_to_indices(const py::object& pyindices, const ssize_t* shape) {
    if (py::isinstance<py::int_>(pyindices)) {
      return pyindices.cast<ssize_t>();
    } else if (py::isinstance<py::slice>(pyindices)) {
      return pyslice_to_slice(shape[0], pyindices.cast<py::slice>());
    } else if (py::isinstance<py::tuple>(pyindices)) {
      ncarray::ArrayIndices indices;
      ssize_t axis { 0 };
      for (auto& item : pyindices.cast<py::tuple>()) {
        if (py::isinstance<py::int_>(item)) {
          indices.emplace_back(item.cast<ssize_t>());
        } else if (py::isinstance<py::slice>(item)) {
          indices.emplace_back(pyslice_to_slice(shape[axis],
                                                item.cast<py::slice>()));
        } else if (item.is(py::ellipsis())) {
          indices.emplace_back(ncarray::Ellipsis{});
        } else {
          throw py::type_error("Invalid indexing argument!");
        }
      }
      return indices;
    }
    throw py::type_error("Invalid indexing argument!");
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
    .export_values()
    .finalize();

// TODO: It would be nice to not need the view conversion when dealing with array
// Arrays don't currently satisfy the concepts though due to py::dtype.
// Minor thing to improve
#define REGISTER_OPERATION(PYMETHOD, OP)                                                               \
    .def("__" PYMETHOD "__", [](const ncarray::NCArrayView& self, const ncarray::NCArrayView& other) { \
      return py::cast(self OP other);                                                                  \
    },                                                                                                 \
      py::is_operator())                                                                               \
    .def("__" PYMETHOD "__", [](const ncarray::NCArrayView& self, const py::array& other) {            \
      return py::cast(self OP pyarray_to_view(other));                                                 \
    },                                                                                                 \
      py::is_operator())                                                                               \
    .def("__r" PYMETHOD "__", [](const ncarray::NCArrayView& self, const py::array& other) {           \
      return py::cast(pyarray_to_view(other) OP self);                                                 \
    },                                                                                                 \
      py::is_operator())

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
    // --- Standard Container Methods --- //
    .def("__len__", [](const ncarray::NCArrayView& self) {
      if (self.ndim() > 0) {
        return self.shape(0);
      }
      return ssize_t(0);
    })
    .def("__iter__", [](const ncarray::NCArrayView& self) {
        return py::make_iterator(self.begin(), self.end());
    })
    // --- Array-Like Methods (indexing, size, shape, dtype, etc) --- //
    .def_property_readonly("size",
                           &ncarray::NCArrayView::size,
                           "The number of items in the NCArray*.")
    .def_property_readonly("ndim",
                           &ncarray::NCArrayView::ndim,
                           "The number of dimensions in the NCArray*.")
    .def_property_readonly("itemsize",
                           &ncarray::NCArrayView::itemsize,
                           "The size in bytes of a single item in the NCArray*.")
    .def_property_readonly("nbytes",
                           &ncarray::NCArrayView::nbytes,
                           "The total size in bytes of all items in the NCArray*.")
    .def("squeeze",
         &ncarray::NCArrayView::squeeze,
         "Collapse and remove all axes of length 1.")
    .def("astype",
         [](const ncarray::NCArrayView& self, ncarray::DType& dtype_out) {
           return self.astype(dtype_out);
         },
         py::arg("dtype"),
         "Convert an NCArray* to the specified data type.")
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
    .def("__setitem__",
         [](const ncarray::NCArrayView& self, py::object idx, py::object val) {
           auto indices = pyindices_to_indices(idx, self.shape());
           ncarray::ViewOrScalar sub_view_or_scalar;
           if (std::holds_alternative<ssize_t>(indices)) {
             sub_view_or_scalar = self[std::get<ssize_t>(indices)];
           } else if (std::holds_alternative<ncarray::Slice>(indices)) {
             sub_view_or_scalar = self[std::get<ncarray::Slice>(indices)];
           } else {
             sub_view_or_scalar = self[std::get<ncarray::ArrayIndices>(indices)];
           }

           auto& view = std::get<ncarray::NCArrayView>(sub_view_or_scalar);
           if (py::isinstance<py::array>(val)) {
             // ... //
           } else if (py::isinstance<ncarray::NCArrayView>(val)) {
             view.assign(val.cast<ncarray::NCArrayView&>());
           } else {
             // Convertible to scalar
             // Use the algorithm directly to avoid the variant gets
             view.fill(val.cast<ncarray::Scalar>());
           }
         },
         py::is_operator())
    // --- Array Reduction Methods (Reduce to scalar) --- //
    .def("sum", &ncarray::NCArrayView::sum)
    .def("max", &ncarray::NCArrayView::max)
    .def("min", &ncarray::NCArrayView::min)
    .def("mean", &ncarray::NCArrayView::mean)
    // --- Binary Array Methods --- //
    REGISTER_OPERATION("add", +)
    REGISTER_OPERATION("mul", *)
    REGISTER_OPERATION("truediv", /)
    // NumPy protocol compatibility
    // __array__(self, dtype=None, copy=None)
    .def("__array__", [](const ncarray::NCArrayView& self,
                         py::object& dtype,
                         py::object& copy) {
      return ncarr_to_numpy(self);
    })
    // __array_priority__ attribute - set high so NCArray* funcs used, and is returned
    .def_property_readonly_static("__array_priority__", [](const py::object&) {
      return 100.0;
    })
    .def("__array_ufunc__", [](const ncarray::NCArrayView& self,
                               py::handle ufunc,
                               py::str method,
                               py::args args,
                               py::kwargs kwargs) {
      if (method.cast<std::string>() != "__call__") {
        return py::none().cast<py::object>();
      }

      // For now, just convert to NumPy
      // TODO: Optimize this with NCArray* directly
      py::list new_args;
      for (auto& arg : args) {
        if (py::isinstance<ncarray::NCArrayView>(arg)) {
          new_args.append(ncarr_to_numpy(arg.cast<ncarray::NCArrayView>()));
        } else {
          new_args.append(arg);
        }
      }
      return ufunc(*new_args, **kwargs);
    });
#undef REGISTER_OPERATION

  py::classh<ncarray::NCArrayRef, ncarray::NCArrayView>(ncarray_module, "NCArrayRef")
    .def(py::init([](const py::array& arr, const bool read_only = false) {
      return pyarray_to_ref(arr, read_only);
    }),
      py::arg("data"),
      py::arg("read_only") = py::cast(false))
    .def(py::init([](const py::list& list, const bool read_only = false) {
      return pylist_to_ref(list, read_only);
    }),
      py::arg("data"),
      py::arg("read_only") = py::cast(false));

  py::classh<ncarray::NCArray, ncarray::NCArrayView>(ncarray_module, "NCArray")
    .def(py::init([](const std::vector<ssize_t>& shape, const ncarray::DType& dtype) {
      return new ncarray::NCArray(shape, dtype);
    }),
      py::arg("shape"),
      py::arg("dtype") = py::cast(ncarray::DType::float32));
} // ncarray_module
