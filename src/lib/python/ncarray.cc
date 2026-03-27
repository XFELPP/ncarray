/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "cartesian.hh" // Cartesian products/indexing logic for operator[]

#include "ncarray/ncarrays.hh"
#include "ncarray/soarrays.hh"

#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <sys/types.h>
#include <vector>

namespace py = pybind11;

namespace {
  /**
   * Cast a pybind11 array dtype to a ncarray::DType.
   *
   * @param[in] dtype The pybind11 datatype.
   * @returns The ncarray datatype.
   */
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

  /**
   * Construct a view over a NumPy array.
   *
   * @tparam ViewType The kind of view to construct. E.g. NCArrayView, SOArrayView.
   * @param[in] arr An input NumPy array to convert.
   * @returns A view of ViewType over the data from the NumPy array.
   */
  template <class ViewType>
  auto pyarray_to_view(const py::array& arr) {
    py::buffer_info info = arr.request();
    auto buf_ptr = info.ptr;
    bool read_only = !arr.writeable();
    // This is temporary!! Careful how you use it!
    ssize_t ptr_axis { -1 }; // -1 means NO pointer axis
    // When you have no pointer axis, just pass the raw pointer
    return ViewType(buf_ptr,
                    arr.ndim(),
                    arr.shape(),
                    arr.strides(),
                    pydtype_to_dtype(arr.dtype()),
                    ptr_axis,
                    read_only);
  }

  /**
   * Construct a reference type array from an array of arrays.
   *
   * NOTE: This function assumes the arrays in the array all have the same shape
   * and strides.
   *
   * @tparam LayoutT The layout specifier for the returned reference type.
   *         E.g., NCOffsetsPolicy or SOArrayPolicy (PEP3118).
   * @param[in] arr The input array of arrays.
   * @param[in] read_only_ Whether this reference should be read only.
   * @returns The constructed reference type.
   */
  template <class LayoutT>
  ncarray::ArrayImpl<LayoutT, ncarray::RefPolicy>
  pyarray_to_ref(const py::array& arr, const bool read_only_ = false) {
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
    return ncarray::ArrayImpl<LayoutT, ncarray::RefPolicy>(data_ptrs,
                                                           shape,
                                                           strides,
                                                           dtype,
                                                           ptr_axis,
                                                           read_only);
  }

  /**
   * Construct a reference type array from a list of arrays.
   *
   * NOTE: This function assumes the arrays in the list all have the same shape
   * and strides.
   *
   * @tparam LayoutT The layout specifier for the returned reference type.
   *         E.g., NCOffsetsPolicy or SOArrayPolicy (PEP3118).
   * @param[in] list The input list of arrays.
   * @param[in] read_only_ Whether this reference should be read only.
   * @returns The constructed reference type.
   */
  template <class LayoutT>
  ncarray::ArrayImpl<LayoutT, ncarray::RefPolicy>
  pylist_to_ref(const py::list& list, const bool read_only_ = false) {
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
        // If any array from the list is read_only, all of them in the *ArrayRef
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
    return ncarray::ArrayImpl<LayoutT, ncarray::RefPolicy>(data_ptrs,
                                                           shape,
                                                           strides,
                                                           dtype,
                                                           ptr_axis,
                                                           read_only);
  }

  /**
   * Construct a NumPy array from an input nc/soarray type object.
   *
   * @tparam ArrayT The kind of array being used. This is a full array specifier,
   *         including storage+layout specifier.
   * @param[in] ncarr The input array to convert.
   * @param[in] dtype The datatype for the converted NumPy array. I.e., if provided
   *          a cast will be performed. Otherwise, the input array dtype will be used.
   * @returns A NumPy array of specified dtype over the input data.
   */
  template <class ArrayT>
  py::array ncarr_to_numpy(const ArrayT& ncarr,
                           std::optional<ncarray::DType> dtype = std::nullopt) {
    ncarray::DType out_dtype = dtype.value_or(ncarr.dtype());

    auto dispatched_copy = [&] <typename OutT> () {
      // TODO: Consider adding a vector accessor on NCArrayView (like ncarr.shape_vec)
      std::vector<ssize_t> shape_vec(ncarr.shape(), ncarr.shape() + ncarr.ndim());
      py::array_t<OutT> out_arr(shape_vec);

      auto* out_ptr  = reinterpret_cast<OutT*>(out_arr.mutable_data());
      ncarr.template copy_into_astype<OutT>(out_ptr);

      return py::array(out_arr);
    };

    return ncarray::dispatch(out_dtype, dispatched_copy);
  }

  /**
   * @def REGISTER_OPERATION(PYMETHOD, OP)
   * @brief A helper to attach a dunder to a class binding for operator overloads.
   * @example REGISTER_OPERATION("add", +) binds operator+(...) to __add__
   * @todo We currently need to convert arrays to view, because the C++ lib cannot
   *       take the array directly.
   */
#define REGISTER_OPERATION(PYMETHOD, OP)                                            \
    .def("__" PYMETHOD "__", [](const ArrayT& self, const ArrayT& other) {          \
      return py::cast(self OP other);                                               \
    },                                                                              \
      py::is_operator())                                                            \
    .def("__" PYMETHOD "__", [](const ArrayT& self, const py::array& other) {       \
      return py::cast(self OP pyarray_to_view<typename ArrayT::ViewType>(other));   \
    },                                                                              \
      py::is_operator())                                                            \
    .def("__r" PYMETHOD "__", [](const ArrayT& self, const py::array& other) {      \
      return py::cast(pyarray_to_view<typename ArrayT::ViewType>(other) OP self);   \
    },                                                                              \
      py::is_operator())                                                            \
    .def("__" PYMETHOD "__", [](const ArrayT& self, const ncarray::Scalar& other) { \
      return py::cast(self OP other);                                               \
    },                                                                              \
      py::is_operator())

  /**
   * Helper function to attach common methods to a Python binding for an array
   * specialization.
   *
   * All array specializations generally have the same functions in C++ and in
   * their Python bindings (plus/minus some speciality features). This function
   * just attaches those all.
   *
   * @tparam ArrayT The kind of array being used. This is a full array specifier,
   *         including storage+layout specifier.
   * @param[in] arr_cl The class of the Python binding.
   */
  template <typename ArrayT>
  void register_common_array_methods(py::classh<ArrayT>& arr_cl) {
    using ViewType = typename ArrayT::ViewType;

    using ViewOrScalar = std::variant<ViewType, ncarray::Scalar>;

    arr_cl.def("__repr__", &ArrayT::repr)
    .def_property_readonly("shape", [](const ArrayT& self) -> py::tuple {
      auto* shape = self.shape();
      py::list l;
      for (ssize_t i = 0; i < self.ndim(); ++i) {
        l.append(shape[i]);
      }
      return l;
    })
    .def_property_readonly("strides", [](const ArrayT& self) -> py::tuple {
      auto* strides = self.strides();
      py::list l;
      for (ssize_t i = 0; i < self.ndim(); ++i) {
        l.append(strides[i]);
      }
      return l;
    })
    // --- Standard Container Methods --- //
    .def("__len__", [](const ArrayT& self) {
      if (self.ndim() > 0) {
        return self.shape(0);
      }
      return ssize_t(0);
    })
    .def("__iter__", [](const ArrayT& self) {
        return py::make_iterator(self.begin(), self.end());
    })
    // --- Array-Like Methods (indexing, size, shape, dtype, etc) --- //
    .def_property_readonly("size",
                           &ArrayT::size,
                           "The number of items in the NCArray*.")
    .def_property_readonly("ndim",
                           &ArrayT::ndim,
                           "The number of dimensions in the NCArray*.")
    .def_property_readonly("itemsize",
                           &ArrayT::itemsize,
                           "The size in bytes of a single item in the NCArray*.")
    .def_property_readonly("nbytes",
                           &ArrayT::nbytes,
                           "The total size in bytes of all items in the NCArray*.")
    .def("squeeze",
         &ArrayT::squeeze,
         "Collapse and remove all axes of length 1.")
    .def("astype",
         [](const ArrayT& self, ncarray::DType& dtype_out) {
           return self.astype(dtype_out);
         },
         py::arg("dtype"),
         "Convert an NCArray* to the specified data type.")
    .def("__getitem__",
         [](const ArrayT& self, py::object idx) -> ViewOrScalar {
           // NOTE: The Python bindings diverge from the C++ library on scalars.
           //       For simplicity, in Python, scalars are returned as scalars.
           //       In C++, they remain as an object tied to the array class.
           ViewType view;
           if (py::isinstance<py::int_>(idx)) {
             auto idx_val = idx.cast<ssize_t>();
             if (idx_val < -self.shape(0) || idx_val >= self.shape(0)) {
               throw py::index_error("Index out of bounds!");
             }
             view = self[idx_val];
           } else if (py::isinstance<py::slice>(idx)) {
             view = self[pyncarray::pyslice_to_slice(self.shape(0),
                                                     idx.cast<py::slice>())];
           } else if (py::isinstance<py::ellipsis>(idx)) {
             view = self[ncarray::Ellipsis{}];
           } else if (py::isinstance<py::tuple>(idx)) {
             view = pyncarray::dispatch_cartesian(idx.cast<py::tuple>(),
                                                  self,
                                                  pyncarray::All{});
           } else {
             throw py::type_error("Invalid indexing argument!");
           }
           // For convenience convert scalars to... scalars
           if (view.ndim() == 0) {
             return view.template get_scalar(view.data());
           }

           return view;
         },
         py::is_operator())
    .def("__setitem__",
         [](const ArrayT& self, py::object idx, py::object val) {
           ViewType view;
           if (py::isinstance<py::int_>(idx)) {
             auto idx_val = idx.cast<ssize_t>();
             if (idx_val < -self.shape(0) || idx_val >= self.shape(0)) {
               throw py::index_error("Index out of bounds!");
             }
             view = self[idx_val];
           } else if (py::isinstance<py::slice>(idx)) {
             view = self[pyncarray::pyslice_to_slice(self.shape(0),
                                                     idx.cast<py::slice>())];
           } else if (py::isinstance<py::ellipsis>(idx)) {
             view = self[ncarray::Ellipsis{}];
           } else if (py::isinstance<py::tuple>(idx)) {
             view =
               pyncarray::dispatch_cartesian(idx.cast<py::tuple>(),
                                             self,
                                             pyncarray::All{});
           } else {
             throw py::type_error("Invalid indexing argument!");
           }

           if (py::isinstance<py::array>(val)) {
             // ... //
           } else if (py::isinstance<ArrayT>(val)) {
             view.assign(val.cast<ArrayT&>());
           } else {
             // Convertible to scalar
             // Use the algorithm directly to avoid the variant gets
             view.fill(val.cast<ncarray::Scalar>());
           }
         },
         py::is_operator())
    // --- Array Reduction Methods (Reduce to scalar) --- //
    .def("sum", &ArrayT::sum)
    .def("max", &ArrayT::max)
    .def("min", &ArrayT::min)
    .def("mean", &ArrayT::mean)
    // --- Binary Array Methods --- //
    REGISTER_OPERATION("add", +)
    REGISTER_OPERATION("sub", -)
    REGISTER_OPERATION("mul", *)
    REGISTER_OPERATION("truediv", /)
    // NumPy protocol compatibility
    // __array__(self, dtype=None, copy=None)
    .def("__array__", [](const ArrayT& self,
                         py::object& dtype,
                         py::object& copy) {
      return ncarr_to_numpy(self);
    })
    // __array_priority__ attribute - set high so NCArray* funcs used, and is returned
    .def_property_readonly_static("__array_priority__", [](const py::object&) {
      return 100.0;
    })
    .def("__array_ufunc__", [](const ArrayT& self,
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
      for (const auto& arg : args) {
        if (py::isinstance<ArrayT>(arg)) {
          new_args.append(ncarr_to_numpy(arg.cast<ArrayT>()));
        } else {
          new_args.append(arg);
        }
      }
      return ufunc(*new_args, **kwargs);
    });
#undef REGISTER_OPERATION
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
