/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCA_PYTHON_UTILITIES_HH
#define NCA_PYTHON_UTILITIES_HH

#include "ncarray/custom_types.hh"
#include "ncarray/ncarrays.hh"
#include "ncarray/soarrays.hh"
#ifdef NCA_HAS_CUDA
// This code doesn't actually have anything GPU-specific (__device__ or kernels)
// They are the specializations for GPU arrays
#include "ncarray/ncdevarrays.cuh"
#include "ncarray/sodevarrays.cuh"
#endif

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

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace pyncarray {
  /**
   * This is a duplicate dispatch to that in ncarry/array_impl.hh but host only.
   * The original needs to be host/device, so we get a lot of spurious warnings.
   *
   * We do NOT want to just silence the warning (e.g. pragma diag_suppress 20014)
   * for nvcc as we want to catch other instances of host/device called from host.
   */
  template <typename Visitor>
  inline auto host_dispatch(ncarray::DType type, Visitor&& visitor) {
    switch (type) {
    case ncarray::DType::bool_: {
      return visitor.template operator()<bool>();
    }
    case ncarray::DType::char_: {
      return visitor.template operator()<char>();
    }
    case ncarray::DType::uint8: {
      return visitor.template operator()<std::uint8_t>();
    }
    case ncarray::DType::uint16: {
      return visitor.template operator()<std::uint16_t>();
    }
    case ncarray::DType::uint32: {
      return visitor.template operator()<std::uint32_t>();
    }
    case ncarray::DType::uint64: {
      return visitor.template operator()<std::uint64_t>();
    }
    case ncarray::DType::int8: {
      return visitor.template operator()<std::int8_t>();
    }
    case ncarray::DType::int16: {
      return visitor.template operator()<std::int16_t>();
    }
    case ncarray::DType::int32: {
      return visitor.template operator()<std::int32_t>();
    }
    case ncarray::DType::int64: {
      return visitor.template operator()<std::int64_t>();
    }
    case ncarray::DType::float32: {
      return visitor.template operator()<float>();
    }
    case ncarray::DType::float64: {
      return visitor.template operator()<double>();
    }
    case ncarray::DType::float128: {
      return visitor.template operator()<long double>();
    }
    case ncarray::DType::complex64: {
      return visitor.template operator()<std::complex<float>>();
    }
    case ncarray::DType::complex128: {
      return visitor.template operator()<std::complex<double>>();
    }
    case ncarray::DType::complex256: {
      return visitor.template operator()<std::complex<long double>>();
    }
    case ncarray::DType::vfloat2: {
      return visitor.template operator()<ncarray::Float2>();
    }
    case ncarray::DType::vfloat3: {
      return visitor.template operator()<ncarray::Float3>();
    }
    case ncarray::DType::vfloat4: {
      return visitor.template operator()<ncarray::Float4>();
    }
    case ncarray::DType::vdouble2: {
      return visitor.template operator()<ncarray::Double2>();
    }
    case ncarray::DType::vdouble3: {
      return visitor.template operator()<ncarray::Double3>();
    }
    case ncarray::DType::vdouble4: {
      return visitor.template operator()<ncarray::Double4>();
    }
    }

    throw std::runtime_error("Tried to dispatch unsupported type!");
  }

  /**
   * Cast a pybind11 array dtype to a ncarray::DType.
   *
   * @param[in] dtype The pybind11 datatype.
   * @returns The ncarray datatype.
   */
  inline ncarray::DType pydtype_to_dtype(py::dtype dtype) {
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
    } else if (dtype.is(py::dtype::of<char>())) {
      return ncarray::dtype_traits<char>::value;
    } else if (dtype.is(py::dtype::of<float>())) {
      return ncarray::dtype_traits<float>::value;
    } else if (dtype.is(py::dtype::of<double>())) {
      return ncarray::dtype_traits<double>::value;
    } else if (dtype.is(py::dtype::of<long double>())) {
      return ncarray::dtype_traits<long double>::value;
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
   * @param[in] target_ndim The target dimensionality of the output with padding.
   *            TODO: This mostly serves as a hack until broadcasting is suppported.
   *            Because of the implicit dimensionality mismatch between NCArrays* and NumPy....
   * @returns A view of ViewType over the data from the NumPy array.
   */
  template <class ViewType>
  inline auto pyarray_to_view(const py::array& arr, ssize_t target_ndim = -1) {
    py::buffer_info info = arr.request();
    auto buf_ptr = info.ptr;
    bool read_only = !arr.writeable();
    // This is temporary!! Careful how you use it!
    ssize_t ptr_axis { -1 }; // -1 means NO pointer axis

    ssize_t ndim { arr.ndim() };
    if (target_ndim > ndim) {
      ndim = target_ndim;
    }

    // NOTE: Because of the implicit extra dimension when constructing NCArrays
    //       there is a mismatch vs NumPy. True broadcasting is not supported yet,
    //       but this padding up front makes NumPy match the NCArray for performing ops.
    // TODO: Revisit and fix this kludge when true broadcasting is added!
    ssize_t padded_shape[NCARRAY_MAX_NDIM];
    ssize_t padded_strides[NCARRAY_MAX_NDIM];

    ssize_t offset { ndim - arr.ndim() };
    for (ssize_t i = 0; i < ndim; ++i) {
      if (i < offset) {
        padded_shape[i] = 1;
        padded_strides[i] = 0;
      } else {
        padded_shape[i] = arr.shape()[i - offset];
        padded_strides[i] = arr.strides()[i - offset];
      }
    }
    return ViewType(buf_ptr,
                    ndim,
                    padded_shape,
                    padded_strides,
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
   * @tparam RefT The reference storage specifier. E.g. RefPolicy or DevRefPolicy.
   * @param[in] arr The input array of arrays.
   * @param[in] read_only_ Whether this reference should be read only.
   * @returns The constructed reference type.
   */
  template <class LayoutT, class RefT = ncarray::RefPolicy>
  inline ncarray::ArrayImpl<LayoutT, RefT>
  pyarray_to_ref(const py::array& arr, const bool read_only_ = false) {
    py::buffer_info arr_info = arr.request();
    ssize_t ndim { arr.ndim() };
    std::vector<ssize_t> shape(arr.shape(), arr.shape() + ndim);
    std::vector<ssize_t> strides(arr.strides(), arr.strides() + ndim);

    shape.insert(shape.begin(), 1);
    if constexpr (std::is_same_v<LayoutT, ncarray::SOArrayPolicy>) {
      strides.insert(strides.begin(), sizeof(void*));
    } else {
      strides.insert(strides.begin(), 1);
    }

    std::vector<void*> data_ptrs { arr_info.ptr };
    ncarray::DType dtype = pydtype_to_dtype(arr.dtype());
    bool read_only { read_only_ };
    if (!arr.writeable()) {
      read_only = true;
    }
    // A single array will have no pointer axis - but we created one artificially
    ssize_t ptr_axis { 0 };
    return ncarray::ArrayImpl<LayoutT, RefT>(data_ptrs,
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
   * @tparam RefT The reference storage specifier. E.g. RefPolicy or DevRefPolicy.
   * @param[in] list The input list of arrays.
   * @param[in] read_only_ Whether this reference should be read only.
   * @returns The constructed reference type.
   */
  template <class LayoutT, class RefT = ncarray::RefPolicy>
  inline ncarray::ArrayImpl<LayoutT, RefT>
  pylist_to_ref(const py::list& list, const bool read_only_ = false) {
    ssize_t len_ptr_axis = list.size();
    std::vector<ssize_t> strides;
    if constexpr (std::is_same_v<LayoutT, ncarray::SOArrayPolicy>) {
      strides.push_back(sizeof(void*));
    } else {
      strides.push_back(1);
    }
    std::vector<ssize_t> shape { len_ptr_axis };
    std::vector<void*> data_ptrs;

    // Assume each array in list is same shape for now
    py::dtype pydtype;
    bool assigned { false };
    bool read_only { read_only_ };
    for (py::handle o : list) {
      py::array data = py::reinterpret_borrow<py::array>(o);
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
    return ncarray::ArrayImpl<LayoutT, RefT>(data_ptrs,
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
  inline py::array ncarr_to_numpy(const ArrayT& ncarr,
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

    return host_dispatch(out_dtype, dispatched_copy);
  }

  inline ncarray::Slice pyslice_to_slice(ssize_t axis_shape, py::slice slice) {
    ssize_t start, stop, step, length;
    if (!slice.compute(axis_shape, &start, &stop, &step, &length)) {
      throw py::error_already_set();
    }
    return ncarray::Slice(start, stop, step);
  }

  template <class ArrayT>
  inline std::vector<ncarray::IndexItem> pytuple_to_indices(const ArrayT& self,
                                                            py::tuple tup) {
    std::vector<ncarray::IndexItem> indices;

    if (static_cast<ssize_t>(tup.size()) > self.ndim()) {
      throw py::index_error("Too many indices for array!");
    }
    bool have_ellipsis { false };
    ssize_t axis = 0;
    for (ssize_t i = 0; i < static_cast<ssize_t>(tup.size()); ++i) {
      auto pyidx = tup[i];
      if (py::isinstance<py::ellipsis>(pyidx)) {
        if (have_ellipsis) {
          throw py::index_error("Index arguments can only contain one ellipsis!");
        }
        have_ellipsis = true;
        indices.push_back(ncarray::IndexItem(ncarray::Ellipsis {}));
        axis += self.ndim() - tup.size() + 1;
      } else {
        if (py::isinstance<py::slice>(pyidx)) {
          indices.push_back(ncarray::IndexItem(pyslice_to_slice(self.shape(axis),
                                                                pyidx.cast<py::slice>())));
        } else if (py::isinstance<py::int_>(pyidx)) {
          auto idx_val { pyidx.cast<ssize_t>() };
          auto ax_shape { self.shape(axis) };
          if (idx_val < -ax_shape || idx_val >= ax_shape) {
            throw py::index_error("Index out of bounds!");
          }
          indices.push_back(ncarray::IndexItem(pyidx.cast<ssize_t>()));
        } else {
          throw py::index_error("Unsupported index type!");
        }
        axis++;
      }
    }

    return indices;
  }
} // namespace pyncarray

#endif // NCA_PYTHON_UTILITIES_HH
