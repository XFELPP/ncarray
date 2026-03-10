#include "ncarrayview.hh"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <type_traits>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace ncarray {
  NCArrayView::NCArrayView(void** data_,
                           std::vector<ssize_t>& shape_,
                           std::vector<ssize_t>& strides_,
                           py::dtype dtype_)
      : m_data(data_)
      , m_shape(shape_)
      , m_strides(strides_)
      , m_offsets(m_strides.size())
      , m_dtype(dtype_)
  {}

  NCArrayView::NCArrayView(void** data_,
                           std::vector<ssize_t>& shape_,
                           std::vector<ssize_t>& strides_,
                           std::vector<ssize_t>& offsets_,
                           py::dtype dtype_)
      : m_data(data_)
      , m_shape(shape_)
      , m_strides(strides_)
      , m_offsets(offsets_)
      , m_dtype(dtype_)
  {}

  py::object NCArrayView::sum() const {
    auto sum_operation = [&] <typename T>() {
      using AccumT = typename op_traits<T>::sum_type;
      auto sum_op_internal = [](const std::uint8_t* data, AccumT* output) {
        *output += static_cast<AccumT>(*reinterpret_cast<const T*>(data));
      };

      return op_impl<T>(sum_op_internal, AccumT {0});
    };

    return dispatch(sum_operation);
  }

  py::object NCArrayView::max() const {
    auto max_operation = [&]<typename T>() {
      // Don't need a broader type for this one
      auto max_op_internal = [](const std::uint8_t* data, T* output) {
        T val = *reinterpret_cast<const T*>(data);
        if (val > *output) {
          *output = val;
        }
      };

      return op_impl<T>(max_op_internal, std::numeric_limits<T>::lowest());
    };

    return dispatch(max_operation);
  }

  py::object NCArrayView::min() const {
    auto min_operation = [&]<typename T>() {
      // Don't need a broader type for this one
      auto min_op_internal = [](const std::uint8_t* data, T* output) {
        T val = *reinterpret_cast<const T*>(data);
        if (val < *output) {
          *output = val;
        }
      };

      return op_impl<T>(min_op_internal, std::numeric_limits<T>::max());
    };

    return dispatch(min_operation);
  }

  py::object NCArrayView::mean() const {
    // Not sure if we should do by type stuff for the mean here?
    double total = sum().cast<double>();
    return py::cast(total / size());
  }

  py::array NCArrayView::add(const py::object& other) const {
    std::vector<ssize_t> other_strides(ndim());
    std::vector<ssize_t> other_offsets(ndim());
    void* other_data { nullptr };

    bool other_is_arr { false };
    if (py::isinstance<py::array>(other)) {
      auto arr = other.cast<py::array>();
      other_strides = std::vector<ssize_t>(arr.strides(), arr.strides() + ndim());
      other_data = const_cast<void*>(arr.data());
      other_is_arr = true;
    } else if (py::isinstance<NCArrayView>(other)) {
      auto view = other.cast<NCArrayView>();
      other_strides = std::vector<ssize_t>(view.strides(), view.strides() + ndim());
      other_offsets = std::vector<ssize_t>(view.offsets(), view.offsets() + ndim());
      other_data = const_cast<void*>(view.data());
    } else {
      throw py::value_error("Incompatible type - must be array or ncarray type.");
    }

    auto add_operation = [&]<typename T>() {
      auto add_op_internal = [](const std::uint8_t* lhs, const std::uint8_t* rhs, T* output) {
        *output = *reinterpret_cast<const T*>(lhs) + *reinterpret_cast<const T*>(rhs);
      };

      return binary_op_impl<T, decltype(add_op_internal), T>(add_op_internal,
                                                             other_is_arr,
                                                             other_data,
                                                             other_strides.data(),
                                                             other_offsets.data());
    };

    return dispatch(add_operation);
  }

  py::array NCArrayView::mul(const py::object& other) const {
    std::vector<ssize_t> other_strides(ndim());
    std::vector<ssize_t> other_offsets(ndim());
    void* other_data { nullptr };

    bool other_is_arr { false };
    if (py::isinstance<py::array>(other)) {
      auto arr = other.cast<py::array>();
      other_strides = std::vector<ssize_t>(arr.strides(), arr.strides() + ndim());
      other_data = const_cast<void*>(arr.data());
      other_is_arr = true;
    } else if (py::isinstance<NCArrayView>(other)) {
      auto view = other.cast<NCArrayView>();
      other_strides = std::vector<ssize_t>(view.strides(), view.strides() + ndim());
      other_offsets = std::vector<ssize_t>(view.offsets(), view.offsets() + ndim());
      other_data = const_cast<void*>(view.data());
    } else {
      throw py::value_error("Incompatible type - must be array or ncarray type.");
    }

    auto mul_operation = [&]<typename T>() {
      auto mul_op_internal = [](const std::uint8_t* lhs, const std::uint8_t* rhs, T* output) {
        if constexpr (std::is_same_v<T, bool>) {
          *output = *reinterpret_cast<const bool*>(lhs) && *reinterpret_cast<const bool*>(rhs);
        } else {
          *output = *reinterpret_cast<const T*>(lhs) * *reinterpret_cast<const T*>(rhs);
        }
      };

      return binary_op_impl<T, decltype(mul_op_internal), T>(mul_op_internal,
                                                             other_is_arr,
                                                             other_data,
                                                             other_strides.data(),
                                                             other_offsets.data());
    };

    return dispatch(mul_operation);
  }

  py::array NCArrayView::truediv(const py::object& other) const {
    std::vector<ssize_t> other_strides(ndim());
    std::vector<ssize_t> other_offsets(ndim());
    void* other_data { nullptr };

    bool other_is_arr { false };
    if (py::isinstance<py::array>(other)) {
      auto arr = other.cast<py::array>();
      other_strides = std::vector<ssize_t>(arr.strides(), arr.strides() + ndim());
      other_data = const_cast<void*>(arr.data());
      other_is_arr = true;
    } else if (py::isinstance<NCArrayView>(other)) {
      auto view = other.cast<NCArrayView>();
      other_strides = std::vector<ssize_t>(view.strides(), view.strides() + ndim());
      other_offsets = std::vector<ssize_t>(view.offsets(), view.offsets() + ndim());
      other_data = const_cast<void*>(view.data());
    } else {
      throw py::value_error("Incompatible type - must be array or ncarray type.");
    }

    auto truediv_operation = [&]<typename T>() {
      using ResultT = typename op_traits<T>::truediv_type;
      auto truediv_op_internal = [](const std::uint8_t* lhs, const std::uint8_t* rhs, ResultT* output) {
        const T lhs_val = *reinterpret_cast<const T*>(lhs);
        const T rhs_val = *reinterpret_cast<const T*>(rhs);
        if (rhs_val == T(0)) {
          *output = std::isfinite(lhs_val) ? std::nan("") : static_cast<ResultT>(lhs_val);
        } else {
          *output = static_cast<ResultT>(lhs_val) / static_cast<ResultT>(rhs_val);
        }
      };

      return binary_op_impl<T, decltype(truediv_op_internal), ResultT>(truediv_op_internal,
                                                                       other_is_arr,
                                                                       other_data,
                                                                       other_strides.data(),
                                                                       other_offsets.data());
    };

    return dispatch(truediv_operation);
  }

  std::string NCArrayView::format_descriptor() const {
    if (m_dtype.is(py::dtype::of<std::uint8_t>())) {
      return py::format_descriptor<std::uint8_t>::format();
    } else if (m_dtype.is(py::dtype::of<std::uint16_t>())) {
      return py::format_descriptor<std::uint16_t>::format();
    } else if (m_dtype.is(py::dtype::of<std::uint32_t>())) {
      return py::format_descriptor<std::uint32_t>::format();
    } else if (m_dtype.is(py::dtype::of<std::uint64_t>())) {
      return py::format_descriptor<std::uint64_t>::format();
    } else if (m_dtype.is(py::dtype::of<std::int8_t>())) {
      return py::format_descriptor<std::int8_t>::format();
    } else if (m_dtype.is(py::dtype::of<std::int16_t>())) {
      return py::format_descriptor<std::int16_t>::format();
    } else if (m_dtype.is(py::dtype::of<std::int32_t>())) {
      return py::format_descriptor<std::int32_t>::format();
    } else if (m_dtype.is(py::dtype::of<std::int64_t>())) {
      return py::format_descriptor<std::int64_t>::format();
    } else if (m_dtype.is(py::dtype::of<bool>())) {
      return py::format_descriptor<bool>::format();

      //} else if (m_dtype.is(py::dtype::of<float16?>())) {
      //  return py::format_descriptor<float16?>::format();
      //} else if (m_dtype.is(py::dtype::of<float32?>())) {
      //  return py::format_descriptor<float32?>::format();

    } else if (m_dtype.is(py::dtype::of<float>())) {
      return py::format_descriptor<float>::format();
    } else if (m_dtype.is(py::dtype::of<double>())) {
      return py::format_descriptor<double>::format();
    }
    // I guess we have some unhandled type here then...
    throw py::type_error();
  }

  std::pair<void**, bool> NCArrayView::handle_int_indices(ssize_t index,
                                                          ssize_t axis,
                                                          void** curr_data,
                                                          std::vector<ssize_t>& new_shape,
                                                          std::vector<ssize_t>& new_strides,
                                                          std::vector<ssize_t>& new_offsets) const {
    if (index < 0) {
      index += new_shape[axis];
    }
    if (index < 0 || index >= new_shape[axis]) {
      throw py::index_error();
    }

    if (axis != 0) {
      new_offsets[axis] += index * m_strides[axis];
    }

    new_shape.erase(new_shape.begin() + axis);
    new_strides.erase(new_strides.begin() + axis);
    new_offsets.erase(new_offsets.begin() + axis);

    if (axis == 0) {
      return {&curr_data[index], false};
    }

    return {curr_data, true};
  }

  std::pair<void**, bool> NCArrayView::handle_slice_indices(py::slice slice,
                                                            ssize_t axis,
                                                            void** curr_data,
                                                            std::vector<ssize_t>& new_shape,
                                                            std::vector<ssize_t>& new_strides,
                                                            std::vector<ssize_t>& new_offsets) const {
    ssize_t start, stop, step, length;
    if (!slice.compute(m_shape[axis], &start, &stop, &step, &length)) {
      throw py::error_already_set();
    }
    if (start < 0) {
      start += new_shape[axis];
    }
    if (stop < 0) {
      stop += new_shape[axis];
    }
    if (start < 0 || stop < 0 || start >= new_shape[axis] || stop > new_shape[axis]) {
      throw py::index_error();
    }

    new_shape[axis] = length;
    if (axis == 0) {
      if (length == 1) {
        return {&curr_data[start], false};
      }
      return {&curr_data[start], true};
    }
    new_strides[axis] *= step;
    new_offsets[axis] += start * m_strides[axis];
    return {curr_data, true};
  }

  std::pair<void**, bool> NCArrayView::handle_tuple_indices(py::tuple indices,
                                                            ssize_t axis,
                                                            void** curr_data,
                                                            std::vector<ssize_t>& new_shape,
                                                            std::vector<ssize_t>& new_strides,
                                                            std::vector<ssize_t>& new_offsets) const {
    size_t n_specified_dim{indices.size()};
    bool first_axis_ptrs{true};
    // Handle each dimension specified by the tuple
    for (auto arg : indices) {
      if (py::isinstance<py::slice>(arg)) {
        // Dealing with slices
        py::slice slice = arg.cast<py::slice>();
        auto [ret_data, new_first_axis_ptrs] = handle_slice_indices(slice,
                                                                    axis,
                                                                    curr_data,
                                                                    new_shape,
                                                                    new_strides,
                                                                    new_offsets);
        if (!new_first_axis_ptrs) {
          first_axis_ptrs = false;
        }
        curr_data = ret_data;
        axis++;
      } else if (py::isinstance<py::int_>(arg)) {
        // Dealing with single integer indices for the axis
        ssize_t idx = arg.cast<ssize_t>();
        auto [ret_data, new_first_axis_ptrs] = handle_int_indices(idx,
                                                                  axis,
                                                                  curr_data,
                                                                  new_shape,
                                                                  new_strides,
                                                                  new_offsets);
        if (!new_first_axis_ptrs) {
          first_axis_ptrs = false;
        }
        curr_data = ret_data;
      } else if (py::isinstance<py::ellipsis>(arg)) {
        // size_t n_skipped { static_cast<size_t>(ndim()) - n_specified_dim };
        // axis = new_shape.size();
        axis += ndim() - static_cast<ssize_t>(n_specified_dim) + 1;
      } else {
        throw py::index_error("Unrecognized indexing type.");
      }
    }
    return {curr_data, first_axis_ptrs};
  }

  std::variant<NCArrayView, py::array>
  NCArrayView::operator[](py::object slices_or_indices) const {
    void** new_data = m_data;
    std::vector<ssize_t> new_shape = m_shape;
    std::vector<ssize_t> new_strides = m_strides;
    std::vector<ssize_t> new_offsets = m_offsets;

    // Check to see if we still need the double pointers for the first axis
    // if not, we'll just return a NumPy array
    bool new_first_axis_ptrs{true};
    size_t axis{0};
    if (py::isinstance<py::int_>(slices_or_indices)) {
      // Just an integer passed as index
      ssize_t idx = slices_or_indices.cast<ssize_t>();

      auto [ret_data, first_axis_ptrs] = handle_int_indices(idx,
                                                            axis,
                                                            new_data,
                                                            new_shape,
                                                            new_strides,
                                                            new_offsets);
      void* arr_data = ret_data[0];
      return py::array(py::buffer_info(arr_data, itemsize(), format_descriptor().c_str(),
                                       new_shape.size(), new_shape, new_strides));
    } else if (py::isinstance<py::slice>(slices_or_indices)) {
      py::slice slice = slices_or_indices.cast<py::slice>();
      auto [ret_data, first_axis_ptrs] = handle_slice_indices(slice,
                                                              axis,
                                                              new_data,
                                                              new_shape,
                                                              new_strides,
                                                              new_offsets);
      new_first_axis_ptrs = first_axis_ptrs;
    } else if (py::isinstance<py::tuple>(slices_or_indices)) {
      py::tuple tup = slices_or_indices.cast<py::tuple>();
      auto [ret_data, first_axis_ptrs] = handle_tuple_indices(tup,
                                                              axis,
                                                              new_data,
                                                              new_shape,
                                                              new_strides,
                                                              new_offsets);

      new_first_axis_ptrs = first_axis_ptrs;
    }
    if (!new_first_axis_ptrs) {
      return py::array(py::buffer_info(new_data[0],
                                       itemsize(),
                                       format_descriptor().c_str(),
                                       new_shape.size(),
                                       new_shape,
                                       new_strides));
    } else {
      return this->new_sub_view(new_data, new_shape, new_strides, new_offsets, m_dtype);
    }
  }

  std::string NCArrayView::repr() const {
    if (m_shape.empty()) {
      return class_name() + "([], dtype=" + py::str(m_dtype).cast<std::string>() + ")";
    }

    std::string prefix{class_name() + "("};
    std::ostringstream oss;
    oss << prefix;

    // We'll indent subsequent lines to match the opening bracket of the first axis
    size_t indent = prefix.size();
    constexpr size_t edge_items = 3;

    repr_recursive(oss, m_data, 0, indent, edge_items);

    oss << ", dtype=" << py::str(m_dtype).cast<std::string>() << ")";
    return oss.str();
  }

  void NCArrayView::repr_recursive(std::ostringstream& oss, void* current_data, ssize_t axis,
                                   ssize_t indent, ssize_t edge_items) const {
    if (m_dtype.is(py::dtype::of<std::uint8_t>())) {
      repr_recursive_dispatched<std::uint8_t>(oss, current_data, axis, indent, edge_items);
    } else if (m_dtype.is(py::dtype::of<std::uint16_t>())) {
      repr_recursive_dispatched<std::uint16_t>(oss, current_data, axis, indent, edge_items);
    } else if (m_dtype.is(py::dtype::of<std::uint32_t>())) {
      repr_recursive_dispatched<std::uint32_t>(oss, current_data, axis, indent, edge_items);
    } else if (m_dtype.is(py::dtype::of<std::uint64_t>())) {
      repr_recursive_dispatched<std::uint64_t>(oss, current_data, axis, indent, edge_items);
    } else if (m_dtype.is(py::dtype::of<std::int8_t>())) {
      repr_recursive_dispatched<std::int8_t>(oss, current_data, axis, indent, edge_items);
    } else if (m_dtype.is(py::dtype::of<std::int16_t>())) {
      repr_recursive_dispatched<std::int16_t>(oss, current_data, axis, indent, edge_items);
    } else if (m_dtype.is(py::dtype::of<std::int32_t>())) {
      repr_recursive_dispatched<std::int32_t>(oss, current_data, axis, indent, edge_items);
    } else if (m_dtype.is(py::dtype::of<std::int64_t>())) {
      repr_recursive_dispatched<std::int64_t>(oss, current_data, axis, indent, edge_items);
    } else if (m_dtype.is(py::dtype::of<bool>())) {
      repr_recursive_dispatched<bool>(oss, current_data, axis, indent, edge_items);
    } else if (m_dtype.is(py::dtype::of<float>())) {
      repr_recursive_dispatched<float>(oss, current_data, axis, indent, edge_items);
    } else if (m_dtype.is(py::dtype::of<double>())) {
      repr_recursive_dispatched<double>(oss, current_data, axis, indent, edge_items);
    }
  }
  NCArrayView NCArrayView::new_sub_view(void** data,
                                        std::vector<ssize_t>& shape,
                                        std::vector<ssize_t>& strides,
                                        std::vector<ssize_t>& offsets,
                                        py::dtype dtype) const {
    return NCArrayView(data, shape, strides, offsets, dtype);
  }
} // namespace ncarray
