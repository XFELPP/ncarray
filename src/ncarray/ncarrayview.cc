#include "ncarrayview.hh"

#include "array_operations.hh"
#include "array_traits.hh"
#include "dtype.hh"
#include "indexing.hh"

//#include <pybind11/numpy.h>
//#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>

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

//namespace py = pybind11;

namespace ncarray {
  NCArrayView::NCArrayView(void** data_,
                           std::vector<ssize_t>& shape_,
                           std::vector<ssize_t>& strides_,
                           DType dtype_)
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
                           DType dtype_)
      : m_data(data_)
      , m_shape(shape_)
      , m_strides(strides_)
      , m_offsets(offsets_)
      , m_dtype(dtype_)
  {}
  /*

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
  */
  /*
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
  */
  std::pair<void**, bool> NCArrayView::handle_int_indices(ssize_t index,
                                                          ssize_t axis,
                                                          void** curr_data,
                                                          std::vector<ssize_t>& new_shape,
                                                          std::vector<ssize_t>& new_strides,
                                                          std::vector<ssize_t>& new_offsets) const {
    size_t shape_diff { static_cast<size_t>(ndim()) - new_shape.size() };
    size_t axis_offset = axis - shape_diff;

    if (index < 0) {
      index += new_shape[axis_offset];
    }
    if (index < 0 || index >= new_shape[axis_offset]) {
      throw index_error("Integer index out of bounds!");
    }

    new_shape.erase(new_shape.begin() + axis_offset);
    new_strides.erase(new_strides.begin() + axis_offset);
    if (new_offsets.size() > 1) {
      // Don't remove the total offset if reducing to a scalar (so we can apply it)
      new_offsets.erase(new_offsets.begin() + axis_offset);
    }

    if (axis == 0) {
      return {&curr_data[index], false};
    }
    new_offsets[axis_offset] += index * m_strides[axis];

    return {curr_data, true};
  }

  std::pair<void**, bool> NCArrayView::handle_slice_indices(Slice slice,
                                                            ssize_t axis,
                                                            void** curr_data,
                                                            std::vector<ssize_t>& new_shape,
                                                            std::vector<ssize_t>& new_strides,
                                                            std::vector<ssize_t>& new_offsets) const {
    size_t shape_diff{static_cast<size_t>(ndim()) - new_shape.size()};
    size_t axis_offset = axis - shape_diff;

    ssize_t start = slice.start;
    ssize_t stop = slice.stop;
    ssize_t step = slice.step;
    ssize_t length = slice.length;

    if (start < 0) {
      start += new_shape[axis_offset];
    }
    if (stop < 0) {
      stop += new_shape[axis_offset];
    }
    if (start < 0 || stop < 0 || start >= new_shape[axis_offset] || stop > new_shape[axis_offset]) {
      throw index_error("Slice indices out of bounds!");
    }

    ssize_t new_axis { axis_offset };

    if (length > 1) {
      new_shape[axis_offset] = length;
    } else {
      new_shape.erase(new_shape.begin() + axis_offset);
      new_strides.erase(new_strides.begin() + axis_offset);
      new_offsets.erase(new_offsets.begin() + axis_offset);
      new_axis -= 1;
    }
    if (axis == 0) {
      if (length == 1) {
        return {&curr_data[start], false};
      }
      return {&curr_data[start], true};
    }
    new_strides[new_axis] *= step;
    new_offsets[new_axis] += start * m_strides[axis];
    return {curr_data, true};
  }

  std::pair<void**, bool> NCArrayView::handle_tuple_indices(ArrayIndices indices,
                                                            ssize_t axis,
                                                            void** curr_data,
                                                            std::vector<ssize_t>& new_shape,
                                                            std::vector<ssize_t>& new_strides,
                                                            std::vector<ssize_t>& new_offsets) const {
    size_t n_specified_dim{indices.size()};
    bool first_axis_ptrs{true};

    // Handle each dimension specified by the tuple
    for (auto arg : indices) {
      if (std::holds_alternative<Slice>(arg)) {
        // Dealing with slices
        Slice slice = std::get<Slice>(arg);
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
      } else if (std::holds_alternative<ssize_t>(arg)) {
        // Dealing with single integer indices for the axis
        ssize_t idx = std::get<ssize_t>(arg);
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
        axis++;
      } else if (std::holds_alternative<Ellipsis>(arg)) {
        axis += ndim() - static_cast<ssize_t>(n_specified_dim) + 1;
      } else {
        throw index_error("Unrecognized indexing type.");
      }
    }
    return {curr_data, first_axis_ptrs};
  }
  /*
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
      new_data = ret_data;
      new_first_axis_ptrs = false;
    } else if (py::isinstance<py::slice>(slices_or_indices)) {
      py::slice slice = slices_or_indices.cast<py::slice>();
      auto [ret_data, first_axis_ptrs] = handle_slice_indices(slice,
                                                              axis,
                                                              new_data,
                                                              new_shape,
                                                              new_strides,
                                                              new_offsets);
      new_first_axis_ptrs = first_axis_ptrs;
      new_data = ret_data;
    } else if (py::isinstance<py::tuple>(slices_or_indices)) {
      py::tuple tup = slices_or_indices.cast<py::tuple>();
      auto [ret_data, first_axis_ptrs] = handle_tuple_indices(tup,
                                                              axis,
                                                              new_data,
                                                              new_shape,
                                                              new_strides,
                                                              new_offsets);

      new_first_axis_ptrs = first_axis_ptrs;
      new_data = ret_data;
    }
    if (!new_first_axis_ptrs) {
      // If we don't have first_axis_ptrs we will be returning an array
      // Offsets are an NCArray* concept only - so adjust the pointer appropriately
      ssize_t total_offset = std::accumulate(new_offsets.begin(),
                                             new_offsets.end(),
                                             static_cast<ssize_t>(0));
      auto* offset_data = reinterpret_cast<std::uint8_t*>(new_data[0]) + total_offset;
      return py::array(py::buffer_info(offset_data,
                                       itemsize(),
                                       format_descriptor().c_str(),
                                       new_shape.size(),
                                       new_shape,
                                       new_strides));
    } else {
      if (new_offsets.size() != new_shape.size()) {
        new_offsets.erase(new_offsets.end() - 1);
      }
      return new_sub_view(new_data, new_shape, new_strides, new_offsets, m_dtype);
    }
  }
 */
  /*
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
  */
  NCArrayView NCArrayView::new_sub_view(void** data,
                                        std::vector<ssize_t>& shape,
                                        std::vector<ssize_t>& strides,
                                        std::vector<ssize_t>& offsets,
                                        DType dtype) const {
    return NCArrayView(data, shape, strides, offsets, dtype);
  }
} // namespace ncarray
