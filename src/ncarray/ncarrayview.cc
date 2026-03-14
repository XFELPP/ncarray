#include "ncarrayview.hh"

#include "array_operations.hh"
#include "array_traits.hh"
#include "dtype.hh"
#include "indexing.hh"

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

namespace ncarray {
  NCArrayView::NCArrayView(void** data_,
                           const std::vector<ssize_t>& shape_,
                           const std::vector<ssize_t>& strides_,
                           DType dtype_,
                           ssize_t ptr_axis)
      : m_data(data_)
      , m_shape(shape_)
      , m_strides(strides_)
      , m_offsets(m_strides.size())
      , m_dtype(dtype_)
      , m_pointer_axis(ptr_axis)
  {}

  NCArrayView::NCArrayView(void** data_,
                           const std::vector<ssize_t>& shape_,
                           const std::vector<ssize_t>& strides_,
                           const std::vector<ssize_t>& offsets_,
                           DType dtype_,
                           ssize_t ptr_axis)
      : m_data(data_)
      , m_shape(shape_)
      , m_strides(strides_)
      , m_offsets(offsets_)
      , m_dtype(dtype_)
      , m_pointer_axis(ptr_axis)
  {}

  NCArrayView::NCArrayView(void** data_,
                           const ssize_t ndim,
                           const ssize_t* shape_,
                           const ssize_t* strides_,
                           DType dtype_,
                           ssize_t ptr_axis)
    : m_data(data_)
    , m_shape(shape_, shape_ + ndim)
    , m_strides(strides_, strides_ + ndim)
    , m_offsets(m_strides.size())
    , m_dtype(dtype_)
    , m_pointer_axis(ptr_axis)
  {}

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

    if (get_is_pointer_axis(*this, axis)) {
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
    size_t shape_diff { static_cast<size_t>(ndim()) - new_shape.size() };
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

    ssize_t new_axis { static_cast<ssize_t>(axis_offset) };

    if (length > 1) {
      new_shape[axis_offset] = length;
    } else {
      new_shape.erase(new_shape.begin() + axis_offset);
      new_strides.erase(new_strides.begin() + axis_offset);
      new_offsets.erase(new_offsets.begin() + axis_offset);
      new_axis -= 1;
    }
    if (get_is_pointer_axis(*this, axis)) {
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

  ViewOrScalar NCArrayView::operator[](ssize_t idx) const {
    void** new_data = m_data;
    std::vector<ssize_t> new_shape = m_shape;
    std::vector<ssize_t> new_strides = m_strides;
    std::vector<ssize_t> new_offsets = m_offsets;

    // Check to see if we still need the double pointers for the first axis
    // if not, we'll just return a NumPy array
    bool new_first_axis_ptrs{true};
    size_t axis{0};

    auto [ret_data, first_axis_ptrs] = handle_int_indices(idx,
                                                          axis,
                                                          new_data,
                                                          new_shape,
                                                          new_strides,
                                                          new_offsets);
    new_data = ret_data;
    new_first_axis_ptrs = false;

    if (new_shape.empty()) {
      return get_scalar(new_data[0]);
    }

    // No pointer axis, so set last arg (pointer axis) to -1
    return new_sub_view(new_data, new_shape, new_strides, new_offsets, m_dtype, -1);
  }

  ViewOrScalar NCArrayView::operator[](Slice slice) const {
    void** new_data = m_data;
    std::vector<ssize_t> new_shape = m_shape;
    std::vector<ssize_t> new_strides = m_strides;
    std::vector<ssize_t> new_offsets = m_offsets;

    // Check to see if we still need the double pointers for the first axis
    // if not, we'll just return a NumPy array
    bool new_first_axis_ptrs{true};
    size_t axis{0};

    auto [ret_data, first_axis_ptrs] = handle_slice_indices(slice,
                                                            axis,
                                                            new_data,
                                                            new_shape,
                                                            new_strides,
                                                            new_offsets);
    new_first_axis_ptrs = first_axis_ptrs;
    new_data = ret_data;

    ssize_t ptr_axis{m_pointer_axis};
    if (!new_first_axis_ptrs) {
      // No pointer axis, so set last arg (pointer axis) to -1
      ptr_axis = -1;
    }
    return new_sub_view(new_data, new_shape, new_strides, new_offsets, m_dtype, ptr_axis);
  }

  ViewOrScalar NCArrayView::operator[](ArrayIndices tup) const {
    void** new_data = m_data;
    std::vector<ssize_t> new_shape = m_shape;
    std::vector<ssize_t> new_strides = m_strides;
    std::vector<ssize_t> new_offsets = m_offsets;

    // Check to see if we still need the double pointers for the first axis
    // if not, we'll just return a NumPy array
    bool new_first_axis_ptrs{true};
    size_t axis{0};

    auto [ret_data, first_axis_ptrs] = handle_tuple_indices(tup,
                                                            axis,
                                                            new_data,
                                                            new_shape,
                                                            new_strides,
                                                            new_offsets);
    new_first_axis_ptrs = first_axis_ptrs;
    new_data = ret_data;

    // ssize_t total_offset =
    //     std::accumulate(new_offsets.begin(), new_offsets.end(), static_cast<ssize_t>(0));

    // auto* offset_data = reinterpret_cast<std::uint8_t*>(new_data[0]) + total_offset;
    if (new_shape.empty()) {
      return get_scalar(new_data[0]);
    }
    ssize_t ptr_axis{m_pointer_axis};
    if (!new_first_axis_ptrs) {
      // No pointer axis, so set last arg (pointer axis) to -1
      ptr_axis = -1;
    }
    return new_sub_view(new_data, new_shape, new_strides, new_offsets, m_dtype, ptr_axis);
  }

  std::string NCArrayView::repr() const {
    if (m_shape.empty()) {
      return class_name() + "([], dtype=" + ncarray::to_string(m_dtype) + ")";
    }

    std::string prefix{class_name() + "("};
    std::ostringstream oss;
    oss << prefix;

    // We'll indent subsequent lines to match the opening bracket of the first axis
    size_t indent = prefix.size();
    constexpr size_t edge_items = 3;

    repr_recursive(oss, m_data, 0, indent, edge_items);

    oss << ", dtype=" << ncarray::to_string(m_dtype) << ")";
    return oss.str();
  }
  void NCArrayView::repr_recursive(std::ostringstream& oss,
                                   void* current_data,
                                   ssize_t axis,
                                   ssize_t indent,
                                   ssize_t edge_items) const {
    dispatch(m_dtype, [&]<typename T> {
               repr_recursive_dispatched<T>(oss, current_data, axis, indent, edge_items);
             });
  }

  NCArrayView NCArrayView::new_sub_view(void** data,
                                        std::vector<ssize_t>& shape,
                                        std::vector<ssize_t>& strides,
                                        std::vector<ssize_t>& offsets,
                                        DType dtype,
                                        ssize_t ptr_axis) const {
    return NCArrayView(data, shape, strides, offsets, dtype, ptr_axis);
  }
} // namespace ncarray
