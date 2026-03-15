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

  std::pair<void**, AxisDescr> NCArrayView::handle_int_indices(ssize_t index,
                                                               ssize_t axis,
                                                               void** curr_data) const {
    if (index < 0) {
      index += m_shape[axis];
    }
    if (index < 0 || index >= m_shape[axis]) {
      throw index_error("Integer index out of bounds!");
    }

    ssize_t length { 1 };
    ssize_t stride { m_strides[axis] };
    ssize_t offset { m_offsets[axis] + index * m_strides[axis] };
    bool is_pointer { false };
    AxisDescr new_axis(axis, length, stride, offset, is_pointer);

    if (get_is_pointer_axis(*this, axis)) {
      new_axis.is_pointer = true;
      new_axis.offset = 0;
      return { reinterpret_cast<void**>(curr_data[index]), new_axis };
    }

    return { curr_data, new_axis };
  }

  std::pair<void**, AxisDescr> NCArrayView::handle_slice_indices(Slice slice,
                                                                 ssize_t axis,
                                                                 void** curr_data) const {
    ssize_t start = slice.start;
    ssize_t stop = slice.stop;
    ssize_t step = slice.step;
    ssize_t length = slice.length;

    if (start < 0) {
      start += m_shape[axis];
    }
    if (stop < 0) {
      stop += m_shape[axis];
    }
    if (start < 0 || stop < 0 || start >= m_shape[axis] || stop > m_shape[axis]) {
      throw index_error("Slice indices out of bounds!");
    }

    ssize_t stride { m_strides[axis] * step };
    ssize_t offset { m_offsets[axis] + start * m_strides[axis] };
    bool is_pointer { false };
    AxisDescr new_axis(axis, length, stride, offset, is_pointer);

    if (get_is_pointer_axis(*this, axis)) {
      new_axis.is_pointer = true;
      if (length == 1) {
        return { reinterpret_cast<void**>(curr_data[start]), new_axis };
      }
      return { &curr_data[start], new_axis };
    }
    return { curr_data, new_axis };
  }

  std::pair<void**, std::vector<AxisDescr>> NCArrayView::handle_tuple_indices(ArrayIndices indices,
                                                                              ssize_t axis,
                                                                              void** curr_data) const {
    size_t n_specified_dim { indices.size() };
    std::vector<AxisDescr> new_axes;
    for (auto arg : indices) {
      if (std::holds_alternative<Slice>(arg)) {
        // Dealing with slices
        Slice slice = std::get<Slice>(arg);
        auto [ret_data, axis_descr] = handle_slice_indices(slice,
                                                           axis,
                                                           curr_data);
        new_axes.push_back(axis_descr);
        curr_data = ret_data;
        axis++;
      } else if (std::holds_alternative<ssize_t>(arg)) {
        ssize_t idx = std::get<ssize_t>(arg);
        auto [ret_data, axis_descr] = handle_int_indices(idx,
                                                         axis,
                                                         curr_data);
        new_axes.push_back(axis_descr);
        curr_data = ret_data;
        axis++;;
      } else if (std::holds_alternative<Ellipsis>(arg)) {
        axis += ndim() - static_cast<ssize_t>(n_specified_dim) + 1;
      } else {
        throw index_error("Unrecognized indexing type.");
      }
    }
    return { curr_data, new_axes };
  }

  ViewOrScalar
  NCArrayView::out_from_axes(void** new_data,
                             std::variant<AxisDescr, std::vector<AxisDescr>> ax_desc) const {
    std::vector<ssize_t> new_shape;
    std::vector<ssize_t> new_strides;
    std::vector<ssize_t> new_offsets;
    ssize_t ptr_axis { -1 };

    std::vector<const AxisDescr*> desc_map(ndim(), nullptr);
    if (std::holds_alternative<AxisDescr>(ax_desc)) {
      desc_map[std::get<AxisDescr>(ax_desc).index] = &std::get<AxisDescr>(ax_desc);
    } else {
      for (const auto& d : std::get<std::vector<AxisDescr>>(ax_desc)) {
        desc_map[d.index] = &d;
      }
    }

    ssize_t total_offset { 0 };
    for (ssize_t i = 0; i < ndim(); ++i) {
      if (desc_map[i]) {
        const auto& d = *desc_map[i];
        // TODO: Consider handling slice separately from length == 1
        // A "NumPy-like" implementation I think would not collapse the dim
        // when you do arr[0:1], whereas it would for arr[0].
        // Currently, NCArray* collapses the dimension in both cases
        if (d.length > 1) {
          new_shape.push_back(d.length);
          new_strides.push_back(d.stride);

          // Accumulate the offset from a prior collapsed dimension
          // and reset after doing so
          new_offsets.push_back(d.offset + total_offset);
          total_offset = 0;

          if (d.is_pointer) {
            ptr_axis = new_shape.size() - 1;
          }
        } else {
          // For a collapsed dimension, accumulate any offset
          total_offset += d.offset;
          if (d.is_pointer) {
            // If the collapsed dimension was a pointer axis, we no longer have one
            ptr_axis = -1;
          }
        }
      } else {
        // Unindexed, ellipsis, or trailing axes
        new_shape.push_back(m_shape[i]);
        new_strides.push_back(m_strides[i]);
        // Accumulate as above
        new_offsets.push_back(m_offsets[i] + total_offset);
        total_offset = 0;

        if (i == m_pointer_axis) {
          ptr_axis = new_shape.size() - 1;
        }
      }
    }

    // If we collapsed all the way to a scalar, total_offset is non-zero and should
    // be baked into the data pointer (it is otherwise tracked in new_offsets)
    // Also it is no longer a double pointer, so pass directly to get_scalar
    if (new_shape.empty()) {
      new_data =
        reinterpret_cast<void**>(reinterpret_cast<std::uint8_t*>(new_data) +
                                 total_offset);
      return get_scalar(new_data);
    } else {
      return
        new_sub_view(new_data, new_shape, new_strides, new_offsets, m_dtype, ptr_axis);
    }
  }

  ViewOrScalar NCArrayView::operator[](ssize_t idx) const {
    void** new_data = m_data;

    size_t axis { 0 };
    auto [ret_data, new_axis] = handle_int_indices(idx, axis, new_data);
    return out_from_axes(ret_data, new_axis);
  }

  ViewOrScalar NCArrayView::operator[](Slice slice) const {
    void** new_data = m_data;

    size_t axis { 0 };
    auto [ret_data, new_axis] = handle_slice_indices(slice, axis, new_data);
    return out_from_axes(ret_data, new_axis);
  }

  ViewOrScalar NCArrayView::operator[](ArrayIndices tup) const {
    void** new_data = m_data;

    size_t axis { 0 };
    auto [ret_data, new_axes] = handle_tuple_indices(tup, axis, new_data);
    return out_from_axes(ret_data, new_axes);
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
    auto internal = [&]<typename T> {
      repr_recursive_dispatched<T>(oss, current_data, axis, indent, edge_items);
    };
    dispatch(m_dtype, internal);
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
