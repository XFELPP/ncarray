#ifndef NCARRAY_NCARRAYVIEW_HH
#define NCARRAY_NCARRAYVIEW_HH

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace ncarray {
  template <typename T>
  struct op_traits {
    using sum_type = T;
    using truediv_type = double;
  };

  // Move small types to larger ones
  template <>
  struct op_traits<int8_t> {
    using sum_type = int64_t;
    using truediv_type = double;
  };

  template <>
  struct op_traits<uint8_t> {
    using sum_type = uint64_t;
    using truediv_type = double;
  };

  class NCArrayView {
  public:
    NCArrayView(void** data_,
                std::vector<ssize_t>& shape_,
                std::vector<ssize_t>& strides_,
                py::dtype dtype_);

    NCArrayView(void** data_,
                std::vector<ssize_t>& shape_,
                std::vector<ssize_t>& strides_,
                std::vector<ssize_t>& offsets_,
                py::dtype dtype_);

    NCArrayView(const NCArrayView& other) = default;
    NCArrayView(NCArrayView&& other) noexcept = default;
    NCArrayView& operator=(const NCArrayView& other) = default;
    NCArrayView& operator=(NCArrayView&& other) noexcept = default;

    virtual ~NCArrayView() = default;

    /**
     * Return a NumPy style string for a __repr__ binding.
     *
     * @return repr_str A string representation of the NCArrayView data. It will
     *         truncate axes which get too long and replace them with "...", as
     *         NumPy's formatting does.
     */
    std::string repr() const;

    /**
     * Support indexing using slices, integers, or tuples thereof.
     *
     * If the new set of indices returns a NumPy-compatible view, return the
     * NumPy array instead of an NCArrayView.
     * TODO: Clean up to remove the duplication of code for various cases.
     */
    std::variant<NCArrayView, py::array>
    operator[](py::object slices_or_indices) const;

    // Basic information - dimensions, shape, etc.
    ssize_t ndim() const {
      return m_shape.size();
    }

    ssize_t itemsize() const {
      return m_dtype.itemsize();
    }

    ssize_t size() const {
      return std::accumulate(m_shape.begin(),
                             m_shape.end(),
                             static_cast<ssize_t>(1),
                             std::multiplies<ssize_t>{});
    }

    ssize_t nbytes() const {
      return size() * itemsize();
    }

    const ssize_t* shape() const {
      return m_shape.data();
    }

    ssize_t shape(ssize_t dim) const {
      if (dim >= static_cast<ssize_t>(m_shape.size())) {
        throw py::index_error();
      }
      return m_shape[dim];
    }

    const ssize_t* strides() const {
      return m_strides.data();
    }

    ssize_t stride(ssize_t dim) const {
      if (dim >= static_cast<ssize_t>(m_strides.size())) {
        throw py::index_error();
      }
      return m_strides[dim];
    }

    const ssize_t* offsets() const {
      return m_offsets.data();
    }

    void* data() const {
      return m_data;
    }

    // Reduction operations
    py::object sum() const;
    py::object max() const;
    py::object min() const;
    py::object mean() const;

    // Binary operations
    py::array add(const py::object& other) const;
    py::array mul(const py::object& other) const;
    py::array div(const py::object& other) const;
    py::array truediv(const py::object& other) const;

  protected:
    template <typename Visitor>
    py::object dispatch(Visitor&& visitor) const {

      if (m_dtype.is(py::dtype::of<std::uint8_t>())) {
        return visitor.template operator()<std::uint8_t>();
      } else if (m_dtype.is(py::dtype::of<std::uint16_t>())) {
        return visitor.template operator()<std::uint16_t>();
      } else if (m_dtype.is(py::dtype::of<std::uint32_t>())) {
        return visitor.template operator()<std::uint32_t>();
      } else if (m_dtype.is(py::dtype::of<std::uint64_t>())) {
        return visitor.template operator()<std::uint64_t>();
      } else if (m_dtype.is(py::dtype::of<std::int8_t>())) {
        return visitor.template operator()<std::int8_t>();
      } else if (m_dtype.is(py::dtype::of<std::int16_t>())) {
        return visitor.template operator()<std::int16_t>();
      } else if (m_dtype.is(py::dtype::of<std::int32_t>())) {
        return visitor.template operator()<std::int32_t>();
      } else if (m_dtype.is(py::dtype::of<std::int64_t>())) {
        return visitor.template operator()<std::int64_t>();
      } else if (m_dtype.is(py::dtype::of<bool>())) {
        return visitor.template operator()<bool>();
      } else if (m_dtype.is(py::dtype::of<float>())) {
        return visitor.template operator()<float>();
      } else if (m_dtype.is(py::dtype::of<double>())) {
        return visitor.template operator()<double>();
      }
      throw py::type_error("Unsupported type for operation");
    }

    template <typename T, typename Op, typename AccumT>
    py::object op_impl(Op op, AccumT identity) const {
      AccumT result = reduce_recursive<T>(m_data, 0, op, identity);
      return py::cast(result);
    }

    template <typename T, typename Op, typename AccumT>
    AccumT reduce_recursive(void* current_data, ssize_t axis, Op op, AccumT acc) const {
      ssize_t dim = m_shape[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(m_shape.size()) - 1);

      for (ssize_t i = 0; i < dim; ++i) {
        if (axis == 0) {
          void* next_ptr = reinterpret_cast<void**>(current_data)[i];
          if (is_last_axis) {
            op(reinterpret_cast<std::uint8_t*>(next_ptr), &acc);
          } else {
            acc = reduce_recursive<T>(next_ptr, axis + 1, op, acc);
          }
        } else {
          std::uint8_t* ptr =
            reinterpret_cast<std::uint8_t*>(current_data) + i * m_strides[axis] + m_offsets[axis];

          if (is_last_axis) {
            op(ptr, &acc);
          } else {
            acc = reduce_recursive<T>(ptr, axis + 1, op, acc);
          }
        }
      }
      return acc;
    }

    template <typename T, typename Op, typename ResultTorAccumT>
    py::object binary_op_impl(Op op,
                              bool other_is_arr,
                              const void* other_data,
                              const ssize_t* other_strides,
                              const ssize_t* other_offsets) const {
      py::array_t<ResultTorAccumT> result(m_shape);
      py::buffer_info result_info = result.request();
      auto* ptr = reinterpret_cast<ResultTorAccumT*>(result_info.ptr);
      binary_op_recursive<T>(m_data,
                             other_data,
                             other_is_arr,
                             other_strides,
                             other_offsets,
                             0,
                             op,
                             ptr);
      return result;
    }

    template <typename T, typename Op, typename ResultTOrAccumT>
    void binary_op_recursive(const void* lhs_data,
                             const void* rhs_data,
                             bool other_is_arr,
                             const ssize_t* other_strides,
                             const ssize_t* other_offsets,
                             ssize_t axis,
                             Op op,
                             ResultTOrAccumT*& res) const {
      // Assume both sides have same shape for this function
      ssize_t dim = m_shape[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(m_shape.size()) - 1);
      for (ssize_t i = 0; i < dim; ++i) {
        if (axis == 0) {
          const void* lhs_next = reinterpret_cast<const void* const*>(lhs_data)[i * m_strides[0]];
          const void* rhs_next { [&]() {
            if (other_is_arr) {
              auto* data =
                reinterpret_cast<const std::uint8_t*>(rhs_data) + i * other_strides[axis];
              return reinterpret_cast<const void*>(data);
            } else {
              return reinterpret_cast<const void* const*>(rhs_data)[i * other_strides[axis]];
            }
          }()};

          if (is_last_axis) {
            op(reinterpret_cast<const std::uint8_t*>(lhs_next),
               reinterpret_cast<const std::uint8_t*>(rhs_next),
               res);
            res++;
          } else {
            binary_op_recursive<T>(lhs_next,
                                   rhs_next,
                                   other_is_arr,
                                   other_strides,
                                   other_offsets,
                                   axis + 1,
                                   op,
                                   res);
          }
        } else {
          const std::uint8_t* lhs_ptr =
            reinterpret_cast<const std::uint8_t*>(lhs_data) + i * m_strides[axis] + m_offsets[axis];
          const std::uint8_t* rhs_ptr = [&]() {
            if (other_is_arr) {
              return reinterpret_cast<const std::uint8_t*>(rhs_data) + i * other_strides[axis];
            } else {
              return
                reinterpret_cast<const std::uint8_t*>(rhs_data) + i * other_strides[axis] + other_offsets[axis];
            }
          }();

          if (is_last_axis) {
            op(lhs_ptr, rhs_ptr, res);
            res++;
          } else {
            binary_op_recursive<T>(lhs_ptr,
                                   rhs_ptr,
                                   other_is_arr,
                                   other_strides,
                                   other_offsets,
                                   axis + 1,
                                   op,
                                   res);
          }
        }
      }
    }


    virtual std::string class_name() const {
      return std::string("NCArrayView");
    }

    /**
     * Recursive helper for repr() that handles arbitrary dimensions.
     */
    void repr_recursive(std::ostringstream& oss,
                        void* current_data,
                        ssize_t axis,
                        ssize_t indent,
                        ssize_t edge_items) const;

    template <class T>
    void repr_recursive_dispatched(std::ostringstream& oss,
                                   void* current_data,
                                   ssize_t axis,
                                   ssize_t indent,
                                   ssize_t edge_items) const {
      ssize_t dim = m_shape[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(m_shape.size()) - 1);
      bool should_truncate = (dim > 2 * edge_items);

      oss << "[";

      auto format_element = [&](size_t i) {
        if (axis == 0) {
          // First axis is a pointer axis
          void* next_ptr = reinterpret_cast<void**>(current_data)[i];
          if (is_last_axis) {
            // Formatting gets garbled with int8/uint8 and ostringstream so cast
            T val = *reinterpret_cast<T*>(next_ptr);
            if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
              if constexpr (std::is_unsigned_v<T>) {
                oss << static_cast<unsigned long>(val);
              } else {
                oss << static_cast<long>(val);
              }
            } else {
              oss << val;
            }
          } else {
            repr_recursive_dispatched<T>(oss, next_ptr, axis + 1, indent + 1, edge_items);
          }
        } else {
          // Subsequent axes use strides and offsets
          uint8_t* ptr =
              reinterpret_cast<uint8_t*>(current_data) + i * m_strides[axis] + m_offsets[axis];
          if (is_last_axis) {
            // Formatting gets garbled with int8/uint8 and ostringstream so cast
            T val = *reinterpret_cast<T*>(ptr);
            if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
              if constexpr (std::is_unsigned_v<T>) {
                oss << static_cast<unsigned long>(val);
              } else {
                oss << static_cast<long>(val);
              }
            } else {
              oss << val;
            }
          } else {
            repr_recursive_dispatched<T>(oss, ptr, axis + 1, indent + 1, edge_items);
          }
        }
      };

      if (!should_truncate) {
        for (ssize_t i = 0; i < dim; ++i) {
          format_element(i);
          if (i < dim - 1) {
            if (is_last_axis) {
              oss << ", ";
            } else {
              oss << ",\n" << std::string(indent + 1, ' ');
              // Extra newline for higher dimensions
              ssize_t ndim{static_cast<ssize_t>(m_shape.size())};
              for (ssize_t j = 0; j < ndim - axis - 2; ++j) {
                oss << "\n";
              }
            }
          }
        }
      } else {
        // Truncate: show first edge_items and last edge_items
        for (ssize_t i = 0; i < edge_items; ++i) {
          format_element(i);
          oss << ", ";
          if (!is_last_axis) {
            oss << "\n" << std::string(indent + 1, ' ');
          }
        }

        oss << "...";

        if (is_last_axis) {
          oss << ", ";
        } else {
          oss << "\n" << std::string(indent + 1, ' ');
        }

        for (ssize_t i = dim - edge_items; i < dim; ++i) {
          format_element(i);
          if (i < dim - 1) {
            if (is_last_axis) {
              oss << ", ";
            } else {
              oss << ",\n" << std::string(indent + 1, ' ');
            }
          }
        }
      }

      oss << "]";
    }

    std::string format_descriptor() const;

  private:
    std::pair<void**, bool> handle_int_indices(ssize_t index,
                                               ssize_t axis,
                                               void** curr_data,
                                               std::vector<ssize_t>& new_shape,
                                               std::vector<ssize_t>& new_strides,
                                               std::vector<ssize_t>& new_offsets) const;

    std::pair<void**, bool> handle_slice_indices(py::slice slice,
                                                 ssize_t axis,
                                                 void** curr_data,
                                                 std::vector<ssize_t>& new_shape,
                                                 std::vector<ssize_t>& new_strides,
                                                 std::vector<ssize_t>& new_offsets) const;

    std::pair<void**, bool> handle_tuple_indices(py::tuple indices,
                                                 ssize_t axis,
                                                 void** curr_data,
                                                 std::vector<ssize_t>& new_shape,
                                                 std::vector<ssize_t>& new_strides,
                                                 std::vector<ssize_t>& new_offsets) const;

    virtual NCArrayView new_sub_view(void** data,
                                     std::vector<ssize_t>& shape,
                                     std::vector<ssize_t>& strides,
                                     std::vector<ssize_t>& offsets,
                                     py::dtype dtype) const;

  protected:
    void** m_data;
    std::vector<ssize_t> m_shape;
    std::vector<ssize_t> m_strides;
    std::vector<ssize_t> m_offsets;
    py::dtype m_dtype;
  };
} // namespace ncarray

#endif // NCARRAY_NCARRAYVIEW_HH
