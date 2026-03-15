#ifndef NCARRAY_NCARRAYVIEW_HH
#define NCARRAY_NCARRAYVIEW_HH

#include "array_operations.hh"
#include "array_traits.hh"
#include "dtype.hh"
#include "indexing.hh"

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ncarray {
  class NCArrayView;
  using ViewOrScalar = std::variant<Scalar, NCArrayView>;

  class NCArrayView {
  public:
    NCArrayView(void** data_,
                const std::vector<ssize_t>& shape_,
                const std::vector<ssize_t>& strides_,
                DType dtype_,
                ssize_t ptr_axis = 0);

    NCArrayView(void** data_,
                const std::vector<ssize_t>& shape_,
                const std::vector<ssize_t>& strides_,
                const std::vector<ssize_t>& offsets_,
                DType dtype_,
                ssize_t ptr_axis = 0);

    // Also provide a constructor with direct pointers
    // This is convenient particularly for construction from Python arrays
    NCArrayView(void** data_,
                const ssize_t ndim,
                const ssize_t* shape_,
                const ssize_t* strides_,
                DType dtype_,
                ssize_t ptr_axis = 0);

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
     */
    ViewOrScalar operator[](ssize_t idx) const;
    ViewOrScalar operator[](Slice slice) const;
    ViewOrScalar operator[](ArrayIndices indices) const;

    ViewOrScalar squeeze() const;

    // Basic information - dimensions, shape, etc.
    inline ssize_t ndim() const { return m_shape.size(); }

    inline ssize_t itemsize() const {
      return static_cast<ssize_t>(ncarray::itemsize(m_dtype));
    }

    inline ssize_t size() const {
      return std::accumulate(m_shape.begin(),
                             m_shape.end(),
                             static_cast<ssize_t>(1),
                             std::multiplies<ssize_t>{});
    }

    inline ssize_t nbytes() const { return size() * itemsize(); }

    inline const ssize_t* shape() const { return m_shape.data(); }

    inline ssize_t shape(ssize_t dim) const {
      if (dim >= static_cast<ssize_t>(m_shape.size())) {
        throw index_error("Requested axis is out of bounds!");
      }
      return m_shape[dim];
    }

    inline const ssize_t* strides() const { return m_strides.data(); }

    inline ssize_t stride(ssize_t dim) const {
      if (dim >= static_cast<ssize_t>(m_strides.size())) {
        throw index_error("Requested axis is out of bounds!");
      }
      return m_strides[dim];
    }

    inline const ssize_t* offsets() const { return m_offsets.data(); }

    inline DType dtype() const { return m_dtype; }

    inline void* data() const { return m_data; }

    bool is_pointer_axis(ssize_t axis) const {
      return axis == m_pointer_axis;
    }

    // Copy, cast, and buffer helpers/utilities
    void copy_into(void* dest_buffer) const;

    template <typename OutT>
    void copy_into_astype(OutT* dest_buffer) const {
      ncarray::copy_into(*this, dest_buffer);
    }

    // TODO: Perhaps this should have some smarter logic to avoid a copy if already
    //       contiguous?
    template<typename R = void, // So compiler evaluates after NCArray (see below too)
             OwningArrayLike ResultType = typename impl::default_owner<R>::type>
    ResultType to_contiguous() const {
      ResultType result(m_shape, m_dtype);
      auto copy_op = [&] <typename T> () {
        T* dest_ptr = reinterpret_cast<T*>(result.data());
        ncarray::copy_into(*this, dest_ptr);
      };

      dispatch(m_dtype, copy_op);

      return result;
    }

    template <typename R = void, // So compiler evaluates after NCArray (see below too)
              OwningArrayLike ResultType = typename impl::default_owner<R>::type>
    ResultType astype(DType& dtype_out) const {
      ResultType result(m_shape, dtype_out);

      auto copy_op = [&]<typename OutT>() {
        OutT* dest_ptr = reinterpret_cast<OutT*>(result.data());
        ncarray::copy_into(*this, dest_ptr);
      };

      dispatch(dtype_out, copy_op);

      return result;
    }

    // Reduction operations
    Scalar sum() const { return ncarray::sum(*this); }

    Scalar max() const { return ncarray::max(*this); }

    Scalar min() const { return ncarray::min(*this); }

    Scalar mean() const { return ncarray::mean(*this); }

    Scalar get_scalar(void* ptr) const {
      auto reduce = [&]<typename T>() -> Scalar {
        return Scalar {*reinterpret_cast<T*>(ptr)};
      };
      return dispatch(m_dtype, reduce);
    }

    // Binary operations
    template <ArrayLike OtherType,
              typename R = void, // Added so compiler evaluates after NCArray defined
              OwningArrayLike ResultType = typename impl::default_owner<R>::type>
    ResultType add(const OtherType& other) const {
      return ncarray::add<NCArrayView, OtherType, ResultType>(*this, other);
    }
    template <ArrayLike OtherType,
              typename R = void,
              OwningArrayLike ResultType = typename impl::default_owner<R>::type>
    ResultType operator+(const OtherType& other) const {
      return ncarray::add<NCArrayView, OtherType, ResultType>(*this, other);
    }

    template <ArrayLike OtherType,
              typename R = void,
              OwningArrayLike ResultType = typename impl::default_owner<R>::type>
    ResultType mul(const OtherType& other) const {
      return ncarray::mul<NCArrayView, OtherType, ResultType>(*this, other);
    }
    template <ArrayLike OtherType,
              typename R = void,
              OwningArrayLike ResultType = typename impl::default_owner<R>::type>
    ResultType operator*(const OtherType& other) const {
      return ncarray::mul<NCArrayView, OtherType, ResultType>(*this, other);
    }

    template <ArrayLike OtherType,
              typename R = void,
              OwningArrayLike ResultType = typename impl::default_owner<R>::type>
    ResultType truediv(const OtherType& other) const {
      return ncarray::truediv<NCArrayView, OtherType, ResultType>(*this, other);
    }
    template <ArrayLike OtherType,
              typename R = void,
              OwningArrayLike ResultType = typename impl::default_owner<R>::type>
    ResultType operator/(const OtherType& other) const {
      return ncarray::truediv<NCArrayView, OtherType, ResultType>(*this, other);
    }

    //NCArray div(const py::object& other) const;

  protected:
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
        if (is_pointer_axis(axis)) {
          void* next_ptr =
            reinterpret_cast<std::uint8_t*>(reinterpret_cast<void**>(current_data)[i]) +
            m_offsets[axis];
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
                oss << "\n" << std::string(indent + 1, ' ');
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
    std::pair<void**, AxisDescr> handle_int_indices(ssize_t index,
                                                    ssize_t axis,
                                                    void** curr_data) const;

    std::pair<void**, AxisDescr> handle_slice_indices(Slice slice,
                                                      ssize_t axis,
                                                      void** curr_data) const;

    std::pair<void**, std::vector<AxisDescr>> handle_tuple_indices(ArrayIndices indices,
                                                                   ssize_t axis,
                                                                   void** curr_data) const;

    ViewOrScalar out_from_axes(void** new_data,
                               std::variant<AxisDescr, std::vector<AxisDescr>> axes) const;

    virtual NCArrayView new_sub_view(void** data,
                                     std::vector<ssize_t>& shape,
                                     std::vector<ssize_t>& strides,
                                     std::vector<ssize_t>& offsets,
                                     DType dtype,
                                     ssize_t ptr_axis = -1) const;

  protected:
    void** m_data;
    std::vector<ssize_t> m_shape;
    std::vector<ssize_t> m_strides;
    std::vector<ssize_t> m_offsets;
    DType m_dtype;

    // If one axis is a pointer axis, indicate here. Otherwise -1
    ssize_t m_pointer_axis { -1 };
  };
} // namespace ncarray

#endif // NCARRAY_NCARRAYVIEW_HH
