#ifndef NCARRAY_ARRAY_OPERATIONS_HH
#define NCARRAY_ARRAY_OPERATIONS_HH

// "Doing" - Generic Algorithms

#include "array_traits.hh"
#include "dtype.hh"

#include <concepts>
#include <limits>
#include <stdexcept>

namespace ncarray {

  namespace impl {
    /**
     * Internal recursive engine for reductions.
     * T: The element type (deduced via dispatch)
     * A: The array-like type (View, Ref, etc.)
     */
    template <typename T, typename Op, typename AccumT, ArrayLike A>
    AccumT reduce_recursive(const A& arr,
                            const void* current_data,
                            ssize_t axis,
                            Op op,
                            AccumT acc) {
      ssize_t dim = arr.shape()[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(arr.ndim()) - 1);

      for (ssize_t i = 0; i < dim; ++i) {
        if (get_is_pointer_axis(arr, axis)) {
          const void* next_ptr = reinterpret_cast<const void* const*>(current_data)[i];
          if (is_last_axis) {
            op(reinterpret_cast<const uint8_t*>(next_ptr), &acc);
          } else {
            acc = reduce_recursive<T>(arr, next_ptr, axis + 1, op, acc);
          }
        } else {
          const ssize_t* strides = arr.strides();
          const ssize_t* offsets = arr.offsets();
          const uint8_t* ptr =
            reinterpret_cast<const uint8_t*>(current_data) + i * strides[axis] + offsets[axis];
          if (is_last_axis) {
            op(ptr, &acc);
          } else {
            acc = reduce_recursive<T>(arr, ptr, axis + 1, op, acc);
          }
        }
      }
      return acc;
    }

    /*
     *
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

     */
  } // namespace impl

  class index_error : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
  };

  class type_error : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
  };

  template <typename Visitor>
  auto dispatch(DType type, Visitor&& visitor) {
    switch (type) {
    case DType::bool_: {
      return visitor.template operator()<bool>();
    }
    case DType::char_: {
      return visitor.template operator()<char>();
    }
    // Stuck becuase of std::uint8_t
    //case DType::uchar: {
    //  return visitor.template operator()<unsigned char>();
    //}
    case DType::uint8: {
      return visitor.template operator()<std::uint8_t>();
    }
    case DType::uint16: {
      return visitor.template operator()<std::uint16_t>();
    }
    case DType::uint32: {
      return visitor.template operator()<std::uint32_t>();
    }
    case DType::uint64: {
      return visitor.template operator()<std::uint64_t>();
    }
    case DType::int8: {
      return visitor.template operator()<std::int8_t>();
    }
    case DType::int16: {
      return visitor.template operator()<std::int16_t>();
    }
    case DType::int32: {
      return visitor.template operator()<std::int32_t>();
    }
    case DType::int64: {
      return visitor.template operator()<std::int64_t>();
    }
    case DType::float32: {
      return visitor.template operator()<float>();
    }
    case DType::float64: {
      return visitor.template operator()<double>();
    }
    case DType::complex64: {
      return visitor.template operator()<std::complex<float>>();
    }
    case DType::complex128: {
      return visitor.template operator()<std::complex<double>>();
    }
    case DType::complex256: {
      return visitor.template operator()<std::complex<long double>>();
    }
    }
    throw type_error("Unsupported type for operation");
  }

  template <ArrayLike A>
  Scalar sum(const A& arr) {
    auto sum_operation = [&]<typename T>() -> Scalar {
      using AccumT = typename op_traits<T>::sum_type;

      auto sum_op_internal = [](const std::uint8_t* data, AccumT* output) {
        *output += static_cast<AccumT>(*reinterpret_cast<const T*>(data));
      };

      ssize_t starting_axis { 0 };
      AccumT result = impl::reduce_recursive<T>(arr,
                                                arr.data(),
                                                starting_axis,
                                                sum_op_internal,
                                                AccumT {0});
      return Scalar {result};
    };

    return dispatch(arr.dtype(), sum_operation);
  }

  template <ArrayLike A>
  Scalar mean(const A& arr) {
    auto sum_operation = [&]<typename T>() -> Scalar {
      using AccumT = typename op_traits<T>::sum_type;
      using ResultT = typename op_traits<T>::truediv_type;

      auto sum_op_internal = [](const std::uint8_t* data, AccumT* output) {
        *output += static_cast<AccumT>(*reinterpret_cast<const T*>(data));
      };

      ssize_t starting_axis{0};
      AccumT result = impl::reduce_recursive<T>(arr,
                                                arr.data(),
                                                starting_axis,
                                                sum_op_internal,
                                                AccumT{0});
      ResultT mean = static_cast<ResultT>(result) / static_cast<double>(arr.size());
      return Scalar {mean};
    };

    return dispatch(arr.dtype(), sum_operation);
  }

  template <ArrayLike A> Scalar max(const A& arr) {
    auto max_operation = [&]<typename T>() {
      // Don't need a broader type for this one
      auto max_op_internal = [](const std::uint8_t* data, T* output) {
        T val = *reinterpret_cast<const T*>(data);
        if (op_traits<T>::greater(val, *output)) {
          *output = val;
        }
      };
      ssize_t starting_axis {0};
      T result = impl::reduce_recursive<T>(arr,
                                           arr.data(),
                                           starting_axis,
                                           max_op_internal,
                                           op_traits<T>::lowest());
      return Scalar{result};
    };

    return dispatch(arr.dtype(), max_operation);
  }

  template <ArrayLike A>
  Scalar min(const A& arr) {
    auto min_operation = [&]<typename T>() {
      // Don't need a broader type for this one
      auto min_op_internal = [](const std::uint8_t* data, T* output) {
        T val = *reinterpret_cast<const T*>(data);
        if (op_traits<T>::less(val, *output)) {
          *output = val;
        }
      };
      ssize_t starting_axis {0};
      T result = impl::reduce_recursive<T>(arr,
                                           arr.data(),
                                           starting_axis,
                                           min_op_internal,
                                           op_traits<T>::max());
      return Scalar {result};
    };

    return dispatch(arr.dtype(), min_operation);
  }
} // namespace ncarray

#endif // NCARRAY_ARRAY_OPERATIONS_HH
