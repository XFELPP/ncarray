#ifndef NCARRAY_ARRAY_OPERATIONS_HH
#define NCARRAY_ARRAY_OPERATIONS_HH

#include "array_traits.hh"
#include "dtype.hh"

#include <cmath>
#include <complex>
#include <concepts>
#include <limits>
#include <stdexcept>

namespace ncarray {

  namespace impl {
    template <typename T, ArrayLike A>
    void fill_recursive(const A& arr, void* current_data, ssize_t axis, T value) {
      ssize_t dim = arr.shape()[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(arr.ndim()) - 1);

      for (ssize_t i = 0; i < dim; ++i) {
        void* next_ptr;
        if (get_is_pointer_axis(arr, axis)) {
          next_ptr = reinterpret_cast<void**>(current_data)[i];
        } else {
          next_ptr = reinterpret_cast<uint8_t*>(current_data) + i * arr.strides()[axis] +
              arr.offsets()[axis];
        }
        if (is_last_axis) {
          *reinterpret_cast<T*>(next_ptr) = value;
        } else {
          fill_recursive<T>(arr, next_ptr, axis + 1, value);
        }
      }
    }

    template <typename DestT, typename SrcT, ArrayLike Dest, ArrayLike Src>
    void assign_recursive(Dest& dest,
                          const Src& src,
                          void* dest_data,
                          const void* src_data,
                          ssize_t axis) {
      ssize_t dim = dest.shape()[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(dest.ndim()) - 1);

      for (ssize_t i = 0; i < dim; ++i) {
        void* next_dest;
        if (get_is_pointer_axis(dest, axis)) {
          next_dest = reinterpret_cast<void**>(dest_data)[i];
        } else {
          next_dest = reinterpret_cast<uint8_t*>(dest_data) + i * dest.strides()[axis] +
              dest.offsets()[axis];
        }

        const void* next_src = [&]() {
          if (get_is_pointer_axis(src, axis)) {
            return reinterpret_cast<const void* const *>(src_data)[i];
          } else {
            auto* data =
              reinterpret_cast<const uint8_t*>(src_data) + i * src.strides()[axis] +
              src.offsets()[axis];
            return reinterpret_cast<const void*>(data);
          }
        }();

        if (is_last_axis) {
          SrcT val = *reinterpret_cast<const SrcT*>(next_src);
          *reinterpret_cast<DestT*>(next_dest) = op_traits<SrcT>::template cast<DestT>(val);
        } else {
          assign_recursive<DestT, SrcT>(dest, src, next_dest, next_src, axis + 1);
        }
      }
    }

    template <typename T, typename OutputType, ArrayLike A>
    void copy_into_recursive(const A& arr,
                             const void* current_src,
                             ssize_t axis,
                             OutputType*& dest) {
      ssize_t dim = arr.shape()[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(arr.ndim()) - 1);

      const ssize_t* strides = arr.strides();
      const ssize_t* offsets = if_has_get_offsets(arr);

      for (ssize_t i = 0; i < dim; ++i) {
        if (get_is_pointer_axis(arr, axis)) {
          const void* next_ptr = reinterpret_cast<const void* const*>(current_src)[i];
          if (is_last_axis) {
            T val = *reinterpret_cast<const T*>(next_ptr);
            *dest = op_traits<T>::template cast<OutputType>(val);
            dest++;
          } else {
            copy_into_recursive<T, OutputType>(arr, next_ptr, axis + 1, dest);
          }
        } else {
          const ssize_t offset = offsets ? offsets[axis] : 0;
          const ssize_t stride = strides[axis];
          const uint8_t* next_ptr =
              reinterpret_cast<const uint8_t*>(current_src) + i * stride + offset;

          if (is_last_axis) {
            T val = *reinterpret_cast<const T*>(next_ptr);
            *dest = op_traits<T>::template cast<OutputType>(val);
            dest++;
          } else {
            copy_into_recursive<T, OutputType>(arr, next_ptr, axis + 1, dest);
          }
        }
      }
    }

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

    template <typename T, typename Op, typename ResultTOrAccumT, ArrayLike A, ArrayLike B>
    void binary_reduce_recursive(const A& left_arr,
                                 const B& right_arr,
                                 const void* lhs_data,
                                 const void* rhs_data,
                                 ssize_t axis,
                                 Op op,
                                 ResultTOrAccumT*& res) {
      // Assume both sides have same shape for this function
      ssize_t dim = left_arr.shape()[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(left_arr.ndim()) - 1);

      const ssize_t* strides_l = left_arr.strides();
      const ssize_t* strides_r = right_arr.strides();

      const ssize_t* offsets_l = if_has_get_offsets(left_arr);
      const ssize_t* offsets_r = if_has_get_offsets(right_arr);

      for (ssize_t i = 0; i < dim; ++i) {
        if (get_is_pointer_axis(left_arr, axis)) {
          const void* lhs_next = reinterpret_cast<const void* const*>(lhs_data)[i * strides_l[axis]];
          const void* rhs_next = [&]() {
            if (get_is_pointer_axis(right_arr, axis)) {
              return reinterpret_cast<const void* const*>(rhs_data)[i * strides_r[axis]];
            } else {
              size_t offset = offsets_r ? offsets_r[axis] : 0;
              auto* data =
                reinterpret_cast<const std::uint8_t*>(rhs_data) + i * strides_r[axis] + offset;
              return reinterpret_cast<const void*>(data);
            }
          }();

          if (is_last_axis) {
            op(reinterpret_cast<const std::uint8_t*>(lhs_next),
               reinterpret_cast<const std::uint8_t*>(rhs_next),
               res);
            res++;
          } else {
            binary_reduce_recursive<T>(left_arr, right_arr, lhs_next, rhs_next, axis + 1, op, res);
          }
        } else {
          size_t offset_l = offsets_l ? offsets_l[axis] : 0;
          const std::uint8_t* lhs_ptr =
            reinterpret_cast<const std::uint8_t*>(lhs_data) + i * strides_l[axis] + offset_l;

          const void* rhs_ptr = [&]() {
            if (get_is_pointer_axis(right_arr, axis)) {
              return reinterpret_cast<const void* const*>(rhs_data)[i * strides_r[axis]];
            } else {
              size_t offset = offsets_r ? offsets_r[axis] : 0;
              auto* data =
                  reinterpret_cast<const std::uint8_t*>(rhs_data) + i * strides_r[axis] + offset;
              return reinterpret_cast<const void*>(data);
            }
          }();

          if (is_last_axis) {
            op(lhs_ptr, reinterpret_cast<const std::uint8_t*>(rhs_ptr), res);
            res++;
          } else {
            binary_reduce_recursive<T>(left_arr, right_arr, lhs_ptr, rhs_ptr, axis + 1, op, res);
          }
        }
      }
    }

    template <typename T, typename ScalarT, typename Op, typename ResultT, ArrayLike A>
    void binary_scalar_recursive(const A& arr,
                                 const void* current_data,
                                 const ScalarT scalar_val,
                                 ssize_t axis,
                                 Op op,
                                 ResultT*& res) {
      ssize_t dim = arr.shape()[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(arr.ndim()) - 1);

      for (ssize_t i = 0; i < dim; ++i) {
        if (get_is_pointer_axis(arr, axis)) {
          const void* next_ptr = reinterpret_cast<const void* const*>(current_data)[i];
          if (is_last_axis) {
            op(reinterpret_cast<const uint8_t*>(next_ptr), scalar_val, res);
            res++;
          } else {
            binary_scalar_recursive<T>(arr, next_ptr, scalar_val, axis + 1, op, res);
          }
        } else {
          const ssize_t* strides = arr.strides();
          const ssize_t* offsets = arr.offsets();
          const uint8_t* ptr =
              reinterpret_cast<const uint8_t*>(current_data) + i * strides[axis] + offsets[axis];
          if (is_last_axis) {
            op(reinterpret_cast<const uint8_t*>(ptr), scalar_val, res);
            res++;
          } else {
            binary_scalar_recursive<T>(arr, ptr, scalar_val, axis + 1, op, res);
          }
        }
      }
    }
  } // namespace impl

  class index_error : public std::invalid_argument {
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

  // Unary reduction operations
  // --------------------------

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

  template <ArrayLike A>
  Scalar max(const A& arr) {
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

  // Binary non-broadcast operations (same shape)
  // --------------------------------------------
  // TODO: In the future, may make ResultType ArrayLike (instead of Owning)
  // but this requires supporting user-provided buffer to put result in
  // TODO: Handle different shape, types and so on for left/right. Not dealt with atm

  template<ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  auto add(const Left& left, const Right& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
      using AccumT = typename op_traits<T>::sum_type;
      return dtype_traits<AccumT>::value;
    });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    auto add_operation = [&]<typename T>() {
      using AccumT = typename op_traits<T>::sum_type;

      auto add_op_internal = [](const std::uint8_t* lhs,
                                const std::uint8_t* rhs,
                                AccumT* output) {
        *output = static_cast<AccumT>(*reinterpret_cast<const T*>(lhs)) + *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      AccumT* result_ptr = reinterpret_cast<AccumT*>(result.data());
      impl::binary_reduce_recursive<T, decltype(add_op_internal), AccumT>(left,
                                                                          right,
                                                                          left.data(),
                                                                          right.data(),
                                                                          starting_axis,
                                                                          add_op_internal,
                                                                          result_ptr);
    };

    dispatch(left.dtype(), add_operation);
    return result;
  }

  template<ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  auto sub(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), left.dtype());

    auto sub_operation = [&]<typename T>() {
      auto sub_op_internal = [](const std::uint8_t* lhs,
                                const std::uint8_t* rhs,
                                T* output) {
        *output = *reinterpret_cast<const T*>(lhs) - *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      T* result_ptr = reinterpret_cast<T*>(result.data());
      impl::binary_reduce_recursive<T, decltype(sub_op_internal), T>(left,
                                                                     right,
                                                                     left.data(),
                                                                     right.data(),
                                                                     starting_axis,
                                                                     sub_op_internal,
                                                                     result_ptr);
    };

    dispatch(left.dtype(), sub_operation);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  auto mul(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), left.dtype());

    auto mul_operation = [&]<typename T>() {
      auto mul_op_internal = [](const std::uint8_t* lhs,
                                const std::uint8_t* rhs,
                                T* output) {
        if constexpr (std::is_same_v<T, bool>) {
          *output = *reinterpret_cast<const bool*>(lhs) && *reinterpret_cast<const bool*>(rhs);
        } else {
          *output = *reinterpret_cast<const T*>(lhs) * *reinterpret_cast<const T*>(rhs);
        }
      };

      ssize_t starting_axis { 0 };
      T* result_ptr = reinterpret_cast<T*>(result.data());
      impl::binary_reduce_recursive<T, decltype(mul_op_internal), T>(left,
                                                                     right,
                                                                     left.data(),
                                                                     right.data(),
                                                                     starting_axis,
                                                                     mul_op_internal,
                                                                     result_ptr);
    };

    dispatch(left.dtype(), mul_operation);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  auto truediv(const Left& left, const Right& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
        using ResultT = typename op_traits<T>::truediv_type;
        return dtype_traits<ResultT>::value;
      });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    auto truediv_operation = [&]<typename T>() {
      using ResultT = typename op_traits<T>::truediv_type;
      auto truediv_op_internal = [](const std::uint8_t* lhs,
                                    const std::uint8_t* rhs,
                                    ResultT* output) {
        const T lhs_val = *reinterpret_cast<const T*>(lhs);
        const T rhs_val = *reinterpret_cast<const T*>(rhs);

        if (rhs_val == T(0)) {
          bool is_finite { false };
          if constexpr (requires { lhs_val.real(); }) {
            is_finite = std::isfinite(lhs_val.real()) && std::isfinite(lhs_val.imag());
          } else {
            is_finite = std::isfinite(lhs_val);
          }
          *output = is_finite ? std::nan("") : static_cast<ResultT>(lhs_val);
        } else {
          *output = static_cast<ResultT>(lhs_val) / static_cast<ResultT>(rhs_val);
        }
      };

      ssize_t starting_axis { 0 };
      ResultT* result_ptr = reinterpret_cast<ResultT*>(result.data());
      impl::binary_reduce_recursive<T, decltype(truediv_op_internal), ResultT>(left,
                                                                               right,
                                                                               left.data(),
                                                                               right.data(),
                                                                               starting_axis,
                                                                               truediv_op_internal,
                                                                               result_ptr);
    };

    dispatch(left.dtype(), truediv_operation);
    return result;
  }

  // --- Binary operations with a scalar broadcast --- //

  template <ArrayLike Left, OwningArrayLike ResultType>
  ResultType add_scalar(const Left& left, const Scalar& right) {
    DType result_dtype = dispatch(left.dtype(), [] <typename T> () {
      using AccumT = typename op_traits<T>::sum_type;

      return dtype_traits<AccumT>::value;
    });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    auto add_operation = [&]<typename T>() {
      using AccumT = typename op_traits<T>::sum_type;

      auto add_op_internal = [](const std::uint8_t* lhs, const T rhs, AccumT* output) {
        *output = static_cast<AccumT>(*reinterpret_cast<const T*>(lhs)) + rhs;
      };

      ssize_t starting_axis { 0 };
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };
      T scalar_val = std::visit(cast_op, right);
      AccumT* result_ptr = reinterpret_cast<AccumT*>(result.data());
      impl::binary_scalar_recursive<T, AccumT>(left,
                                               left.data(),
                                               scalar_val,
                                               starting_axis,
                                               add_op_internal,
                                               result_ptr);
    };

    dispatch(left.dtype(), add_operation);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  ResultType sub_scalar(const Left& left, const Scalar& right) {
    ResultType result(left.ndim(), left.shape(), left.dtype());

    auto sub_operation = [&]<typename T>() {
      auto sub_op_internal = [](const std::uint8_t* lhs, const T rhs, T* output) {
        *output = *reinterpret_cast<const T*>(lhs) - rhs;
      };

      ssize_t starting_axis { 0 };
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };
      T scalar_val = std::visit(cast_op, right);
      T* result_ptr = reinterpret_cast<T*>(result.data());
      impl::binary_scalar_recursive<T>(left,
                                       left.data(),
                                       scalar_val,
                                       starting_axis,
                                       sub_op_internal,
                                       result_ptr);
    };

    dispatch(left.dtype(), sub_operation);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  auto mul_scalar(const Left& left, const Scalar& right) {
    ResultType result(left.ndim(), left.shape(), left.dtype());

    auto mul_operation = [&]<typename T>() {
      auto mul_op_internal = [](const std::uint8_t* lhs, const T rhs, T* output) {
        if constexpr (std::is_same_v<T, bool>) {
          *output = *reinterpret_cast<const bool*>(lhs) && rhs;
        } else {
          *output = *reinterpret_cast<const T*>(lhs) * rhs;
        }
      };

      ssize_t starting_axis { 0 };
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };
      T scalar_val = std::visit(cast_op, right);
      T* result_ptr = reinterpret_cast<T*>(result.data());
      impl::binary_scalar_recursive<T>(left,
                                       left.data(),
                                       scalar_val,
                                       starting_axis,
                                       mul_op_internal,
                                       result_ptr);
    };

    dispatch(left.dtype(), mul_operation);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  auto truediv_scalar(const Left& left, const Scalar& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
        using ResultT = typename op_traits<T>::truediv_type;
        return dtype_traits<ResultT>::value;
      });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    auto truediv_operation = [&]<typename T>() {
      using ResultT = typename op_traits<T>::truediv_type;
      auto truediv_op_internal = [](const std::uint8_t* lhs,
                                    const T rhs,
                                    ResultT* output) {
        const T lhs_val = *reinterpret_cast<const T*>(lhs);

        if (rhs == T(0)) {
          bool is_finite { false };
          if constexpr (requires { lhs_val.real(); }) {
            is_finite = std::isfinite(lhs_val.real()) && std::isfinite(lhs_val.imag());
          } else {
            is_finite = std::isfinite(lhs_val);
          }
          *output = is_finite ? std::nan("") : static_cast<ResultT>(lhs_val);
        } else {
          *output = static_cast<ResultT>(lhs_val) / static_cast<ResultT>(rhs);
        }
      };

      ssize_t starting_axis { 0 };
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };
      T scalar_val = std::visit(cast_op, right);
      ResultT* result_ptr = reinterpret_cast<ResultT*>(result.data());
      impl::binary_scalar_recursive<T>(left,
                                       left.data(),
                                       scalar_val,
                                       starting_axis,
                                       truediv_op_internal,
                                       result_ptr);
    };

    dispatch(left.dtype(), truediv_operation);
    return result;
  }

  // --- Copy and Modification --- //

  template <ArrayLike A>
  void fill(A& arr, Scalar val) {
    auto fill_op = [&] <typename T> () {
      auto fill_op_internal = [](auto&& arg) -> T {
        using FromT = std::decay_t<decltype(arg)>;
        return ncarray::op_traits<FromT>::template cast<T>(arg);
      };
      T target_val = std::visit(fill_op_internal, val);
      impl::fill_recursive<T>(arr, arr.data(), 0, target_val);
    };

    dispatch(arr.dtype(), fill_op);
  }

  template <ArrayLike A, typename OutputType>
  void copy_into(const A& arr, OutputType*& dest) {
    auto copy_op = [&] <typename T> () {
      ssize_t starting_axis { 0 };
      impl::copy_into_recursive<T, OutputType>(arr, arr.data(), starting_axis, dest);
    };
    dispatch(arr.dtype(), copy_op);
  }

  template <ArrayLike Dest, ArrayLike Src>
  void assign(Dest dest, const Src src) {
    // Only deal with identical shapes for now
    if (dest.ndim() != src.ndim()) {
      throw type_error("Shapes must match for assignment");
    }
    for (ssize_t i = 0; i < dest.ndim(); ++i) {
      if (dest.shape(i) != src.shape(i)) {
        throw type_error("Shapes must match for assignment");
      }
    }
    auto assign_op = [&] <typename DestT> () {
      auto assign_op_internal = [&] <typename SrcT> () {

        ssize_t starting_axis { 0 };
        impl::assign_recursive<DestT, SrcT>(dest, src, dest.data(), src.data(), starting_axis);
      };

      dispatch(src.dtype(), assign_op_internal);
    };
    dispatch(dest.dtype(), assign_op);
  }

} // namespace ncarray

#endif // NCARRAY_ARRAY_OPERATIONS_HH
