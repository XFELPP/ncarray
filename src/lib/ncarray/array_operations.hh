/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ARRAY_OPERATIONS_HH
#define NCARRAY_ARRAY_OPERATIONS_HH

#include "ncarray/array_traits.hh"
#include "ncarray/dtype.hh"
#include "ncarray/array_impl.hh"
#ifdef __CUDACC__
#include "ncarray/engines.hh"
#endif

#include <cmath>
#include <complex>
#include <concepts>
#include <limits>
#include <stdexcept>

namespace ncarray {

  namespace impl {
    /**
     * Recursively fill an array with with a scalar value.
     *
     * This function can cast the value as it is assigned.
     *
     * @tparam T The type of the value to fill into the array.
     * @tparam A The type of the array being operated on (ArrayLike constrained.)
     * @param[in] arr The array.
     * @param[out] current_data The pointer to the current data in the array.
     * @param[in] axis The current axis being traversed.
     * @param[in] value The value to fill into the array.
     */
    template <typename T, ArrayLike A>
    void fill_recursive(const A& arr, void* current_data, ssize_t axis, T value) {
      ssize_t dim = arr.shape()[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(arr.ndim()) - 1);

      for (ssize_t i = 0; i < dim; ++i) {
        void* next_ptr = arr.advance(current_data, axis, i);
        if (is_last_axis) {
          *reinterpret_cast<T*>(next_ptr) = value;
        } else {
          fill_recursive<T>(arr, next_ptr, axis + 1, value);
        }
      }
    }

    /**
     * Recursively assign one array to another. The shapes MUST match.
     *
     * This function can cast the values as they are assigned.
     *
     * @tparam DestT The dtype of the destination array.
     * @tparam SrcT The dtype of the source array.
     * @tparam Dest The type of array to assign to. (ArrayLike constrained.)
     * @tparam Src The type of array to pull values from. (ArrayLike constrained.)
     * @param[out] dest The array to assign to.
     * @param[in] src The array to pull values from.
     * @param[in] dest_data The pointer to the current data in the destination.
     * @param[in] src_data The pointer to the current data in the source.
     * @param[in] axis The current axis being traversed.
     */
    template <typename DestT, typename SrcT, ArrayLike Dest, ArrayLike Src>
    void assign_recursive(Dest& dest,
                          const Src& src,
                          void* dest_data,
                          const void* src_data,
                          ssize_t axis) {
      ssize_t dim = dest.shape()[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(dest.ndim()) - 1);

      for (ssize_t i = 0; i < dim; ++i) {
        void* next_dest = dest.advance(dest_data, axis, i);

        const void* next_src = const_cast<const void*>(src.advance(src_data, axis, i));

        if (is_last_axis) {
          SrcT val = *reinterpret_cast<const SrcT*>(next_src);
          *reinterpret_cast<DestT*>(next_dest) = op_traits<SrcT>::template cast<DestT>(val);
        } else {
          assign_recursive<DestT, SrcT>(dest, src, next_dest, next_src, axis + 1);
        }
      }
    }

    /**
     * Recursively copy an array into an output location.
     *
     * This function can cast the value as it is assigned.
     *
     * @tparam T The dtype for the source array.
     * @tparam OutputType The dtype for the destination array.
     * @param[in] arr The array.
     * @param[in] current_src The pointer to the current data in the source array.
     * @param[in] axis The current axis being traversed.
     * @param[out] dest Reference to pointer for the destination array.
     */
    template <typename T, typename OutputType, ArrayLike A>
    void copy_into_recursive(const A& arr,
                             const void* current_src,
                             ssize_t axis,
                             OutputType*& dest) {
      ssize_t dim = arr.shape()[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(arr.ndim()) - 1);

      for (ssize_t i = 0; i < dim; ++i) {
        const void* next_ptr = const_cast<const void*>(arr.advance(current_src, axis, i));

        if (is_last_axis) {
          T val = *reinterpret_cast<const T*>(next_ptr);
          *dest = op_traits<T>::template cast<OutputType>(val);
          dest++;
        } else {
          copy_into_recursive<T, OutputType>(arr, next_ptr, axis + 1, dest);
        }
      }
    }

    /**
     * Recursively reduce an array to a scalar.
     *
     * This function can cast the value as it is assigned.
     *
     * @tparam T The dtype for the source array.
     * @tparam Op The type of the reduction function.
     * @tparam AccumT The type of the value being accumulated into.
     * @tparam A The type of the array object.
     * @param[in] arr The array.
     * @param[in] current_data The pointer to the current data in the source array.
     * @param[in] axis The current axis being traversed.
     * @param[in] op The reduction operation.
     * @param[in] acc The identity value for the reduction. (on recursion it accumulates)
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
        const void* next_ptr = const_cast<const void*>(arr.advance(current_data, axis, i));

        if (is_last_axis) {
          op(reinterpret_cast<const std::uint8_t*>(next_ptr), &acc);
        } else {
          acc = reduce_recursive<T>(arr, next_ptr, axis + 1, op, acc);
        }
      }
      return acc;
    }

    /**
     * Recursively operate on two arrays.
     *
     * E.g. this function can do binary add/subtract/multiply and so on.
     *
     * This function can cast the value as it is assigned.
     *
     * @tparam T The dtype for the source array.
     * @tparam Op The type of the reduction function.
     * @tparam AccumT The type of the value being accumulated into.
     * @tparam A The type of the left hand side array.
     * @tparam B The type of the right hand side array.
     * @param[in] left_arr The left hand side array.
     * @param[in] right_arr The right hand side array.
     * @param[in] lhs_data The pointer to the current data in the left hand side array.
     * @param[in] right_data The pointer to the current right hand side's data.
     * @param[in] axis The current axis being traversed.
     * @param[in] op The reduction operation.
     * @param[out] res The output array's data.
     */
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

      for (ssize_t i = 0; i < dim; ++i) {
        const void* lhs_next = const_cast<const void*>(left_arr.advance(lhs_data, axis, i));
        const void* rhs_next = const_cast<const void*>(right_arr.advance(rhs_data, axis, i));

        if (is_last_axis) {
          op(reinterpret_cast<const std::uint8_t*>(lhs_next),
             reinterpret_cast<const std::uint8_t*>(rhs_next), res);
          res++;
        } else {
          binary_reduce_recursive<T>(left_arr, right_arr, lhs_next, rhs_next, axis + 1, op, res);
        }
      }
    }

    /**
     * Recursively operate on an array, broadcasting a scalar.
     *
     * E.g. this function can do binary add/subtract/multiply and so on.
     *
     * This function can cast the value as it is assigned.
     *
     * @tparam T The dtype for the source array.
     * @tparam ScalarT The type of the scalar.
     * @tparam Op The type of the reduction function.
     * @tparam ResultT The type of the output array.
     * @tparam A The type of the left hand side array.
     * @param[in] arr The array.
     * @param[in] current_data The pointer to the current data in the source array.
     * @param[in] scalar_val The scalar to broadcast.
     * @param[in] axis The current axis being traversed.
     * @param[in] op The reduction operation.
     * @param[out] res The output array.
     */
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
        const void* next_ptr = const_cast<const void*>(arr.advance(current_data, axis, i));
        if (is_last_axis) {
          op(reinterpret_cast<const uint8_t*>(next_ptr), scalar_val, res);
          res++;
        } else {
          binary_scalar_recursive<T>(arr, next_ptr, scalar_val, axis + 1, op, res);
        }
      }
    }
  } // namespace impl

  class index_error : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
  };

  template <typename Visitor> auto dispatch(DType type, Visitor&& visitor) {
    switch (type) {
    case DType::bool_: {
      return visitor.template operator()<bool>();
    }
    case DType::char_: {
      return visitor.template operator()<char>();
    }
    // Stuck becuase of std::uint8_t
    // case DType::uchar: {
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
      AccumT result =
          impl::reduce_recursive<T>(arr, arr.data(), starting_axis, sum_op_internal, AccumT{0});
      return Scalar{result};
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

      ssize_t starting_axis { 0 };
      AccumT result =
          impl::reduce_recursive<T>(arr,
                                    arr.data(),
                                    starting_axis,
                                    sum_op_internal,
                                    AccumT{0});
      ResultT mean = static_cast<ResultT>(result) / static_cast<double>(arr.size());
      return Scalar{mean};
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
      ssize_t starting_axis { 0 };
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
      ssize_t starting_axis { 0 };
      T result = impl::reduce_recursive<T>(arr,
                                           arr.data(),
                                           starting_axis,
                                           min_op_internal,
                                           op_traits<T>::max());
      return Scalar{result};
    };

    return dispatch(arr.dtype(), min_operation);
  }

  template <class L, class S>
  Scalar ArrayImpl<L, S>::sum() const {
    return ncarray::sum(*this);
  }

  template <class L, class S>
  Scalar ArrayImpl<L, S>::max() const {
    return ncarray::max(*this);
  }

  template <class L, class S>
  Scalar ArrayImpl<L, S>::min() const {
    return ncarray::min(*this);
  }

  template <class L, class S>
  Scalar ArrayImpl<L, S>::mean() const {
    return ncarray::mean(*this);
  }

  // Binary non-broadcast operations (same shape)
  // --------------------------------------------
  // TODO: In the future, may make ResultType ArrayLike (instead of Owning)
  // but this requires supporting user-provided buffer to put result in
  // TODO: Handle different shape, types and so on for left/right. Not dealt with atm

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  auto add(const Left& left, const Right& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
      using AccumT = typename op_traits<T>::sum_type;
      return dtype_traits<AccumT>::value;
    });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    auto add_operation = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_add(left, right, result.view());
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA GPU kernel with GCC.");
#endif
      } else {
        using AccumT = typename op_traits<T>::sum_type;

        auto add_op_internal = [](const std::uint8_t* lhs, const std::uint8_t* rhs, AccumT* output) {
          *output = static_cast<AccumT>(*reinterpret_cast<const T*>(lhs)) +
            *reinterpret_cast<const T*>(rhs);
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
      }
    };

    dispatch(left.dtype(), add_operation);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  auto sub(const Left& left, const Right& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
      using DiffT = typename op_traits<T>::diff_type;
      return dtype_traits<DiffT>::value;
    });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    auto sub_operation = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_sub(left, right, result.view());
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA GPU kernel with GCC.");
#endif
      } else {
        using DiffT = typename op_traits<T>::diff_type;
        auto sub_op_internal = [](const std::uint8_t* lhs, const std::uint8_t* rhs, DiffT* output) {
          *output =
            static_cast<DiffT>(*reinterpret_cast<const T*>(lhs)) - *reinterpret_cast<const T*>(rhs);
        };

        ssize_t starting_axis { 0 };
        DiffT* result_ptr = reinterpret_cast<DiffT*>(result.data());
        impl::binary_reduce_recursive<T, decltype(sub_op_internal), DiffT>(left,
                                                                           right,
                                                                           left.data(),
                                                                           right.data(),
                                                                           starting_axis,
                                                                           sub_op_internal,
                                                                           result_ptr);
      }
    };

    dispatch(left.dtype(), sub_operation);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  auto mul(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), left.dtype());

    auto mul_operation = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_mul(left, right, result.view());
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA GPU kernel with GCC.");
#endif
      } else {
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
      }
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
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_truediv(left, right, result.view());
#else
        throw std::runtime_error("Fatal: tried to compile a cuda GPU kernel with GCC.");
#endif
      }
      auto truediv_op_internal = [](const std::uint8_t* lhs,
                                    const std::uint8_t* rhs,
                                    ResultT* output) {
        const T lhs_val = *reinterpret_cast<const T*>(lhs);
        const T rhs_val = *reinterpret_cast<const T*>(rhs);

        if (rhs_val == T(0)) {
          bool is_finite{false};
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

  template <class L, class S>
  template <ArrayLike OtherType>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::add(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::add<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::operator+(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::add<ViewType, OtherType, OwnerType>(*this, other);
  }

  template <class L, class S>
  template <ArrayLike OtherType>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::sub(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::sub<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::operator-(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::sub<ViewType, OtherType, OwnerType>(*this, other);
  }

  template <class L, class S>
  template <ArrayLike OtherType>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::mul(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::mul<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::operator*(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::mul<ViewType, OtherType, OwnerType>(*this, other);
  }

  template <class L, class S>
  template <ArrayLike OtherType>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::truediv(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::truediv<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::operator/(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::truediv<ViewType, OtherType, OwnerType>(*this, other);
  }

  // --- Binary operations with a scalar broadcast --- //

  template <ArrayLike Left, OwningArrayLike ResultType>
  ResultType add_scalar(const Left& left, const Scalar& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
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
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
      using DiffT = typename op_traits<T>::diff_type;
      return dtype_traits<DiffT>::value;
    });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    auto sub_operation = [&]<typename T>() {
      using DiffT = typename op_traits<T>::diff_type;

      auto sub_op_internal = [](const std::uint8_t* lhs, const T rhs, DiffT* output) {
        *output = static_cast<DiffT>(*reinterpret_cast<const T*>(lhs)) - rhs;
      };

      ssize_t starting_axis { 0 };
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };
      T scalar_val = std::visit(cast_op, right);
      DiffT* result_ptr = reinterpret_cast<DiffT*>(result.data());
      impl::binary_scalar_recursive<T, DiffT>(left,
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
      auto truediv_op_internal = [](const std::uint8_t* lhs, const T rhs, ResultT* output) {
        const T lhs_val = *reinterpret_cast<const T*>(lhs);

        if (rhs == T(0)) {
          bool is_finite{false};
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

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::add(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::add_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::operator+(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::add_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::sub(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::sub_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::operator-(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::sub_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::mul(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::mul_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::operator*(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::mul_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::truediv(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::truediv_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::operator/(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::truediv_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  // --- Iterators --- //
  template <typename L, typename S>
  typename ArrayImpl<L, S>::Iterator ArrayImpl<L, S>::begin() {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using Iterator = typename ArrayImpl<L, S>::Iterator;

    Metadata offset_type;
    if constexpr (requires { this->m_offsets; }) {
      offset_type = this->m_offsets;
    } else {
      offset_type = this->m_suboffsets;
    }

    return Iterator(ViewType(*this), 0);
  }

  template <typename L, typename S>
  typename ArrayImpl<L, S>::Iterator ArrayImpl<L, S>::end() {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using Iterator = typename ArrayImpl<L, S>::Iterator;

    Metadata offset_type;
    if constexpr (requires { this->m_offsets; }) {
      offset_type = this->m_offsets;
    } else {
      offset_type = this->m_suboffsets;
    }

    ssize_t len = this->ndim() > 0 ? this->m_shape[0] : 0;

    return Iterator(ViewType(*this), len);
  }

  template <typename L, typename S>
  typename ArrayImpl<L, S>::ConstIterator ArrayImpl<L, S>::begin() const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using ConstIterator = typename ArrayImpl<L, S>::ConstIterator;

    Metadata offset_type;
    if constexpr (requires { this->m_offsets; }) {
      offset_type = this->m_offsets;
    } else {
      offset_type = this->m_suboffsets;
    }

    return ConstIterator(ViewType(*this), 0);
  }

  template <typename L, typename S>
  typename ArrayImpl<L, S>::ConstIterator ArrayImpl<L, S>::end() const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using ConstIterator = typename ArrayImpl<L, S>::ConstIterator;

    Metadata offset_type;
    if constexpr (requires { this->m_offsets; }) {
      offset_type = this->m_offsets;
    } else {
      offset_type = this->m_suboffsets;
    }

    ssize_t len = this->ndim() > 0 ? this->m_shape[0] : 0;

    return ConstIterator(ViewType(*this), len);
  }

  // --- Copy and Modification --- //
  template <ArrayLike A>
  void fill(A& arr, Scalar val) {
    auto fill_op = [&]<typename T>() {
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
    auto copy_op = [&]<typename T>() {
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
    auto assign_op = [&]<typename DestT>() {
      auto assign_op_internal = [&]<typename SrcT>() {
        ssize_t starting_axis { 0 };
        impl::assign_recursive<DestT, SrcT>(dest,
                                            src,
                                            dest.data(),
                                            src.data(),
                                            starting_axis);
      };

      dispatch(src.dtype(), assign_op_internal);
    };
    dispatch(dest.dtype(), assign_op);
  }

  template <class L, class S>
  void ArrayImpl<L, S>::copy_into(void* dest_buffer) const {
    auto copy_op = [&]<typename T>() {
      T* dest_ptr = reinterpret_cast<T*>(dest_buffer);
      ncarray::copy_into(*this, dest_ptr);
    };

    dispatch(this->m_dtype, copy_op);
  }

  template <class L, class S>
  template <typename OutT>
  void ArrayImpl<L, S>::copy_into_astype(OutT* dest_buffer) const {
    ncarray::copy_into(*this, dest_buffer);
  }

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::to_contiguous() const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    OwnerType result(this->m_shape, this->m_dtype);

    auto copy_op = [&]<typename T>() {
      T* dest_ptr = reinterpret_cast<T*>(result.data());
      ncarray::copy_into(*this, dest_ptr);
    };

    dispatch(this->m_dtype, copy_op);

    return result;
  }

  template <class L, class S>
  typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::astype(DType& dtype_out) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    OwnerType result(this->m_shape, this->m_dtype);

    auto copy_op = [&]<typename OutT>() {
      OutT* dest_ptr = reinterpret_cast<OutT*>(result.data());
      ncarray::copy_into(*this, dest_ptr);
    };

    dispatch(dtype_out, copy_op);

    return result;
  }

  template <class L, class S>
  void ArrayImpl<L, S>::assign(ArrayLike auto arr) {
    if (this->m_read_only) {
      throw type_error("Cannot modify a read-only view!");
    }

    ncarray::assign(*this, arr);
  }

  template <class L, class S>
  void ArrayImpl<L, S>::fill(Scalar val) {
    if (this->m_read_only) {
      throw type_error("Cannot modify a read-only view!");
    }

    ncarray::fill(*this, val);
  }

} // namespace ncarray

#endif // NCARRAY_ARRAY_OPERATIONS_HH
