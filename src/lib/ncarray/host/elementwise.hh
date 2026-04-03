/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_HOST_ELEMENTWISE_HH
#define NCARRAY_HOST_ELEMENTWISE_HH

#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"

namespace ncarray {
  namespace host {
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
        if (axis == static_cast<ssize_t>(arr.ndim())) {
          // Base case handles when you index to a scalar (the loop below wouldn't enter)
          *reinterpret_cast<T*>(current_data) = value;
          return;
        }
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
        if (axis == static_cast<ssize_t>(dest.ndim())) {
          // Base case handles when you index to a scalar (the loop below wouldn't enter)
          SrcT val = *reinterpret_cast<const SrcT*>(src_data);
          *reinterpret_cast<DestT*>(dest_data) = op_traits<SrcT>::template cast<DestT>(val);
          return;
        }
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
       * Recursively operate on a single array for unary operations (like logical not).
       *
       * This function can cast the value as it is assigned.
       *
       * @tparam T The dtype for the source array.
       * @tparam Op The type of the reduction function.
       * @tparam AccumT The type of the value being accumulated into.
       * @tparam A The type of the array.
       * @param[in] arr The input array.
       * @param[in] data The pointer to the current data in the array.
       * @param[in] axis The current axis being traversed.
       * @param[in] op The reduction operation.
       * @param[out] res The output array's data.
       */
      template <typename T, typename Op, typename ResultTOrAccumT, ArrayLike A>
      void unary_reduce_recursive(const A& arr,
                                  const void* data,
                                  ssize_t axis,
                                  Op op,
                                  ResultTOrAccumT*& res) {
        // Assume both sides have same shape for this function
        ssize_t dim = arr.shape()[axis];
        bool is_last_axis = (axis == static_cast<ssize_t>(arr.ndim()) - 1);

        for (ssize_t i = 0; i < dim; ++i) {
          const void* next = const_cast<const void*>(arr.advance(data, axis, i));

          if (is_last_axis) {
            op(reinterpret_cast<const std::uint8_t*>(next), res);
            res++;
          } else {
            unary_reduce_recursive<T>(arr, next, axis + 1, op, res);
          }
        }
      }

      /**
       * Recursively operate on two arrays.
       *
       * This function assumes the arrays are the same shape, except if there are
       * extra padding dimensions. In that case (i.e., leading dimensions of size 1)
       * a shape mismatch is tolerated.
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

          // Align dimensions from the right side
          ssize_t r_axis =
            axis - (static_cast<ssize_t>(left_arr.ndim()) - static_cast<ssize_t>(right_arr.ndim()));

          const void* rhs_next = (r_axis >= 0)
            ? right_arr.advance(rhs_data, r_axis, (right_arr.shape(r_axis) == 1 ? 0 : i))
            : rhs_data;

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
       * Recursively operate inplace on an array for unary operations.
       *
       * This function can cast the value as it is assigned.
       *
       * @tparam T The dtype for the source array.
       * @tparam Op The type of the reduction function.
       * @tparam A The type of the array.
       * @param[in] arr The input array.
       * @param[in] data The pointer to the current data in the array.
       * @param[in] axis The current axis being traversed.
       * @param[in] op The reduction operation.
       * @param[out] res The output array's data.
       */
      template <typename T, typename Op, ArrayLike A>
      void inplace_unary_reduce_recursive(const A& arr,
                                          const void* data,
                                          ssize_t axis,
                                          Op op) {
        // Assume both sides have same shape for this function
        ssize_t dim = arr.shape()[axis];
        bool is_last_axis = (axis == static_cast<ssize_t>(arr.ndim()) - 1);

        for (ssize_t i = 0; i < dim; ++i) {
          void* next = arr.advance(data, axis, i);

          if (is_last_axis) {
            op(reinterpret_cast<std::uint8_t*>(next));
          } else {
            inplace_unary_reduce_recursive<T>(arr, next, axis + 1, op);
          }
        }
      }

      /**
       * Binary operation recursed inplace.
       *
       * This function assumes the arrays are the same shape, except if there are
       * extra padding dimensions. In that case (i.e., leading dimensions of size 1)
       * a shape mismatch is tolerated.
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
       */
      template <typename T, typename Op, ArrayLike A, ArrayLike B>
      void inplace_binary_reduce_recursive(const A& left_arr,
                                           const B& right_arr,
                                           const void* lhs_data,
                                           const void* rhs_data,
                                           ssize_t axis,
                                           Op op) {
        // Assume both sides have same shape for this function
        ssize_t dim = left_arr.shape()[axis];
        bool is_last_axis = (axis == static_cast<ssize_t>(left_arr.ndim()) - 1);

        for (ssize_t i = 0; i < dim; ++i) {
          void* lhs_next = const_cast<void*>(left_arr.advance(lhs_data, axis, i));

          // Align dimensions from the right side
          ssize_t r_axis =
            axis - (static_cast<ssize_t>(left_arr.ndim()) - static_cast<ssize_t>(right_arr.ndim()));

          const void* rhs_next = (r_axis >= 0)
            ? right_arr.advance(rhs_data, r_axis, (right_arr.shape(r_axis) == 1 ? 0 : i))
            : rhs_data;

          if (is_last_axis) {
            op(reinterpret_cast<std::uint8_t*>(lhs_next),
               reinterpret_cast<const std::uint8_t*>(rhs_next));
          } else {
            inplace_binary_reduce_recursive<T>(left_arr,
                                               right_arr,
                                               lhs_next,
                                               rhs_next,
                                               axis + 1,
                                               op);
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

      /**
       * Binary operation recursed inplace, broadcasting a scalar.
       *
       * This function assumes the arrays are the same shape, except if there are
       * extra padding dimensions. In that case (i.e., leading dimensions of size 1)
       * a shape mismatch is tolerated.
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
       */
      template <typename T, typename ScalarT, typename Op, ArrayLike A>
      void inplace_binary_scalar_recursive(const A& arr,
                                           void* current_data,
                                           const ScalarT scalar_val,
                                           ssize_t axis,
                                           Op op) {
        // Assume both sides have same shape for this function
        ssize_t dim = arr.shape()[axis];
        bool is_last_axis = (axis == static_cast<ssize_t>(arr.ndim()) - 1);

        for (ssize_t i = 0; i < dim; ++i) {
          void* next_ptr = arr.advance(current_data, axis, i);

          if (is_last_axis) {
            op(reinterpret_cast<std::uint8_t*>(next_ptr), scalar_val);
          } else {
            inplace_binary_scalar_recursive<T>(arr,
                                               next_ptr,
                                               scalar_val,
                                               axis + 1,
                                               op);
          }
        }
      }
    } // namespace impl

    // --- Binary non-broadcast operations (same shape) --- //

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void add_recursive(const Left& left, const Right& right, ResultType& result) {
      using AccumT = typename op_traits<T>::sum_type;

      auto add_op_internal = [](const std::uint8_t* lhs,
                                const std::uint8_t* rhs,
                                AccumT* output) {
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

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void sub_recursive(const Left& left, const Right& right, ResultType& result) {
      using DiffT = typename op_traits<T>::diff_type;

      auto sub_op_internal = [](const std::uint8_t* lhs,
                                const std::uint8_t* rhs,
                                DiffT* output) {
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

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void mul_recursive(const Left& left, const Right& right, ResultType& result) {
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

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void truediv_recursive(const Left& left, const Right& right, ResultType& result) {
      using ResultT = typename op_traits<T>::truediv_type;

      auto truediv_op_internal = [](const std::uint8_t* lhs,
                                    const std::uint8_t* rhs,
                                    ResultT* output) {

        const T lhs_val = *reinterpret_cast<const T*>(lhs);
        const T rhs_val = *reinterpret_cast<const T*>(rhs);

        if (rhs_val == T(0)) {
          bool is_finite { false };
          // custom_types.hh has an overload of isfinite
          using std::isfinite;
          if constexpr (requires { lhs_val.real(); }) {
            is_finite = isfinite(lhs_val.real()) && isfinite(lhs_val.imag());
          } else {
            is_finite = isfinite(lhs_val);
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
    }

    // --- Inplace binary operations --- //

    template <typename T, ArrayLike Left, ArrayLike Right>
    void inplace_add_recursive(Left& left, const Right& right) {
      auto add_op_internal = [](std::uint8_t* lhs, const std::uint8_t* rhs) {
        *reinterpret_cast<T*>(lhs) += *reinterpret_cast<const T*>(rhs);
      };

      impl::inplace_binary_reduce_recursive<T>(left,
                                               right,
                                               left.data(),
                                               right.data(),
                                               0,
                                               add_op_internal);
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    void inplace_sub_recursive(Left& left, const Right& right) {
      auto sub_op_internal = [](std::uint8_t* lhs, const std::uint8_t* rhs) {
        *reinterpret_cast<T*>(lhs) -= *reinterpret_cast<const T*>(rhs);
      };

      impl::inplace_binary_reduce_recursive<T>(left,
                                               right,
                                               left.data(),
                                               right.data(),
                                               0,
                                               sub_op_internal);
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    void inplace_mul_recursive(Left& left, const Right& right) {
      auto mul_op_internal = [](std::uint8_t* lhs, const std::uint8_t* rhs) {
        if constexpr (std::is_same_v<T, bool>) {
          *reinterpret_cast<T*>(lhs) &= *reinterpret_cast<const T*>(rhs);
        } else {
          *reinterpret_cast<T*>(lhs) *= *reinterpret_cast<const T*>(rhs);
        }
      };

      impl::inplace_binary_reduce_recursive<T>(left,
                                               right,
                                               left.data(),
                                               right.data(),
                                               0,
                                               mul_op_internal);
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    void inplace_truediv_recursive(Left& left, const Right& right) {
      using ResultT = typename op_traits<T>::truediv_type;
      auto div_op_internal = [](std::uint8_t* lhs, const std::uint8_t* rhs) {
        *reinterpret_cast<T*>(lhs) =
          static_cast<T>(static_cast<ResultT>(*reinterpret_cast<T*>(lhs)) /
                         static_cast<ResultT>(*reinterpret_cast<const T*>(rhs)));
      };

      impl::inplace_binary_reduce_recursive<T>(left,
                                               right,
                                               left.data(),
                                               right.data(),
                                               0,
                                               div_op_internal);
    }

    // --- Binary operations with a scalar broadcast --- //

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    void add_scalar_recursive(const Left& left, const Scalar& right, ResultType& result) {
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
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    void sub_scalar_recursive(const Left& left, const Scalar& right, ResultType& result) {
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
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    void mul_scalar_recursive(const Left& left, const Scalar& right, ResultType& result) {
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
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    void truediv_scalar_recursive(const Left& left,
                                  const Scalar& right,
                                  ResultType& result) {
      using ResultT = typename op_traits<T>::truediv_type;

      auto truediv_op_internal = [](const std::uint8_t* lhs, const T rhs, ResultT* output) {
        const T lhs_val = *reinterpret_cast<const T*>(lhs);

        if (rhs == T(0)) {
          bool is_finite { false };
          // custom_types.hh has an overload of isfinite
          using std::isfinite;
          if constexpr (requires { lhs_val.real(); }) {
            is_finite = isfinite(lhs_val.real()) && isfinite(lhs_val.imag());
          } else {
            is_finite = isfinite(lhs_val);
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
    }

    // --- Inplace binary operations with a scalar broadcast --- //

    template <typename T, ArrayLike Left>
    void inplace_add_scalar_recursive(Left& left, const Scalar& right) {
      auto add_op_internal = [](uint8_t* lhs, const T rhs) {
        *reinterpret_cast<T*>(lhs) += rhs;
      };
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };
      T scalar_val = std::visit(cast_op, right);
      impl::inplace_binary_scalar_recursive<T>(left,
                                               left.data(),
                                               scalar_val,
                                               0,
                                               add_op_internal);
    }

    template <typename T, ArrayLike Left>
    void inplace_sub_scalar_recursive(Left& left, const Scalar& right) {
      auto sub_op_internal = [](uint8_t* lhs, const T rhs) {
        *reinterpret_cast<T*>(lhs) -= rhs;
      };
      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);
      impl::inplace_binary_scalar_recursive<T>(left,
                                               left.data(),
                                               scalar_val,
                                               0,
                                               sub_op_internal);
    }

    template <typename T, ArrayLike Left>
    void inplace_mul_scalar_recursive(Left& left, const Scalar& right) {
      auto mul_op_internal = [](uint8_t* lhs, const T rhs) {
        if constexpr (std::is_same_v<T, bool>) {
          *reinterpret_cast<bool*>(lhs) = *reinterpret_cast<bool*>(lhs) && rhs;
        } else {
          *reinterpret_cast<T*>(lhs) *= rhs;
        }
      };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);
      impl::inplace_binary_scalar_recursive<T>(left,
                                               left.data(),
                                               scalar_val,
                                               0,
                                               mul_op_internal);
    }

    template <typename T, ArrayLike Left>
    void inplace_truediv_scalar_recursive(Left& left, const Scalar& right) {
      using ResultT = typename op_traits<T>::truediv_type;

      auto div_op_internal = [](uint8_t* lhs, const T rhs) {
        *reinterpret_cast<T*>(lhs) =
          static_cast<T>(static_cast<ResultT>(*reinterpret_cast<T*>(lhs)) /
                         static_cast<ResultT>(rhs));
      };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);
      impl::inplace_binary_scalar_recursive<T>(left,
                                               left.data(),
                                               scalar_val,
                                               0,
                                               div_op_internal);
    }

    // --- Logical and boolean operators --- //

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void equal_recursive(const Left& left, const Right& right, ResultType& result) {
      auto equal_op_internal = [](const std::uint8_t* lhs,
                                  const std::uint8_t* rhs,
                                  bool* output) {
        *output = *reinterpret_cast<const T*>(lhs) == *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      bool* result_ptr = reinterpret_cast<bool*>(result.data());
      impl::binary_reduce_recursive<T, decltype(equal_op_internal), bool>(left,
                                                                          right,
                                                                          left.data(),
                                                                          right.data(),
                                                                          starting_axis,
                                                                          equal_op_internal,
                                                                          result_ptr);
    }


    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void not_equal_recursive(const Left& left, const Right& right, ResultType& result) {
      auto not_equal_op_internal = [](const std::uint8_t* lhs,
                                      const std::uint8_t* rhs,
                                      bool* output) {
        *output = *reinterpret_cast<const T*>(lhs) != *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      bool* result_ptr = reinterpret_cast<bool*>(result.data());
      impl::binary_reduce_recursive<T, decltype(not_equal_op_internal), bool>(left,
                                                                              right,
                                                                              left.data(),
                                                                              right.data(),
                                                                              starting_axis,
                                                                              not_equal_op_internal,
                                                                              result_ptr);
    }

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void less_than_recursive(const Left& left,
                             const Right& right,
                             ResultType& result) {
      auto less_than_op_internal = [](const std::uint8_t* lhs,
                                      const std::uint8_t* rhs,
                                      bool* output) {
        *output = *reinterpret_cast<const T*>(lhs) < *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      bool* result_ptr = reinterpret_cast<bool*>(result.data());
      impl::binary_reduce_recursive<T, decltype(less_than_op_internal), bool>(left,
                                                                              right,
                                                                              left.data(),
                                                                              right.data(),
                                                                              starting_axis,
                                                                              less_than_op_internal,
                                                                              result_ptr);
    }

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void less_equal_than_recursive(const Left& left,
                                   const Right& right,
                                   ResultType& result) {
      auto less_eq_op_internal = [](const std::uint8_t* lhs,
                                    const std::uint8_t* rhs,
                                    bool* output) {
        *output = *reinterpret_cast<const T*>(lhs) <= *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      bool* result_ptr = reinterpret_cast<bool*>(result.data());
      impl::binary_reduce_recursive<T, decltype(less_eq_op_internal), bool>(left,
                                                                            right,
                                                                            left.data(),
                                                                            right.data(),
                                                                            starting_axis,
                                                                            less_eq_op_internal,
                                                                            result_ptr);
    }


    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void greater_than_recursive(const Left& left,
                                const Right& right,
                                ResultType& result) {
      auto greater_than_op_internal = [](const std::uint8_t* lhs,
                                         const std::uint8_t* rhs,
                                         bool* output) {

        *output = *reinterpret_cast<const T*>(lhs) > *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      bool* result_ptr = reinterpret_cast<bool*>(result.data());
      impl::binary_reduce_recursive<T, decltype(greater_than_op_internal), bool>(left,
                                                                                 right,
                                                                                 left.data(),
                                                                                 right.data(),
                                                                                 starting_axis,
                                                                                 greater_than_op_internal,
                                                                                 result_ptr);
    }

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void greater_equal_than_recursive(const Left& left,
                                      const Right& right,
                                      ResultType& result) {
      auto greater_eq_op_internal = [](const std::uint8_t* lhs,
                                         const std::uint8_t* rhs,
                                         bool* output) {

        *output = *reinterpret_cast<const T*>(lhs) >= *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      bool* result_ptr = reinterpret_cast<bool*>(result.data());
      impl::binary_reduce_recursive<T, decltype(greater_eq_op_internal), bool>(left,
                                                                               right,
                                                                               left.data(),
                                                                               right.data(),
                                                                               starting_axis,
                                                                               greater_eq_op_internal,
                                                                               result_ptr);
    }

    // logical ops

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void logical_and_recursive(const Left& left,
                               const Right& right,
                               ResultType& result) {
      auto logical_and_op_internal = [](const std::uint8_t* lhs,
                                      const std::uint8_t* rhs,
                                      bool* output) {
        *output = *reinterpret_cast<const T*>(lhs) && *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      bool* result_ptr = reinterpret_cast<bool*>(result.data());
      impl::binary_reduce_recursive<T, decltype(logical_and_op_internal), bool>(left,
                                                                                right,
                                                                                left.data(),
                                                                                right.data(),
                                                                                starting_axis,
                                                                                logical_and_op_internal,
                                                                                result_ptr);
    }

    template <typename T, ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
    void logical_or_recursive(const Left& left,
                              const Right& right,
                              ResultType& result) {
      auto logical_or_op_internal = [](const std::uint8_t* lhs,
                                       const std::uint8_t* rhs,
                                       bool* output) {

        *output = *reinterpret_cast<const T*>(lhs) || *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      bool* result_ptr = reinterpret_cast<bool*>(result.data());
      impl::binary_reduce_recursive<T, decltype(logical_or_op_internal), bool>(left,
                                                                               right,
                                                                               left.data(),
                                                                               right.data(),
                                                                               starting_axis,
                                                                               logical_or_op_internal,
                                                                               result_ptr);
    }

    template <typename T, ArrayLike Array, OwningArrayLike ResultType>
    void logical_not_recursive(const Array& arr, ResultType& result) {
      auto logical_not_op_internal = [](const std::uint8_t* ptr, bool* output) {
        if constexpr (requires { !static_cast<bool>(*reinterpret_cast<const T*>(ptr)); }) {
          *output = !static_cast<bool>(*reinterpret_cast<const T*>(ptr));
        } else {
          *output = false;
        }
      };

      ssize_t starting_axis { 0 };
      bool* result_ptr = reinterpret_cast<bool*>(result.data());
      impl::unary_reduce_recursive<T, decltype(logical_not_op_internal), bool>(arr,
                                                                               arr.data(),
                                                                               starting_axis,
                                                                               logical_not_op_internal,
                                                                               result_ptr);
    }

    // --- Inplace logical operators --- //

    template <typename T, ArrayLike Left, ArrayLike Right>
    void inplace_logical_and_recursive(Left& left, const Right& right) {
      auto logical_and_op_internal = [](std::uint8_t* lhs, const std::uint8_t* rhs) {
        *reinterpret_cast<bool*>(lhs) =
          *reinterpret_cast<T*>(lhs) && *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      impl::inplace_binary_reduce_recursive<T, decltype(logical_and_op_internal), bool>(left,
                                                                                        right,
                                                                                        left.data(),
                                                                                        right.data(),
                                                                                        starting_axis,
                                                                                        logical_and_op_internal);
    }

    template <typename T, ArrayLike Left, ArrayLike Right>
    void inplace_logical_or_recursive(Left& left,
                                      const Right& right) {
      auto logical_or_op_internal = [](std::uint8_t* lhs, const std::uint8_t* rhs) {

        *reinterpret_cast<bool*>(lhs) =
          *reinterpret_cast<T*>(lhs) || *reinterpret_cast<const T*>(rhs);
      };

      ssize_t starting_axis { 0 };
      impl::inplace_binary_reduce_recursive<T, decltype(logical_or_op_internal), bool>(left,
                                                                                       right,
                                                                                       left.data(),
                                                                                       right.data(),
                                                                                       starting_axis,
                                                                                       logical_or_op_internal);
    }

    // --- Logical operators with scalar broadcast --- //

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    void logical_and_scalar_recursive(const Left& left,
                                      const Scalar& right,
                                      ResultType& result) {
      auto and_op = [](const std::uint8_t* a, const T b, bool* out) {
        if constexpr (requires { static_cast<bool>(*reinterpret_cast<const T*>(a)); }) {
          *out = static_cast<bool>(*reinterpret_cast<const T*>(a)) && static_cast<bool>(b);
        } else {
          *out = false;
        }
      };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      bool* res_ptr = reinterpret_cast<bool*>(result.data());

      ssize_t starting_axis { 0 };
      impl::binary_scalar_recursive<T>(left,
                                       left.data(),
                                       scalar_val,
                                       starting_axis,
                                       and_op,
                                       res_ptr);
    }

    template <typename T, ArrayLike Left, OwningArrayLike ResultType>
    void logical_or_scalar_recursive(const Left& left,
                                     const Scalar& right,
                                     ResultType& result) {
      auto or_op = [](const std::uint8_t* a, const T b, bool* out) {
        if constexpr (requires { static_cast<bool>(*reinterpret_cast<const T*>(a)); }) {
          *out = static_cast<bool>(*reinterpret_cast<const T*>(a)) || static_cast<bool>(b);
        } else {
          *out = false;
        }
      };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      bool* res_ptr = reinterpret_cast<bool*>(result.data());
      ssize_t starting_axis { 0 };
      impl::binary_scalar_recursive<T>(left,
                                       left.data(),
                                       scalar_val,
                                       starting_axis,
                                       or_op,
                                       res_ptr);
    }

    // --- Inplace logical operators with scalar broadcast --- //

    template <typename T, ArrayLike Left>
    void inplace_logical_and_scalar_recursive(Left& left, const Scalar& right) {
      auto and_op = [](std::uint8_t* a, const T b) {
        *reinterpret_cast<bool*>(a) =
          static_cast<bool>(*reinterpret_cast<const T*>(a)) && static_cast<bool>(b);
      };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      ssize_t starting_axis { 0 };
      impl::inplace_binary_scalar_recursive<T>(left,
                                               left.data(),
                                               scalar_val,
                                               starting_axis,
                                               and_op);
    }

    template <typename T, ArrayLike Left>
    void inplace_logical_or_scalar_recursive(Left& left, const Scalar& right) {
      auto or_op = [](std::uint8_t* a, const T b) {
        *reinterpret_cast<bool*>(a) =
            static_cast<bool>(*reinterpret_cast<const T*>(a)) || static_cast<bool>(b);
      };

      auto cast_op = [](auto&& arg) {
        using FromT = std::decay_t<decltype(arg)>;
        return op_traits<FromT>::template cast<T>(arg);
      };

      T scalar_val = std::visit(cast_op, right);

      ssize_t starting_axis { 0 };
      impl::inplace_binary_scalar_recursive<T>(left,
                                               left.data(),
                                               scalar_val,
                                               starting_axis,
                                               or_op);
    }
  } // namespace host
} // namespace ncarray

#endif // NCARRAY_HOST_ELEMENTWISE_HH
