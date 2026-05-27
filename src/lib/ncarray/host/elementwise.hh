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
#include "ncarray/op_traits.hh"

namespace ncarray {
  namespace host {
    namespace impl {
      /**
       * Scatter src values into dest using indices. Src and indices must be the
       * of the same shape.
       */
      template <
        typename DestT,
        typename IndexT,
        typename SrcT,
        class Op,
        ArrayLike Dest,
        ArrayLike Index,
        ArrayLike Src
      >
      void scatter_reduce_recursive(Dest& dest,
                                    const Index& indices,
                                    const Src& src,
                                    void* dest_ptr,
                                    const void* idx_ptr,
                                    const void* src_ptr,
                                    ssize_t axis,
                                    Op op) {
        if (axis == static_cast<ssize_t>(src.ndim())) {
          // Base case
          IndexT idx = *reinterpret_cast<const IndexT*>(idx_ptr);

          if (op_traits<IndexT>::ge(idx, 0) &&
              op_traits<IndexT>::less(idx, op_traits<ssize_t>::template cast<IndexT>(dest.size()))) {
            void* addr =
              const_cast<void*>(dest.advance(dest_ptr,
                                             0,
                                             op_traits<IndexT>::template cast<ssize_t>(idx)));

            op(reinterpret_cast<DestT*>(addr), reinterpret_cast<const SrcT*>(src_ptr));
          }
          return;
        }

        ssize_t dim = src.shape()[axis];
        for (ssize_t i = 0; i < dim; ++i) {
          const void* next_idx = indices.advance(idx_ptr, axis, i);
          const void* next_src = src.advance(src_ptr, axis, i);
          scatter_reduce_recursive<DestT, IndexT, SrcT>(dest,
                                                        indices,
                                                        src,
                                                        dest_ptr,
                                                        next_idx,
                                                        next_src,
                                                        axis + 1,
                                                        op);
        }
      }

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

    // --- Binary operations --- //

    template <typename DestT, typename CoordsT, ArrayExpression Expr, class Result>
    void execute_expression_recursive(Expr expr,
                                      Result result,
                                      CoordsT& coords,
                                      ssize_t axis) {
      if (axis == coords.size() - 1) {
        const auto limit { result.shape(axis) };

        for (auto i = 0; i < limit; ++i) {
          coords[axis] = i;

          result[coords] = expr.template eval<DestT>(coords);
        }
      } else {
        const auto limit { result.shape(axis) };

        for (auto i = 0; i < limit; ++i) {
          coords[axis] = i;
          execute_expression_recursive<DestT, CoordsT>(expr, result, coords, axis + 1);
        }
      }
    }

    // -- Scatter operations --- //
    template <
      typename DestT,
      typename IndexT,
      typename SrcT,
      ArrayLike Dest,
      ArrayLike Index,
      ArrayLike Src
    >
    void scatter_add_recursive(Dest& dest, const Index& indices, const Src& src) {
      auto add_op_internal = [](DestT* dest_ptr, const SrcT* src_ptr) {
        *dest_ptr += op_traits<SrcT>::template cast<DestT>(*src_ptr);
      };

      ssize_t starting_axis { 0 };
      void* dest_ptr { const_cast<void*>(dest.data()) };
      impl::scatter_reduce_recursive<DestT, IndexT, SrcT>(dest,
                                                          indices,
                                                          src,
                                                          dest_ptr,
                                                          indices.data(),
                                                          src.data(),
                                                          starting_axis,
                                                          add_op_internal);
    }
  } // namespace host
} // namespace ncarray

#endif // NCARRAY_HOST_ELEMENTWISE_HH
