/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_HOST_REDUCTIONS_HH
#define NCARRAY_HOST_REDUCTIONS_HH

#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"

namespace ncarray {
  namespace host {
    namespace impl {
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
    } // namespace impl

    // --- Unary reduction operations --- //
    template <typename T, ArrayLike A>
    Scalar sum_recursive(const A& arr) {
      using AccumT = typename op_traits<T>::sum_type;

      auto sum_op_internal = [](const std::uint8_t* data, AccumT* output) {
        *output += static_cast<AccumT>(*reinterpret_cast<const T*>(data));
      };

      ssize_t starting_axis { 0 };
      AccumT result = impl::reduce_recursive<T>(arr,
                                                arr.data(),
                                                starting_axis,
                                                sum_op_internal,
                                                AccumT { 0 });
      return Scalar { result };
    }

    // Mean, variance and standard deviation
    template <typename T, ArrayLike A>
    Scalar mean_recursive(const A& arr) {
      using AccumT = typename op_traits<T>::sum_type;
      using ResultT = typename op_traits<T>::truediv_type;

      auto sum_op_internal = [](const std::uint8_t* data, AccumT* output) {
        *output += static_cast<AccumT>(*reinterpret_cast<const T*>(data));
      };

      ssize_t starting_axis { 0 };
      AccumT result = impl::reduce_recursive<T>(arr,
                                                arr.data(),
                                                starting_axis,
                                                sum_op_internal,
                                                AccumT { 0 });
      ResultT mean = static_cast<ResultT>(result) / static_cast<double>(arr.size());
      return Scalar { mean };
    }

    template <typename T, ArrayLike A>
    Scalar var_recursive(const A& arr, ssize_t ddof = 0) {
      using ResultT = typename op_traits<T>::truediv_type;

      ResultT mean = std::get<ResultT>(mean_recursive<T>(arr));

      // NOTE: This is different than NumPy for complex numbers which would use std;:norm.
      // TODO: Consider whether to use NumPy style norming for complex
      auto variance_op_internal = [&](const std::uint8_t* data, ResultT* output) {
        ResultT val = static_cast<ResultT>(*reinterpret_cast<const T*>(data));
        ResultT diff = val - mean;

        *output += diff * diff;
      };

      ssize_t starting_axis { 0 };
      ResultT sum_squared_diff = impl::reduce_recursive<T>(arr,
                                                           arr.data(),
                                                           starting_axis,
                                                           variance_op_internal,
                                                           ResultT { 0 });

      return Scalar { sum_squared_diff / static_cast<double>(arr.size() - ddof) };
    }

    template <typename T, ArrayLike A>
    Scalar std_recursive(const A& arr, ssize_t ddof = 0) {
      using ResultT = typename op_traits<T>::truediv_type;

      ResultT var = std::get<ResultT>(var_recursive<T>(arr, ddof));

      return Scalar { nca_sqrt(var) }; // custom_types.hh (handles vector types)
    }

    // Max and Argmax
    template <typename T, ArrayLike A>
    Scalar max_recursive(const A& arr) {
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
      return Scalar { result };
    }

    template <typename T, ArrayLike A>
    Scalar argmax_recursive(const A& arr) {
      ssize_t counter { 0 };
      ssize_t lin_argmax_idx { 0 };

      // Output is just discarded here - capture the count by reference
      auto argmax_op_internal = [&](const std::uint8_t* data, T* output) {
        T val = *reinterpret_cast<const T*>(data);
        if (op_traits<T>::greater(val, *output)) {
          *output = val;
          lin_argmax_idx = counter;
        }
        counter++;
      };
      ssize_t starting_axis { 0 };
      impl::reduce_recursive<T>(arr,
                                arr.data(),
                                starting_axis,
                                argmax_op_internal,
                                op_traits<T>::lowest());
      return Scalar { lin_argmax_idx };
    }

    // Min and argmin
    template <typename T, ArrayLike A>
    Scalar min_recursive(const A& arr) {
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
      return Scalar { result };
    }

    template <typename T, ArrayLike A>
    Scalar argmin_recursive(const A& arr) {
      ssize_t counter { 0 };
      ssize_t lin_argmin_idx { 0 };

      // Output is just discarded here - capture the count by reference
      auto argmin_op_internal = [&](const std::uint8_t* data, T* output) {
        T val = *reinterpret_cast<const T*>(data);
        if (op_traits<T>::less(val, *output)) {
          *output = val;
          lin_argmin_idx = counter;
        }
        counter++;
      };

      ssize_t starting_axis { 0 };
      impl::reduce_recursive<T>(arr,
                                arr.data(),
                                starting_axis,
                                argmin_op_internal,
                                op_traits<T>::max());
      return Scalar { lin_argmin_idx };
    }
  } // namespace host
} // namespace ncarray

#endif // NCARRAY_HOST_REDUCTIONS_HH
