/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ENGINES_HOSTENGINE_HH
#define NCARRAY_ENGINES_HOSTENGINE_HH

#include "ncarray/array_element_proxy.hh"
#include "ncarray/array_impl.hh"
#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"
#include "ncarray/expression.hh"
#include "ncarray/host/elementwise.hh"
#include "ncarray/host/reductions.hh"
#include "ncarray/indexing.hh"
#include "ncarray/layout.hh"
#include "ncarray/mvnode.hh"
#include "ncarray/op_traits.hh"
#include "ncarray/reductions.hh"

#ifdef NCA_HAS_OPENMP
#include "omp.h"
#endif

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cstdint>
#include <type_traits>

namespace ncarray {
  /**
   * The HostEngine is ultimately responsible for dispatching all operations involving
   * arrays on the host/CPU. This includes binary operations, reductions, unary
   * operations, copies, assignments, fills and so on.
   */
  struct HostEngine {

    // --- Binary Operations --- //

    template <typename DestT, class Expr, OwningArrayLike Result>
    static void execute_binary_expression(const Expr& expr, Result& result) {
      ssize_t size { result.size() };

      bool use_32bit { (size < (1LL << 31)) };

      auto launch_recursive = [&](auto rank) {
        constexpr int NDim = decltype(rank)::value;

        constexpr ssize_t starting_axis { 1 };

        const ssize_t limit { result.shape(0) };

        if (use_32bit) {
          using CoordsT = StaticCoords<NDim, std::uint32_t>;

#ifdef NCA_HAS_OPENMP
          #pragma omp parallel for schedule(static)
#endif
          for (ssize_t i = 0; i < limit; ++i) {
            CoordsT coords;
            coords[0] = i;

            if constexpr (NDim > 1) {
              host::execute_expression_recursive<DestT, CoordsT>(expr,
                                                                 result,
                                                                 coords,
                                                                 starting_axis);
            } else {
              result[coords] = expr.template eval<DestT>(coords);
            }
          }
        } else {
          using CoordsT = StaticCoords<NDim, ssize_t>;

#ifdef NCA_HAS_OPENMP
          #pragma omp parallel for schedule(static)
#endif
          for (ssize_t i = 0; i < limit; ++i) {
            CoordsT coords;
            coords[0] = i;
            if constexpr (NDim > 1) {
              host::execute_expression_recursive<DestT, CoordsT>(expr,
                                                                 result,
                                                                 coords,
                                                                 starting_axis);
            } else {
              result[coords] = expr.template eval<DestT>(coords);
            }
          }
        }
      };

      switch (result.ndim()) {
      case 1:  launch_recursive(std::integral_constant<int, 1>  {});  break;
      case 2:  launch_recursive(std::integral_constant<int, 2>  {});  break;
      case 3:  launch_recursive(std::integral_constant<int, 3>  {});  break;
      case 4:  launch_recursive(std::integral_constant<int, 4>  {});  break;
      case 5:  launch_recursive(std::integral_constant<int, 5>  {});  break;
      case 6:  launch_recursive(std::integral_constant<int, 6>  {});  break;
      case 7:  launch_recursive(std::integral_constant<int, 7>  {});  break;
      case 8:  launch_recursive(std::integral_constant<int, 8>  {});  break;
      case 9:  launch_recursive(std::integral_constant<int, 9>  {});  break;
      case 10: launch_recursive(std::integral_constant<int, 10> {});  break;
      }
    }

    // --- Axis-Aware Reductions --- //

    template <
      typename T,
      ReductionTraits<T> Traits,
      ArrayExpression Source,
      ArrayLike Result
    >
    static void execute_reduce_axes(const Source& src,
                                    const ReductionParams& params,
                                    Result& res) {
      using AccumT = typename Traits::AccumT<T>;

      res.fill(Scalar(Traits::template fill<T>()));

      ssize_t input_size { src.size() };

      for (ssize_t i = 0; i < input_size; ++i) {
        ssize_t j { 0 };
        ssize_t temp_i { i };

        // Map input index 'i' to output index 'j'
        for (int dim = params.ndim - 1; dim >= 0; --dim) {
          ssize_t coord = temp_i % params.shape[dim];
          temp_i /= params.shape[dim];
          j += coord * params.strides[dim];
        }
        T src_item;
        if constexpr (Expression<Source>) {
          src_item = src.template eval<T>(i);
        } else {
          T& arr_item = static_index(src, i);
          src_item = arr_item;
        }
        //T& arr_item = arr[i];
        //AccumT& res_item = res[j];
        auto res_proxy = static_index(res, j);
        AccumT& res_item = res_proxy;
        AccumT accum_res = Traits::template reduce<T>(res_item,
                                                      Traits::template transform<T>(i, src_item));
        res_proxy = Traits::template store<T>(accum_res);
      }
    }

    // --- Full Reductions (To Scalar) --- //

    template <typename T, ArrayLike A>
    static Scalar execute_sum(const A& arr) {
      return host::sum_recursive<T>(arr);
    }

    template <typename T, ArrayLike A>
    static Scalar execute_mean(const A& arr) {
      return host::mean_recursive<T>(arr);
    }
    template <typename T, ArrayLike A>
    static Scalar execute_var(const A& arr, ssize_t ddof) {
      return host::var_recursive<T>(arr);
    }
    template <typename T, ArrayLike A>
    static Scalar execute_std(const A& arr, ssize_t ddof) {
      return host::std_recursive<T>(arr);
    }

    template <typename T, ArrayLike A>
    static Scalar execute_max(const A& arr) {
      return host::max_recursive<T>(arr);
    }
    template <typename T, ArrayLike A>
    static Scalar execute_argmax(const A& arr) {
      return host::argmax_recursive<T>(arr);
    }

    template <typename T, ArrayLike A>
    static Scalar execute_min(const A& arr) {
      return host::min_recursive<T>(arr);
    }
    template <typename T, ArrayLike A>
    static Scalar execute_argmin(const A& arr) {
      return host::argmin_recursive<T>(arr);
    }

    // Logical reductions - all and any
    template <typename T, ArrayLike A>
    static Scalar execute_all(const A& arr) {
      return host::all_recursive<T>(arr);
    }

    template <typename T, ArrayLike A>
    static Scalar execute_any(const A& arr) {
      return host::any_recursive<T>(arr);
    }

    // --- Copy and Modification --- //

    template <typename T, ArrayLike Left>
    static void execute_fill(Left& left, const Scalar& val) {
      ssize_t starting_axis { 0 };
      auto fill_op_internal = [](auto&& arg) -> T {
        using FromT = std::decay_t<decltype(arg)>;
        return ncarray::op_traits<FromT>::template cast<T>(arg);
      };
      T target_val = std::visit(fill_op_internal, val);
      host::impl::fill_recursive<T>(left, left.data(), starting_axis, target_val);
    }

    template <typename T, ArrayLike Left, typename OutputType>
    static void execute_copy_into(Left& left, OutputType*& dest) {
      ssize_t starting_axis { 0 };
      host::impl::copy_into_recursive<T, OutputType>(left,
                                                     left.data(),
                                                     starting_axis,
                                                     dest);
    }

    template <typename DestT, ArrayLike Dest, ArrayLike Src>
    static void execute_assign(Dest& dest, const Src& src) {
      auto assign_op_internal = [&]<typename SrcT>() {
        ssize_t starting_axis { 0 };
        host::impl::assign_recursive<DestT, SrcT>(dest,
                                                  src,
                                                  dest.data(),
                                                  src.data(),
                                                  starting_axis);
      };

      dispatch(src.dtype(), assign_op_internal);
    }
  };
} // namespace ncarray

#endif // NCARRAY_ENGINES_HOSTENGINE_HH
