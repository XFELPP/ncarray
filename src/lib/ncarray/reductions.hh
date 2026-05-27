/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_REDUCTIONS_HH
#define NCARRAY_REDUCTIONS_HH

#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"
#ifdef __CUDACC__
#include "ncarray/device/atomic.cuh"
#include "ncarray/device/warp.cuh"
#endif
#include "ncarray/device/mem_pool.cuh"
#include "ncarray/device/utilities.cuh"
#include "ncarray/host/elementwise.hh"
#include "ncarray/host/reductions.hh"
#include "ncarray/op_traits.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cstdint>
#include <type_traits>

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

#ifndef NCA_D
#ifdef __CUDACC__
#define NCA_D __device__
#else
#define NCA_D
#endif
#endif
namespace ncarray {

  // --- Reduction Traits ---//

  /**
   * Interface specifier for a valid reduction operation.
   *
   * Traits adhering to this specification can be used for performing axis-aware
   * reductions. In the case that the reduction accumulates into a type that is
   * different from the desired output type, an additional dual_atomic function
   * (not included in this concept) would be required for operation on GPU. See
   * argmax/argmin or variance traits for examples.
   *
   * The identity function returns the identity for the accumulator type.
   *
   * The fill function will, in general, return the identity; however, if the
   * output type is not the same as the accumulator type this may be different.
   * For example, the argmax reduction accumulates key/value pairs (max and index).
   * However, the output is just the index, so fill returns an integer, while identity
   * returns the key/value pair.
   *
   * The transform function converts an index and value into the acummulator type.
   *
   * The reduce function performs the necessary comparisons/reductions on a single
   * pair of elements.
   *
   * The store function converts an accumulator type back to the output type.
   *
   * The atomic function is used for a generalized atomic reduction on GPU.
   * - A dual_atomic function may be needed as well.
   */
  template <class Traits, typename T>
  concept ReductionTraits = requires(ssize_t idx,
                                     T val,
                                     typename Traits::template AccumT<T> a,
                                     typename Traits::template AccumT<T> b,
                                     typename Traits::template AccumT<T> res,
                                     typename Traits::template AccumT<T>* dest,
                                     typename Traits::template AccumT<T> aval) {
    typename Traits::template AccumT<T>;
    { Traits::template identity<T>() };
    { Traits::template fill<T>() };
    { Traits::template transform<T>(idx, val) };
    { Traits::template reduce<T>(a, b) };
    { Traits::template store<T>(res, 0.0) };
    { Traits::template atomic<T>(dest, aval) };
  };

  template <class Traits, typename T>
  concept DualAtomicReduction =
    ReductionTraits<Traits, T> &&
    requires(typename Traits::template AccumT<T>* scratch,
             typename Traits::template AccumT<T> val,
             typename Traits::template OutT<T>* res) {
    { Traits::template dual_atomic<T>(scratch, val, res, 0.0) };
  };

  /**
   * Traits for a summation along axes.
   */
  struct SumTraits {
    template <typename T>
    using AccumT = typename op_traits<T>::sum_type;

    template <typename T>
    using OutT = AccumT<T>;

    template <typename T>
    static NCA_HD inline AccumT<T> identity() { return AccumT<T> { 0 }; }

    template <typename T>
    static NCA_HD inline auto fill() { return AccumT<T> { 0 }; }

    template <typename T>
    static NCA_HD inline AccumT<T> transform(ssize_t idx, T val) {
      return op_traits<T>::template cast<AccumT<T>>(val);
    }

    template <typename T>
    static NCA_HD inline AccumT<T> reduce(AccumT<T> a, AccumT<T> b) { return a + b; }

    template <typename T>
    static NCA_HD inline auto store(AccumT<T> res, double ddof = 0.0) { return res; }

    template <typename T>
    static NCA_D inline void atomic(AccumT<T>* dest, AccumT<T> val) {
#ifdef __CUDACC__
      device::nca_atomic_add(dest, val);
#endif
    }
  };

  /**
   * Traits to find maxima along axes.
   */
  struct MaxTraits {
    template <typename T>
    using AccumT = T;

    template <typename T>
    using OutT = AccumT<T>;

    template <typename T>
    static NCA_HD inline AccumT<T> identity() { return op_traits<T>::lowest(); }

    template <typename T>
    static NCA_HD inline auto fill() { return op_traits<T>::lowest(); }

    template <typename T>
    static NCA_HD inline AccumT<T> transform(ssize_t idx, T val) {
      return val;
    }

    template <typename T>
    static NCA_HD inline T reduce(AccumT<T> a, AccumT<T> b) {
      if (op_traits<T>::greater(a, b)) {
        return a;
      }
      return b;
    }

    template <typename T>
    static NCA_HD inline auto store(AccumT<T> res, double ddof = 0.0) {
      return res;
    }

    template <typename T>
    static NCA_D inline void atomic(AccumT<T>* dest, AccumT<T> val) {
#ifdef __CUDACC__
      device::nca_atomic_max(dest, val);
#endif
    }
  };

  /**
   * Traits to find the indices of maxima along axes.
   */
  struct ArgmaxTraits {
    template <typename T>
    using AccumT = KeyValPair<ssize_t, T>;

    template <typename T>
    using OutT = std::int64_t;

    template <typename T>
    static NCA_HD inline AccumT<T> identity() {
      return { -1, op_traits<T>::lowest() };
    }

    template <typename T>
    static NCA_HD inline auto fill() { return OutT<T> { -1 }; }

    template <typename T>
    static NCA_HD inline AccumT<T> transform(ssize_t idx, T val) {
      return KeyValPair<ssize_t, T>(idx, val);
    }

    template <typename T>
    static NCA_HD inline AccumT<T> reduce(AccumT<T> a, AccumT<T> b) {
      if (op_traits<T>::greater(a.val, b.val)) {
        return a;
      }
      return b;
    }

    template <typename T>
    static NCA_HD inline auto store(AccumT<T> res, double ddof = 0.0) {
      return res.key;
    }

    template <typename T>
    static NCA_D inline void atomic(AccumT<T>* dest, AccumT<T> val) {
#ifdef __CUDACC__
      device::nca_atomic_argmax(dest, val);
#endif
    }

    template <typename OutT, typename SrcT>
    static NCA_D inline void dual_atomic(AccumT<SrcT>* scratch,
                                         AccumT<SrcT> val,
                                         OutT* res,
                                         double ddof = 0.0) {
#ifdef __CUDACC__
      device::impl::nca_dual_atomic_apply(scratch,
                                          val,
                                          res,
                                          ArgmaxTraits::template reduce<SrcT>,
                                          ArgmaxTraits::template store<SrcT>);
#endif
    }
  };

  /**
   * Traits to find minima along axes.
   */
  struct MinTraits {
    template <typename T>
    using AccumT = T;

    template <typename T>
    using OutT = AccumT<T>;

    template <typename T>
    static NCA_HD AccumT<T> identity() {
      return op_traits<T>::max();
    }

    template <typename T>
    static NCA_HD auto fill() { return op_traits<T>::max(); }

    template <typename T>
    static NCA_HD inline AccumT<T> transform(ssize_t idx, T val) {
      return val;
    }

    template <typename T>
    static NCA_HD inline AccumT<T> reduce(AccumT<T> a, AccumT<T> b) {
      if (op_traits<T>::less(a, b)) {
        return a;
      }
      return b;
    }

    template <typename T>
    static NCA_HD inline auto store(AccumT<T> res, double ddof = 0.0) {
      return res;
    }

    template <typename T>
    static NCA_D inline void atomic(AccumT<T>* dest, AccumT<T> val) {
#ifdef __CUDACC__
      device::nca_atomic_min(dest, val);
#endif
    }
  };

  /**
   * Traits to find the indices of minima along axes.
   */
  struct ArgminTraits {
    template <typename T>
    using AccumT = KeyValPair<ssize_t, T>;

    template <typename T>
    using OutT = std::int64_t;

    template <typename T>
    static NCA_HD inline AccumT<T> identity() {
      return { -1, op_traits<T>::max() };
    }

    template <typename T>
    static NCA_HD inline auto fill() { return OutT<T> { -1 }; }

    template <typename T>
    static NCA_HD inline AccumT<T> transform(ssize_t idx, T val) {
      return KeyValPair<ssize_t, T>(idx, val);
    }

    template <typename T>
    static NCA_HD inline AccumT<T> reduce(AccumT<T> a, AccumT<T> b) {
      if (op_traits<T>::less(a.val, b.val)) {
        return a;
      }
      return b;
    }

    template <typename T>
    static NCA_HD inline auto store(AccumT<T> res, double ddof = 0.0) {
      return res.key;
    }

    template <typename T>
    static NCA_D inline void atomic(AccumT<T>* dest, AccumT<T> val) {
#ifdef __CUDACC__
      device::nca_atomic_argmin(dest, val);
#endif
    }

    template <typename OutT, typename SrcT>
    static NCA_D inline void dual_atomic(AccumT<SrcT>* scratch,
                                         AccumT<SrcT> val,
                                         OutT* res,
                                         double ddof = 0.0) {
#ifdef __CUDACC__
      device::impl::nca_dual_atomic_apply(scratch,
                                          val,
                                          res,
                                          ArgminTraits::template reduce<SrcT>,
                                          ArgminTraits::template store<SrcT>);
#endif
    }
  };

  struct MeanTraits {
    template <typename T>
    using AccumT = typename op_traits<T>::sum_type;

    template <typename T>
    using OutT = AccumT<T>;

    template <typename T>
    static NCA_HD inline AccumT<T> identity() {
      return AccumT<T> { 0 };
    }

    template <typename T>
    static NCA_HD inline auto fill() {
      return AccumT<T> { 0 };
    }

    template <typename T>
    static NCA_HD inline AccumT<T> transform(ssize_t idx, T val) {
      return op_traits<T>::template cast<AccumT<T>>(val);
    }

    template <typename T>
    static NCA_HD inline AccumT<T> reduce(AccumT<T> a, AccumT<T> b) {
      return a + b;
    }

    template <typename T>
    static NCA_HD inline auto store(AccumT<T> res, double ddof = 0.0) {
      return res;
    }

    template <typename T>
    static NCA_D inline void atomic(AccumT<T>* dest, AccumT<T> val) {
#ifdef __CUDACC__
      device::nca_atomic_add(dest, val);
#endif
    }
  };

  /**
   * Traits to find the variance along axes.
   */
  struct VarTraits {
    template <typename T>
    using AccumT = VarAccumulator<typename op_traits<T>::truediv_type>;

    template <typename T>
    using OutT = typename op_traits<T>::truediv_type;

    template <typename T>
    static NCA_HD inline AccumT<T> identity() {
      using ResultT = op_traits<T>::truediv_type;

      return AccumT<T> { 0.0, ResultT { 0.0 }, ResultT { 0.0 } };
    }
    template <typename T>
    static NCA_HD inline auto fill() { return 0.0; }

    template <typename T>
    static NCA_HD inline AccumT<T> transform(ssize_t idx, T val) {
      using ResultT = op_traits<T>::truediv_type;

      return AccumT<T>(1.0, static_cast<ResultT>(val), ResultT { 0.0 });
    }

    template <typename T>
    static NCA_HD inline AccumT<T> reduce(AccumT<T> a, AccumT<T> b) {
      return AccumT<T>::merge(a, b);
    }

    template <typename T>
    static NCA_HD inline auto store(AccumT<T> res, double ddof = 0.0) {
      using ResultT = op_traits<T>::truediv_type;
      double n { res.count };
      ResultT var = res.m2 / (n - static_cast<ResultT>(ddof));
      return var;
    }

    template <typename T>
    static NCA_D inline void atomic(AccumT<T>* dest, AccumT<T> val) {
#ifdef __CUDACC__
      using ResultT = op_traits<T>::truediv_type;

      device::nca_atomic_accumulator_merge<ResultT>(dest, val);
#endif
    }

    template <typename OutT, typename SrcT>
    static NCA_D inline void dual_atomic(AccumT<SrcT>* scratch,
                                         AccumT<SrcT> val,
                                         OutT* res,
                                         double ddof = 0.0) {
#ifdef __CUDACC__
      auto ddof_var = [ddof] __device__ (auto v) {
        return VarTraits::template store<SrcT>(v, ddof);
      };
      device::impl::nca_dual_atomic_apply(scratch,
                                          val,
                                          res,
                                          VarTraits::template reduce<SrcT>,
                                          ddof_var);
#endif
    }
  };

  /**
   * Traits to find the standard deviation along axes.
   */
  struct StdTraits {
    template <typename T>
    using AccumT = VarAccumulator<typename op_traits<T>::truediv_type>;

    template <typename T>
    using OutT = typename op_traits<T>::truediv_type;

    template <typename T>
    static NCA_HD inline AccumT<T> identity() {
      using ResultT = op_traits<T>::truediv_type;

      return AccumT<T> { 0.0, ResultT { 0.0 }, ResultT { 0.0 } };
    }

    template <typename T>
    static NCA_HD inline auto fill() { return 0.0; }

    template <typename T>
    static NCA_HD inline AccumT<T> transform(ssize_t idx, T val) {
      using ResultT = op_traits<T>::truediv_type;

      return AccumT<T>(1.0, static_cast<ResultT>(val), ResultT { 0.0 });
    }

    template <typename T>
    static NCA_HD inline AccumT<T> reduce(AccumT<T> a, AccumT<T> b) {
      return AccumT<T>::merge(a, b);
    }

    template <typename T>
    static NCA_HD inline auto store(AccumT<T> res, double ddof = 0.0) {
      using ResultT = op_traits<T>::truediv_type;
      double n { res.count };
      ResultT var = res.m2 / (n - static_cast<ResultT>(ddof));
      return nca_sqrt(var);
    }

    template <typename T>
    static NCA_D inline void atomic(AccumT<T>* dest, AccumT<T> val) {
#ifdef __CUDACC__
      using ResultT = op_traits<T>::truediv_type;

      device::nca_atomic_accumulator_merge<ResultT>(dest, val);
#endif
    }

    template <typename OutT, typename SrcT>
    static NCA_D inline void dual_atomic(AccumT<SrcT>* scratch,
                                         AccumT<SrcT> val,
                                         OutT* res,
                                         double ddof = 0.0) {
#ifdef __CUDACC__
      auto ddof_std = [ddof] __device__ (auto v) {
        return StdTraits::template store<SrcT>(v, ddof);
      };
      device::impl::nca_dual_atomic_apply(scratch,
                                          val,
                                          res,
                                          StdTraits::template reduce<SrcT>,
                                          ddof_std);
#endif
    }
  };

  /**
   * Traits to find minima along axes.
   */
  struct AllTraits {
    template <typename T>
    using AccumT = bool;

    template <typename T>
    using OutT = bool;

    template <typename T>
    static NCA_HD AccumT<T> identity() {
      return true;
    }

    template <typename T>
    static NCA_HD auto fill() { return true; }

    template <typename T>
    static NCA_HD inline AccumT<T> transform(ssize_t idx, T val) {
      return op_traits<T>::template cast<bool>(val);
    }

    template <typename T>
    static NCA_HD inline AccumT<T> reduce(AccumT<T> a, AccumT<T> b) {
      return a && b;
    }

    template <typename T>
    static NCA_HD inline auto store(AccumT<T> res, double ddof = 0.0) {
      return res;
    }

    template <typename T>
    static NCA_D inline void atomic(AccumT<T>* dest, AccumT<T> val) {
#ifdef __CUDACC__
      device::nca_atomic_logical_and(dest, val);
#endif
    }
  };

  /**
   * Traits to find minima along axes.
   */
  struct AnyTraits {
    template <typename T>
    using AccumT = bool;

    template <typename T>
    using OutT = bool;

    template <typename T>
    static NCA_HD AccumT<T> identity() { return false; }

    template <typename T>
    static NCA_HD auto fill() { return false; }

    template <typename T>
    static NCA_HD inline AccumT<T> transform(ssize_t idx, T val) {
      return op_traits<T>::template cast<bool>(val);
    }

    template <typename T>
    static NCA_HD inline AccumT<T> reduce(AccumT<T> a, AccumT<T> b) {
      return a || b;
    }

    template <typename T>
    static NCA_HD inline auto store(AccumT<T> res, double ddof = 0.0) {
      return res;
    }

    template <typename T>
    static NCA_D inline void atomic(AccumT<T>* dest, AccumT<T> val) {
#ifdef __CUDACC__
      device::nca_atomic_logical_or(dest, val);
#endif
    }
  };

  /**
   * @brief A wrapping class for reduction traits.
   *
   * The Reducer templates on the various reduction traits binding the static
   * functions into an instantiable struct. In some cases, it may be preferable
   * to pass an object to perform the reductions, where passing the static
   * functions, or lambdas wrapping them, may be less ideal, or impossible.
   */
  template <typename T, typename Traits>
  struct Reducer {
    using AccumT = typename Traits::template AccumT<T>;
    using OutT = typename Traits::template OutT<T>;

    NCA_HD inline AccumT identity() const {
      return Traits::template identity<T>();
    }

    NCA_HD inline AccumT transform(ssize_t idx, T val) const {
      return Traits::template transform<T>(idx, val);
    }

    NCA_HD inline AccumT reduce(AccumT a, AccumT b) const {
      return Traits::template reduce<T>(a, b);
    }

    NCA_HD inline auto store(AccumT val, double ddof = 0.0) const {
      return Traits::template store<T>(val, ddof);
    }

    NCA_HD inline void atomic(AccumT* dest, AccumT val) {
#ifdef __CUDACC__
      return Traits::template atomic<T>(dest, val);
#endif
    }

    NCA_HD inline void dual_atomic(AccumT* scratch,
                                   AccumT val,
                                   OutT* res,
                                   double ddof = 0.0) {
      if constexpr (DualAtomicReduction<Traits, T>) {
#ifdef __CUDACC__
        return Traits::template dual_atomic<T>(scratch, val, res, ddof);
#endif
      }
    }
  };
} // namespace ncarray

#endif // NCARRAY_REDUCTIONS_HH
