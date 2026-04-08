/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ARRAY_OPERATIONS_HH
#define NCARRAY_ARRAY_OPERATIONS_HH

#include "ncarray/array_impl.hh"
#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"
#include "ncarray/engines.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cmath>
#include <complex>
#include <concepts>
#include <limits>
#include <stdexcept>

namespace ncarray {
  class index_error : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
  };

  // Unary reduction operations
  // --------------------------

  template <ArrayLike A>
  inline Scalar sum(const A& arr) {
    auto sum_operation = [&]<typename T>() -> Scalar {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_sum<T>(arr);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_sum<T>(arr);
      }
    };

    return dispatch(arr.dtype(), sum_operation);
  }

  template <ArrayLike A>
  inline Scalar mean(const A& arr) {
    auto sum_operation = [&]<typename T>() -> Scalar {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_mean<T>(arr);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_mean<T>(arr);
      }
    };

    return dispatch(arr.dtype(), sum_operation);
  }

  template <ArrayLike A>
  inline Scalar var(const A& arr, ssize_t ddof) {
    auto var_operation = [&] <typename T> () -> Scalar {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_var<T>(arr, ddof);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_var<T>(arr, ddof);
      }
    };

    return dispatch(arr.dtype(), var_operation);
  }

  template <ArrayLike A>
  inline Scalar std(const A& arr, ssize_t ddof) {
    auto std_operation = [&]<typename T>() -> Scalar {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_std<T>(arr, ddof);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_std<T>(arr, ddof);
      }
    };

    return dispatch(arr.dtype(), std_operation);
  }

  template <ArrayLike A>
  inline Scalar max(const A& arr) {
    auto max_operation = [&]<typename T>() -> Scalar {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_max<T>(arr);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_max<T>(arr);
      }
    };

    return dispatch(arr.dtype(), max_operation);
  }

  template <ArrayLike A>
  inline Scalar argmax(const A& arr) {
    auto argmax_operation = [&] <typename T> () -> Scalar {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_argmax<T>(arr);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_argmax<T>(arr);
      }
    };

    return dispatch(arr.dtype(), argmax_operation);
  }


  template <ArrayLike A>
  inline Scalar min(const A& arr) {
    auto min_operation = [&]<typename T>() -> Scalar {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_min<T>(arr);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_min<T>(arr);
      }
    };

    return dispatch(arr.dtype(), min_operation);
  }

  template <ArrayLike A>
  inline Scalar argmin(const A& arr) {
    auto argmin_operation = [&] <typename T> () -> Scalar {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_argmin<T>(arr);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_argmin<T>(arr);
      }
    };

    return dispatch(arr.dtype(), argmin_operation);
  }

  template <ArrayLike A>
  inline Scalar all(const A& arr) {
    auto all_operation = [&]<typename T>() -> Scalar {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_all<T>(arr);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_all<T>(arr);
      }
    };

    return dispatch(arr.dtype(), all_operation);
  }

  template <ArrayLike A>
  inline Scalar any(const A& arr) {
    auto any_operation = [&]<typename T>() -> Scalar {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        return GPUEngine::execute_any<T>(arr);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        return HostEngine::execute_any<T>(arr);
      }
    };

    return dispatch(arr.dtype(), any_operation);
  }

  template <class L, class S>
  inline Scalar ArrayImpl<L, S>::sum() const {
    return ncarray::sum(*this);
  }

  // max and argmax
  template <class L, class S>
  inline Scalar ArrayImpl<L, S>::max() const {
    return ncarray::max(*this);
  }
  template <class L, class S>
  inline Scalar ArrayImpl<L, S>::argmax() const {
    return ncarray::argmax(*this);
  }

  // min and argmin
  template <class L, class S>
  inline Scalar ArrayImpl<L, S>::min() const {
    return ncarray::min(*this);
  }
  template <class L, class S>
  inline Scalar ArrayImpl<L, S>::argmin() const {
    return ncarray::argmin(*this);
  }

  // mean, variance and standard deviation
  template <class L, class S>
  inline Scalar ArrayImpl<L, S>::mean() const {
    return ncarray::mean(*this);
  }
  template <class L, class S>
  inline Scalar ArrayImpl<L, S>::var(ssize_t ddof) const {
    return ncarray::var(*this, ddof);
  }
  template <class L, class S>
  inline Scalar ArrayImpl<L, S>::std(ssize_t ddof) const {
    return ncarray::std(*this, ddof);
  }

  // logical reduction ops - all and any
  template <class L, class S>
  inline Scalar ArrayImpl<L, S>::all() const {
    return ncarray::all(*this);
  }

  template <class L, class S>
  inline Scalar ArrayImpl<L, S>::any() const {
    return ncarray::any(*this);
  }

  // --- Scattering and Keyed Reduction Operations --- //
  template <ArrayLike Dest, ArrayLike Index, ArrayLike Src>
  inline void scatter_add(Dest& dest, const Index& indices, const Src& src) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto dest_view = dest.view();
    auto indices_view = indices.view();
    auto src_view = src.view();
    auto indices_dtype = indices.dtype();
    if (indices_dtype != DType::int64 ||
        indices_dtype != DType::int32 ||
        indices_dtype != DType::uint64 ||
        indices_dtype != DType::uint32) {
      return;
    }

    auto visit_dest = [&] <typename DestT> () {
      auto visit_indices = [&] <typename IndexT> () {
        if constexpr (std::is_same_v<typename std::decay_t<Dest>::MemType, DevTag>) {
#ifdef __CUDACC__
          GPUEngine::execute_scatter_add<DestT, IndexT>(dest_view,
                                                        indices_view,
                                                        src_view);
#else
          throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
        } else {
          HostEngine::execute_scatter_add<DestT, IndexT>(dest_view,
                                                         indices_view,
                                                         src_view);
        }
      };

      dispatch_integers(indices.dtype(), visit_indices);
    };

    dispatch(dest.dtype(), visit_dest);
  }

  template <class L, class S>
  template <ArrayLike Index, ArrayLike Src>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::scatter_add(const Index& indices,
                                                       const Src& src) {
    auto indices_dtype = indices.dtype();
    if (indices_dtype != DType::int64  ||
        indices_dtype != DType::int32  ||
        indices_dtype != DType::uint64 ||
        indices_dtype != DType::uint32) {
      return *this;
    }
    ncarray::scatter_add(*this, indices, src);
    return *this;
  }

  template <ArrayLike Keys, ArrayLike Vals, OwningArrayLike Result>
  inline auto reduce_by_key(const Keys& keys, const Vals& vals) {
    Scalar max_key { keys.max() };

    auto visit_op = [&] <typename KeyT> () {
      return op_traits<KeyT>::template cast<ssize_t>(std::get<KeyT>(max_key)) + 1;
    };
    ssize_t n_bins { dispatch(keys.dtype(), visit_op) };

    Result result(1, &n_bins, vals.dtype());

    auto reduce_op = [&] <typename ValT> () {
      result.fill(ValT { 0 });
    };
    dispatch(vals.dtype(), reduce_op);

    result.scatter_add(keys, vals);

    return result;
  }

  template <class L, class S>
  template <ArrayLike Keys>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::reduce_by_key(const Keys& keys) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::reduce_by_key<Keys, ViewType, OwnerType>(keys, this->view());
  }

  // Binary non-broadcast operations (same shape)
  // --------------------------------------------
  // TODO: In the future, may make ResultType ArrayLike (instead of Owning)
  // but this requires supporting user-provided buffer to put result in
  // TODO: Handle different shape, types and so on for left/right. Not dealt with atm

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto add(const Left& left, const Right& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
      using AccumT = typename op_traits<T>::sum_type;
      return dtype_traits<AccumT>::value;
    });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();
    auto result_view = result.view();

    auto add_operation = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_add<T>(left_view, right_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_add<T>(left_view, right_view, result_view);
      }
    };

    dispatch(left.dtype(), add_operation);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto sub(const Left& left, const Right& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
      using DiffT = typename op_traits<T>::diff_type;
      return dtype_traits<DiffT>::value;
    });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();
    auto result_view = result.view();

    auto sub_operation = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_sub<T>(left_view, right_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_sub<T>(left_view, right_view, result_view);
      }
    };

    dispatch(left.dtype(), sub_operation);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto mul(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), left.dtype());

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();
    auto result_view = result.view();

    auto mul_operation = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_mul<T>(left_view, right_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_mul<T>(left_view, right_view, result_view);
      }
    };

    dispatch(left.dtype(), mul_operation);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto truediv(const Left& left, const Right& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
      using ResultT = typename op_traits<T>::truediv_type;
      return dtype_traits<ResultT>::value;
    });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();
    auto result_view = result.view();

    auto truediv_operation = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_truediv<T>(left_view, right_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_truediv<T>(left_view, right_view, result_view);
      }
    };

    dispatch(left.dtype(), truediv_operation);
    return result;
  }

  // --- Inplace binary operations --- //

  template <ArrayLike Left, ArrayLike Right>
  inline void inplace_add(Left& left, const Right& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();

    auto add_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_inplace_add<T>(left_view, right_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_inplace_add<T>(left_view, right_view);
      }
    };
    dispatch(left.dtype(), add_op);
  }

  template <ArrayLike Left, ArrayLike Right>
  inline void inplace_sub(Left& left, const Right& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();

    auto sub_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_inplace_sub<T>(left_view, right_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_inplace_sub<T>(left_view, right_view);
      }
    };
    dispatch(left.dtype(), sub_op);
  }

  template <ArrayLike Left, ArrayLike Right>
  inline void inplace_mul(Left& left, const Right& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();

    auto mul_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_inplace_mul<T>(left_view, right_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_inplace_mul<T>(left_view, right_view);
      }
    };
    dispatch(left.dtype(), mul_op);
  }

  template <ArrayLike Left, ArrayLike Right>
  inline void inplace_truediv(Left& left, const Right& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();

    auto div_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_inplace_truediv<T>(left_view, right_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_inplace_truediv<T>(left_view, right_view);
      }
    };

    dispatch(left.dtype(), div_op);
  }

  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::add(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::add<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator+(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::add<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::iadd(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_add<ThisType, OtherType>(*this, other);
    return *this;
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator+=(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_add<ThisType, OtherType>(*this, other);
    return *this;
  }

  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::sub(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::sub<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator-(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::sub<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::isub(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_sub<ThisType, OtherType>(*this, other);
    return *this;
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator-=(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_sub<ThisType, OtherType>(*this, other);
    return *this;
  }

  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::mul(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::mul<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator*(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::mul<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::imul(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_mul<ThisType, OtherType>(*this, other);
    return *this;
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator*=(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_mul<ThisType, OtherType>(*this, other);
    return *this;
  }

  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::truediv(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::truediv<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator/(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;
    return ncarray::truediv<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::itruediv(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_truediv<ThisType, OtherType>(*this, other);

    return *this;
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator/=(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_truediv<ThisType, OtherType>(*this, other);

    return *this;
  }

  // --- Binary operations with a scalar broadcast --- //

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline ResultType add_scalar(const Left& left, const Scalar& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
      using AccumT = typename op_traits<T>::sum_type;

      return dtype_traits<AccumT>::value;
    });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto add_operation = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_add_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_add_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), add_operation);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline ResultType sub_scalar(const Left& left, const Scalar& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
      using DiffT = typename op_traits<T>::diff_type;
      return dtype_traits<DiffT>::value;
    });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto sub_operation = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_sub_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_sub_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), sub_operation);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline auto mul_scalar(const Left& left, const Scalar& right) {
    ResultType result(left.ndim(), left.shape(), left.dtype());

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto mul_operation = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_mul_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_mul_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), mul_operation);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline auto truediv_scalar(const Left& left, const Scalar& right) {
    DType result_dtype = dispatch(left.dtype(), []<typename T>() {
      using ResultT = typename op_traits<T>::truediv_type;
      return dtype_traits<ResultT>::value;
    });

    ResultType result(left.ndim(), left.shape(), result_dtype);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto truediv_operation = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_truediv_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_truediv_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), truediv_operation);
    return result;
  }

  // --- Inplace binary operations with a scalar broadcast --- //

  template <ArrayLike Left>
  inline void inplace_add_scalar(Left& left, const Scalar& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();

    auto add_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_inplace_add_scalar<T>(left_view, right);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_inplace_add_scalar<T>(left_view, right);
      }
    };
    dispatch(left.dtype(), add_op);
  }

  template <ArrayLike Left>
  inline void inplace_sub_scalar(Left& left, const Scalar& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();

    auto sub_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_inplace_sub_scalar<T>(left_view, right);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_inplace_sub_scalar<T>(left_view, right);
      }
    };
    dispatch(left.dtype(), sub_op);
  }
  template <ArrayLike Left>
  inline void inplace_mul_scalar(Left& left, const Scalar& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();

    auto mul_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_inplace_mul_scalar<T>(left_view, right);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_inplace_mul_scalar<T>(left_view, right);
      }
    };
    dispatch(left.dtype(), mul_op);
  }

  template <ArrayLike Left>
  inline void inplace_truediv_scalar(Left& left, const Scalar& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();

    auto truediv_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_inplace_truediv_scalar<T>(left_view, right);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_inplace_truediv_scalar<T>(left_view, right);
      }
    };
    dispatch(left.dtype(), truediv_op);
  }

  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::add(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::add_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator+(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::add_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::iadd(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_add_scalar<ThisType>(*this, other);
    return *this;
  }
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator+=(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_add_scalar<ThisType>(*this, other);
    return *this;
  }

  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::sub(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::sub_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator-(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::sub_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::isub(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_sub_scalar<ThisType>(*this, other);
    return *this;
  }
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator-=(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_sub_scalar<ThisType>(*this, other);
    return *this;
  }

  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::mul(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::mul_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator*(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::mul_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::imul(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_mul_scalar<ThisType>(*this, other);
    return *this;
  }
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator*=(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_mul_scalar<ThisType>(*this, other);
    return *this;
  }

  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::truediv(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::truediv_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator/(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::truediv_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::itruediv(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_truediv_scalar<ThisType>(*this, other);
    return *this;
  }
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator/=(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;
    ncarray::inplace_truediv_scalar<ThisType>(*this, other);
    return *this;
  }

  // --- Logical and boolean operators --- //

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto is_equal(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();
    auto result_view = result.view();

    auto equal_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_equal<T>(left_view, right_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_equal<T>(left_view, right_view, result_view);
      }
    };

    dispatch(left.dtype(), equal_op);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto is_not_equal(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();
    auto result_view = result.view();

    auto not_equal_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_not_equal<T>(left_view, right_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_not_equal<T>(left_view, right_view, result_view);
      }
    };

    dispatch(left.dtype(), not_equal_op);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto is_less_than(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();
    auto result_view = result.view();

    auto less_than_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_less_than<T>(left_view, right_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_less_than<T>(left_view, right_view, result_view);
      }
    };

    dispatch(left.dtype(), less_than_op);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto is_less_equal_than(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();
    auto result_view = result.view();

    auto less_than_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_less_equal_than<T>(left_view, right_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_less_equal_than<T>(left_view, right_view, result_view);
      }
    };

    dispatch(left.dtype(), less_than_op);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto is_greater_than(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();
    auto result_view = result.view();

    auto greater_than_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_greater_than<T>(left_view, right_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_greater_than<T>(left_view, right_view, result_view);
      }
    };

    dispatch(left.dtype(), greater_than_op);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto is_greater_equal_than(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    auto greater_than_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_greater_equal_than<T>(left, right, result);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_greater_equal_than<T>(left, right, result);
      }
    };

    dispatch(left.dtype(), greater_than_op);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto logical_and(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();
    auto result_view = result.view();

    auto logical_and_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_logical_and<T>(left_view, right_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_logical_and<T>(left_view, right_view, result_view);
      }
    };

    dispatch(left.dtype(), logical_and_op);
    return result;
  }

  template <ArrayLike Left, ArrayLike Right, OwningArrayLike ResultType>
  inline auto logical_or(const Left& left, const Right& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();
    auto result_view = result.view();

    auto logical_or_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_logical_or<T>(left_view, right_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_logical_or<T>(left_view, right_view, result_view);
      }
    };

    dispatch(left.dtype(), logical_or_op);
    return result;
  }

  template <ArrayLike Array, OwningArrayLike ResultType>
  inline auto logical_not(const Array& arr) {
    ResultType result(arr.ndim(), arr.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto arr_view = arr.view();
    auto result_view = result.view();

    auto logical_not_op = [&] <typename T> () {
      if constexpr (std::is_same_v<typename std::decay_t<Array>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_logical_not<T>(arr_view, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_logical_not<T>(arr_view, result_view);
      }
    };

    dispatch(arr.dtype(), logical_not_op);
    return result;
  }

  // equal
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_equal(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_equal<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator==(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_equal<ViewType, OtherType, OwnerType>(*this, other);
  }

  // not equal
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_not_equal(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_not_equal<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator!=(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_not_equal<ViewType, OtherType, OwnerType>(*this, other);
  }

  // less than
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_less_than(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_less_than<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator<(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_less_than<ViewType, OtherType, OwnerType>(*this, other);
  }

  // less equal than
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_less_equal_than(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_less_equal_than<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator<=(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_less_equal_than<ViewType, OtherType, OwnerType>(*this, other);
  }

  // greater than
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_greater_than(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_greater_than<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator>(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_greater_than<ViewType, OtherType, OwnerType>(*this, other);
  }

  // greater equal than
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_greater_equal_than(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_greater_equal_than<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator>=(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_greater_equal_than<ViewType, OtherType, OwnerType>(*this, other);
  }

  // logical and
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::logical_and(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::logical_and<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator&&(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::logical_and<ViewType, OtherType, OwnerType>(*this, other);
  }

  // logical or
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::logical_or(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::logical_or<ViewType, OtherType, OwnerType>(*this, other);
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator||(const OtherType& other) const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::logical_or<ViewType, OtherType, OwnerType>(*this, other);
  }

  // logical not
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::logical_not() const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::logical_not<ViewType, OwnerType>(*this);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::operator!() const {
    using ViewType = typename ArrayImpl<L, S>::ViewType;
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::logical_not<ViewType, OwnerType>(*this);
  }

  // --- Comparison operators with scalar broadcast --- //

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline auto is_equal_scalar(const Left& left, const Scalar& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto equal_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_equal_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_equal_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), equal_op);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline auto is_not_equal_scalar(const Left& left, const Scalar& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto not_equal_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_not_equal_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_not_equal_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), not_equal_op);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline auto is_less_than_scalar(const Left& left, const Scalar& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto less_than_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_less_than_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_less_than_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), less_than_op);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline auto is_less_equal_than_scalar(const Left& left, const Scalar& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto less_than_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_less_equal_than_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_less_equal_than_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), less_than_op);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline auto is_greater_than_scalar(const Left& left, const Scalar& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto greater_than_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_greater_than_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_greater_than_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), greater_than_op);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline auto is_greater_equal_than_scalar(const Left& left, const Scalar& right) {
    ResultType result(left.ndim(), left.shape(), DType::bool_);

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto greater_than_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_greater_equal_than_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_greater_equal_than_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), greater_than_op);
    return result;
  }

  // equal
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_equal(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_equal_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator==(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_equal_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  // not equal
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_not_equal(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_not_equal_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator!=(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_not_equal_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  // less than
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_less_than(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_less_than_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator<(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_less_than_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  // less equal than
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_less_equal_than(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_less_equal_than_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator<=(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_less_equal_than_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  // greater than
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_greater_than(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_greater_than_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator>(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_greater_than_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  // greater equal than
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::is_greater_equal_than(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_greater_equal_than_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator>=(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::is_greater_equal_than_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  // --- Inplace logical operators --- //

  template <ArrayLike Left, ArrayLike Right>
  inline auto inplace_logical_and(Left& left, const Right& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();

    auto logical_and_op = [&]<typename T>() {
      if constexpr (std::is_same_v<T, bool>) {
        if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
          GPUEngine::execute_inplace_logical_and<T>(left_view, right_view);
#else
          throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
        } else {
          HostEngine::execute_inplace_logical_and<T>(left_view, right_view);
        }
      }
    };

    dispatch(left.dtype(), logical_and_op);
  }

  template <ArrayLike Left, ArrayLike Right>
  inline auto inplace_logical_or(Left& left, const Right& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto right_view = right.view();

    auto logical_or_op = [&]<typename T>() {
      if constexpr (std::is_same_v<T, bool>) {
        if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
          GPUEngine::execute_inplace_logical_or<T>(left_view, right_view);
#else
          throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
        } else {
          HostEngine::execute_inplace_logical_or<T>(left_view, right_view);
        }
      }
    };

    dispatch(left.dtype(), logical_or_op);
  }

  // inplace logical and
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::ilogical_and(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_logical_and<ThisType, OtherType>(*this, other);
    return *this;
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator&=(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_logical_and<ThisType, OtherType>(*this, other);
    return *this;
  }

  // inplace logical or
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::ilogical_or(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_logical_or<ThisType, OtherType>(*this, other);
    return *this;
  }
  template <class L, class S>
  template <ArrayLike OtherType>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator|=(const OtherType& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_logical_or<ThisType, OtherType>(*this, other);
    return *this;
  }

  // --- Logical operators with scalar broadcast --- //

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline auto logical_and_scalar(const Left& left, const Scalar& right) {
    ResultType result(left.ndim(), left.shape(), left.dtype());

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto and_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_logical_and_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_logical_and_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), and_op);
    return result;
  }

  template <ArrayLike Left, OwningArrayLike ResultType>
  inline auto logical_or_scalar(const Left& left, const Scalar& right) {
    ResultType result(left.ndim(), left.shape(), left.dtype());

    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();
    auto result_view = result.view();

    auto or_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_logical_or_scalar<T>(left_view, right, result_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_logical_or_scalar<T>(left_view, right, result_view);
      }
    };

    dispatch(left.dtype(), or_op);
    return result;
  }

  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::logical_and(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::logical_and_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator&&(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::logical_and_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::logical_or(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::logical_or_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }
  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::operator||(const Scalar& other) const {
    using OwnerType = typename ArrayImpl<L, S>::OwnerType;

    return ncarray::logical_or_scalar<ArrayImpl<L, S>, OwnerType>(*this, other);
  }

  // --- Inplace logical operators with scalar broadcast --- //
  // logical and
  template <ArrayLike Left>
  inline void inplace_logical_and_scalar(Left& left, const Scalar& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();

    auto and_op = [&]<typename T>() {
      if constexpr (std::is_same_v<T, bool>) {
        if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
          GPUEngine::execute_inplace_logical_and_scalar<T>(left_view, right);
#else
          throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
        } else {
          HostEngine::execute_inplace_logical_and_scalar<T>(left_view, right);
        }
      }
    };
    dispatch(left.dtype(), and_op);
  }

  // logical or
  template <ArrayLike Left>
  inline void inplace_logical_or_scalar(Left& left, const Scalar& right) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto left_view = left.view();

    auto or_op = [&]<typename T>() {
      if constexpr (std::is_same_v<T, bool>) {
        if constexpr (std::is_same_v<typename std::decay_t<Left>::MemType, DevTag>) {
#ifdef __CUDACC__
          GPUEngine::execute_inplace_logical_or_scalar<T>(left_view, right);
#else
          throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
        } else {
          HostEngine::execute_inplace_logical_or_scalar<T>(left_view, right);
        }
      }
    };
    dispatch(left.dtype(), or_op);
  }

  // inplace logical and
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::ilogical_and(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_logical_and_scalar<ThisType>(*this, other);
    return *this;
  }
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator&=(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_logical_and_scalar<ThisType>(*this, other);
    return *this;
  }

  // inplace logical or
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::ilogical_or(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_logical_or_scalar<ThisType>(*this, other);
    return *this;
  }
  template <class L, class S>
  inline ArrayImpl<L, S>& ArrayImpl<L, S>::operator|=(const Scalar& other) {
    using ThisType = ArrayImpl<L, S>;

    ncarray::inplace_logical_or_scalar<ThisType>(*this, other);
    return *this;
  }

  // --- Iterators --- //

  template <typename L, typename S>
  inline typename ArrayImpl<L, S>::Iterator ArrayImpl<L, S>::begin() {
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
  inline typename ArrayImpl<L, S>::Iterator ArrayImpl<L, S>::end() {
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
  inline typename ArrayImpl<L, S>::ConstIterator ArrayImpl<L, S>::begin() const {
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
  inline typename ArrayImpl<L, S>::ConstIterator ArrayImpl<L, S>::end() const {
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
  inline void fill(A& arr, Scalar val) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto arr_view = arr.view();

    auto fill_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_fill<T>(arr_view, val);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++");
#endif
      } else {
        HostEngine::execute_fill<T>(arr_view, val);
      }
    };

    dispatch(arr.dtype(), fill_op);
  }

  template <ArrayLike A, typename OutputType>
  inline void copy_into(const A& arr, OutputType* dest) {
    // Convert to views to decrease binary size with fewer template instantiations
    auto arr_view = arr.view();

    auto copy_op = [&]<typename T>() {
      if constexpr (std::is_same_v<typename std::decay_t<A>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_copy_into<T>(arr_view, dest);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_copy_into<T>(arr_view, dest);
      }
    };

    dispatch(arr.dtype(), copy_op);
  }

  template <ArrayLike Dest, ArrayLike Src>
  inline void assign(Dest& dest, const Src& src) {
    // Only deal with identical shapes for now
    if (dest.ndim() != src.ndim()) {
      throw type_error("Shapes must match for assignment");
    }
    for (ssize_t i = 0; i < dest.ndim(); ++i) {
      if (dest.shape(i) != src.shape(i)) {
        throw type_error("Shapes must match for assignment");
      }
    }

    // Convert to views to decrease binary size with fewer template instantiations
    auto dest_view = dest.view();
    auto src_view = src.view();

    auto assign_op = [&] <typename DestT> () {
      // Dispatch dest type - the engines do the double dispatch to get the src type
      if constexpr (std::is_same_v<typename std::decay_t<Dest>::MemType, DevTag>) {
#ifdef __CUDACC__
        GPUEngine::execute_assign<DestT>(dest_view, src_view);
#else
        throw std::runtime_error("Fatal: tried to compile a CUDA kernel as C++.");
#endif
      } else {
        HostEngine::execute_assign<DestT>(dest_view, src_view);
      }
    };
    dispatch(dest.dtype(), assign_op);
  }

  template <class L, class S>
  inline void ArrayImpl<L, S>::copy_into(void* dest_buffer) const {
    auto copy_op = [&]<typename T>() {
      T* dest_ptr = reinterpret_cast<T*>(dest_buffer);
      ncarray::copy_into(*this, dest_ptr);
    };

    dispatch(this->m_dtype, copy_op);
  }

  template <class L, class S>
  template <typename OutT>
  inline void ArrayImpl<L, S>::copy_into_astype(OutT* dest_buffer) const {
    ncarray::copy_into(*this, dest_buffer);
  }

  template <class L, class S>
  inline typename ArrayImpl<L, S>::OwnerType ArrayImpl<L, S>::to_contiguous() const {
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
  inline typename ArrayImpl<L, S>::OwnerType
  ArrayImpl<L, S>::astype(DType& dtype_out) const {
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
  inline void ArrayImpl<L, S>::assign(const ArrayLike auto& arr) {
    if (this->m_read_only) {
      throw type_error("Cannot modify a read-only view!");
    }

    ncarray::assign(*this, arr);
  }

  template <class L, class S>
  inline void ArrayImpl<L, S>::fill(Scalar val) {
    if (this->m_read_only) {
      throw type_error("Cannot modify a read-only view!");
    }

    ncarray::fill(*this, val);
  }

} // namespace ncarray

#endif // NCARRAY_ARRAY_OPERATIONS_HH
