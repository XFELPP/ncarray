/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ARRAY_TRAITS_HH
#define NCARRAY_ARRAY_TRAITS_HH

#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"
#include "ncarray/storage.hh"

#ifdef __CUDACC_RTC__
typedef long long ssize_t;
#else
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif
#endif

#ifdef __CUDACC_RTC__
#include <cuda/std/concepts>

using cuda::std::convertible_to;
using cuda::std::is_base_of_v;
using cuda::std::is_convertible_v;
using cuda::std::remove_cvref_t;
using cuda::std::same_as;
#else
#include <concepts>
#include <vector>

using std::convertible_to;
using std::is_base_of_v;
using std::is_convertible_v;
using std::remove_cvref_t;
using std::same_as;
#endif

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
  namespace impl {
    template <typename T = void> struct default_owner;
  } // namespace impl
  /**
   * The following concepts enforce the interface for the generic algorithms.
   */
  template <typename T>
  concept Shaped = requires(const T arr) {
    { arr.ndim() } -> convertible_to<ssize_t>;
    { arr.shape() } -> convertible_to<const ssize_t*>;
  };

  template <typename T>
  concept Strided = requires(const T arr) {
    { arr.strides() } -> convertible_to<const ssize_t*>;
  };

  template <typename T>
  concept HasData = requires(const T arr) {
    { arr.data() } -> convertible_to<const void*>;
  };

  template <typename T>
  concept HasDType = requires(const T arr) {
    { arr.dtype() } -> same_as<ncarray::DType>;
  };

  template <typename T>
  concept ArrayLike = Shaped<T> && Strided<T> && HasData<T> && HasDType<T>;

  // Mutable/writable arrays can get data that is not just const void*
  template <typename T>
  concept MutableArrayLike = ArrayLike<T> && requires(T arr) {
    { arr.data() } -> same_as<void*>;
  };

  template <class T>
  concept ViewArrayLike = ArrayLike<T> &&
    is_base_of_v<ViewTag, typename remove_cvref_t<T>::StoragePolicy>;

#ifndef __CUDACC_RTC__
  // ArrayLikes that own the data should be constructable from just the shape and type
  // This indicates they can control data buffer
  template <typename T>
  concept OwningArrayLike = ArrayLike<T> && requires(std::vector<ssize_t> shape, DType dtype) {
    T(shape, dtype);
  };
#endif
} // namespace ncarray

#endif // NCARRAY_ARRAY_TRAITS_HH
