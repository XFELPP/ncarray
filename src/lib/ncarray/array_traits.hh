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

namespace hd_std = cuda::std;

#else
#include <concepts>
#include <vector>

namespace hd_std = std;

#endif

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
  /**
   * Determines an object that has dimensionality and defined shape for each axis.
   */
  template <typename T>
  concept Shaped = requires(const T arr) {
    { arr.ndim() } -> hd_std::convertible_to<ssize_t>;
    { arr.shape() } -> hd_std::convertible_to<const ssize_t*>;
  };

  /**
   * Determines an object that provides strides in bytes per axis.
   */
  template <typename T>
  concept Strided = requires(const T arr) {
    { arr.strides() } -> hd_std::convertible_to<const ssize_t*>;
  };

  /**
   * Determines an object that has exists over some data.
   */
  template <typename T>
  concept HasData = requires(const T arr) {
    { arr.data() } -> hd_std::convertible_to<const void*>;
  };

  /**
   * Determines an object that has a known data type for its elements.
   */
  template <typename T>
  concept HasDType = requires(const T arr) {
    { arr.dtype() } -> hd_std::same_as<ncarray::DType>;
  };

  /**
   * Determines an object as meeting an the necessary requirements to be an array.
   */
  template <typename T>
  concept ArrayLike = Shaped<T> && Strided<T> && HasData<T> && HasDType<T>;

  /**
   * Determines an array that is mutable.
   *
   * @deprecated The array mutability is now accessed via a `read only` bool flag.
   * @todo Update concepts for correct mutability checks.
   */
  template <typename T>
  concept MutableArrayLike = ArrayLike<T> && requires(T arr) {
    { arr.data() } -> hd_std::same_as<void*>;
  };

  /**
   * Determines an array that is a light-weight, non-owning, view of the data.
   */
  template <class T>
  concept ViewArrayLike = ArrayLike<T> &&
    hd_std::is_base_of_v<ViewTag, typename hd_std::remove_cvref_t<T>::StoragePolicy>;

#ifndef __CUDACC_RTC__
  /**
   * Determines an array that owns (i.e., allocated space for) the data it describes.
   *
   * @note This concept is not applicable to device code. It is not usable in device code.
   */
  template <typename T>
  concept OwningArrayLike = ArrayLike<T> && requires(std::vector<ssize_t> shape, DType dtype) {
    T(shape, dtype);
  };
#endif
} // namespace ncarray

#endif // NCARRAY_ARRAY_TRAITS_HH
