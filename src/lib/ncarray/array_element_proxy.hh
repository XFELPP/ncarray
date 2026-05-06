/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ARRAY_ELEMENT_PROXY_HH
#define NCARRAY_ARRAY_ELEMENT_PROXY_HH

#include "ncarray/dtype.hh"
#include "ncarray/op_traits.hh"

#ifdef __CUDACC_RTC__
#include <cuda/std/type_traits>

using cuda::std::enable_if_t;

#else

#include <type_traits>

using std::enable_if_t;

#endif

namespace ncarray {
  // --- Array Element Proxy --- //
  /**
   * The element proxy can be returned by arrays during indexing to provide
   * reference based access to the underlying data.
   *
   * This mostly provides improved ergonomics.
   *
   * operator T& style functions do NOT type check. If you require type checking
   * an `get<T>` function is provided.
   */
  struct ArrayElementProxy {
    void* m_data;
    DType m_dtype;

    template <typename T, typename = enable_if_t<is_in_type_list_v<T, all_supported_types>>>
    NCA_HD inline operator T&() const {
      return *reinterpret_cast<T*>(m_data);
    }

    template <typename T, typename = enable_if_t<is_in_type_list_v<T, all_supported_types>>>
    NCA_HD inline operator const T&() const {
      return *reinterpret_cast<const T*>(m_data);
    }

    template <typename T>
    NCA_HD inline ArrayElementProxy& operator=(const T& val) {
      auto assign_op = [&]<typename ArrayT>() {
        *reinterpret_cast<ArrayT*>(m_data) = op_traits<T>::template cast<ArrayT>(val);
      };

      // NOTE: The dispatch operation only works on base_types.
      // Proxy objects may refer to all_supported_types (like accumulators.)
      // TODO: Consider whether those extended types need to be dispatched as well....
      dispatch(m_dtype, assign_op);
      return *this;
    }

    // --- Type checking versions --- //
    template <typename T>
    NCA_HD inline T& get() {
      assert(m_dtype == dtype_traits<T>::value);
      return *reinterpret_cast<T*>(m_data);
    }

    template <typename T>
    NCA_HD inline const T& get() const {
      assert(m_dtype == dtype_traits<T>::value);
      return *reinterpret_cast<const T*>(m_data);
    }

    template <typename T>
    NCA_HD inline void set(const T& val) {
      assert(m_dtype == dtype_traits<T>::value);
      *reinterpret_cast<T*>(m_data) = val;
    }
  };
} // namespace ncarray

#endif // NCARRAY_ARRAY_ELEMENT_PROXY_HH
