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

namespace hd_std = cuda::std;

#else

#include <type_traits>

namespace hd_std = std;

#endif

namespace ncarray {
  // --- Array Element Proxy --- //
  /**
   * The element proxy can be returned by arrays during indexing to provide
   * reference based access to the underlying data.
   *
   * This mostly provides improved ergonomics.
   *
   * @note Be aware that the semantics of these proxy objects can at times be subtely
   * different than may immediately be expected from intuition. Proxy references are
   * const in general; however, they may return const or non-const references to the
   * underlying array data elements they proxy. Because of this, at times care must
   * be taken to carefully consider the const/non-const semantics, particular when
   * dealing with r-value references.
   *
   * In general, operator T& style coercion/cast functions do NOT type check.
   * The `get<T>` variants will type check in DEBUG builds using assertions.
   *
   * The assignment operator= will perform a cast. The `set<T>` will type check
   * in DEBUG builds using assertions.
   */
  struct ArrayElementProxy {
    void* m_data;
    DType m_dtype;

    /**
     * Coerce to the respective type if the type is among the standard array DTypes.
     *
     * This function does NOT check the actual datatype of the array.
     *
     * @tparam T The datatype to coerce to. Enabled only for standard DTypes to prevent
     *           issues, particularly with bool coercions.
     * @returns Reference to the array element as the requested type.
     */
    template <typename T, typename = hd_std::enable_if_t<is_in_type_list_v<T, all_supported_types>>>
    NCA_HD inline operator T&() const {
      return *reinterpret_cast<T*>(m_data);
    }

    /**
     * Coerce to the respective type if the type is among the standard array DTypes.
     *
     * This function does NOT check the actual datatype of the array.
     *
     * @tparam T The datatype to coerce to. Enabled only for standard DTypes to prevent
     *           issues, particularly with bool coercions.
     * @returns Const reference to the array element as the requested type.
     */
    template <typename T, typename = hd_std::enable_if_t<is_in_type_list_v<T, all_supported_types>>>
    NCA_HD inline operator const T&() const {
      return *reinterpret_cast<const T*>(m_data);
    }

    /**
     * Assign `val` to the underlying array element referenced.
     *
     * This function will cast `val` as appropriate.
     *
     * @note The dispatch operation only works on base_types; proxies may refer to
     *       extended types though, such as accumulators.
     * @todo Consider whether dispatch support is required for extended types like Key/Val structs.
     *
     * @tparam T The datatype of the value to be assigned.
     * @param val The value to assign to the array element.
     * @returns The ArrayElementProxy.
     */
    template <typename T>
    NCA_HD inline ArrayElementProxy& operator=(const T& val) {
      auto assign_op = [&]<typename ArrayT>() {
        *reinterpret_cast<ArrayT*>(m_data) = op_traits<T>::template cast<ArrayT>(val);
      };

      // 
      // Proxy objects may refer to all_supported_types (like accumulators.)
      // TODO: Consider whether those extended types need to be dispatched as well....
      dispatch(m_dtype, assign_op);
      return *this;
    }

    // --- Type checking versions --- //
    /**
     * Retrieve the underlying array element as a reference of the indicated type.
     *
     * This function will check the types match in DEBUG builds with assertions.
     *
     * @tparam T The datatype to coerce the array element to.
     * @returns Reference to the array element as the requested type.
     */
    template <typename T>
    NCA_HD inline T& get() {
      assert(m_dtype == dtype_traits<T>::value);
      return *reinterpret_cast<T*>(m_data);
    }

    /**
     * Retrieve the underlying array element as a reference of the indicated type.
     *
     * This function will check the types match in DEBUG builds with assertions.
     *
     * @tparam T The datatype to coerce the array element to.
     * @returns Const reference to the array element as the requested type.
     */
    template <typename T>
    NCA_HD inline const T& get() const {
      assert(m_dtype == dtype_traits<T>::value);
      return *reinterpret_cast<const T*>(m_data);
    }

    /**
     * Assign the underlying array data element the value of `val`.
     *
     * This function will check the types match in DEBUG builds with assertions.
     *
     * @tparam T The datatype to coerce the array element to.
     * @returns Reference to the array element as the requested type.
     */
    template <typename T>
    NCA_HD inline void set(const T& val) {
      assert(m_dtype == dtype_traits<T>::value);
      *reinterpret_cast<T*>(m_data) = val;
    }
  };
} // namespace ncarray

#endif // NCARRAY_ARRAY_ELEMENT_PROXY_HH
