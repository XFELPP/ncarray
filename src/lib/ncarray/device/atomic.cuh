/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_DEVICE_ATOMIC_CUH
#define NCARRAY_DEVICE_ATOMIC_CUH

#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"

#include <concepts>
#include <type_traits>

namespace ncarray {
  namespace device {
    namespace impl {
      template <typename T>
      concept SupportsAtomicAdd =
        std::is_same_v<std::remove_cvref_t<T>, int>                 ||
        std::is_same_v<std::remove_cvref_t<T>, unsigned>            ||
        std::is_same_v<std::remove_cvref_t<T>, unsigned long long>  ||
        std::is_same_v<std::remove_cvref_t<T>, float>               ||
        std::is_same_v<std::remove_cvref_t<T>, double>              ||
        //std::is_same_v<std::remove_cvref_t<T>, __half>              || // cc 7+
        //std::is_same_v<std::remove_cvref_t<T>, __half2>             || // cc 7+
        //std::is_same_v<std::remove_cvref_t<T>, __nv_bfloat162>      || // cc 8+
        //std::is_same_v<std::remove_cvref_t<T>, __nv_bfloat16>       || // cc 8+
        std::is_same_v<std::remove_cvref_t<T>, float2>              ||   // cc 9+
        std::is_same_v<std::remove_cvref_t<T>, float4>;                  // cc 9+

      template <typename T>
      concept SupportsAtomicMinMax =
        std::is_same_v<std::remove_cvref_t<T>, int>                 ||
        std::is_same_v<std::remove_cvref_t<T>, unsigned>            ||
        std::is_same_v<std::remove_cvref_t<T>, unsigned long long>;

      template <bool NativeOp, typename T, class AtomicFn, class CASFn>
      __device__ inline void nca_atomic_apply(T* addr,
                                              T val,
                                              AtomicFn&& atomic,
                                              CASFn&& fn) {
        if constexpr (NativeOp) {
          atomic(addr, val);
        } else if constexpr (Vector2DType<T>) {
          // TODO: Support casting Float4 etc to float4 (since this supports atomics)
          nca_atomic_apply<NativeOp>(&addr->x, val.x, atomic, fn);
          nca_atomic_apply<NativeOp>(&addr->y, val.y, atomic, fn);
          if constexpr (Vector3DType<T>) {
            nca_atomic_apply<NativeOp>(&addr->z, val.z, atomic, fn);
          }
          if constexpr (Vector4DType<T>) {
            nca_atomic_apply<NativeOp>(&addr->w, val.w, atomic, fn);
          }
        } else if constexpr (requires { addr->real(); }) {
          using ScalarT = typename std::decay_t<decltype(addr->real())>;
          auto* raw = reinterpret_cast<ScalarT*>(addr);
          nca_atomic_apply<NativeOp>(&raw[0], val.real(), atomic, fn);
          nca_atomic_apply<NativeOp>(&raw[1], val.imag(), atomic, fn);
        } else if constexpr (sizeof(T) <= 8) {
          using IntT = typename std::conditional_t<sizeof(T) == 4, int, unsigned long long>;
          IntT* addr_as_int = reinterpret_cast<IntT*>(addr);
          IntT assumed;
          IntT old = *addr_as_int;
          do {
            assumed = old;

            T old_val = *reinterpret_cast<const T*>(&assumed);
            T new_val = fn(old_val, val);
            if (new_val == old_val) {
              break;
            }
            old = atomicCAS(addr_as_int, assumed, *reinterpret_cast<const IntT*>(&new_val));
          } while (assumed != old);
        }
      }
    } // namespace impl

    /**
     * Generalized atomicAdd for all supported ncarray types T.
     * When possible, falls back onto the provide atomic primitives. Otherwise, it
     * will attempt component-wise atomics for vector types and complex numbers. The
     * final fallback is a CAS loop via int casts for anything else up to 8 bytes.
     */
    template <typename T>
    __device__ inline void nca_atomic_add(T* addr, T val) {
      auto nativeOp = [] __device__ (auto* a, auto v) {
        return atomicAdd(a, v);
      };

      using ValT = typename op_traits<T>::value_type;
      auto CASOp = [] __device__ (ValT old_v, ValT v) {
        return old_v + v;
      };

      impl::nca_atomic_apply<impl::SupportsAtomicAdd<T>>(addr,
                                                         val,
                                                         nativeOp,
                                                         CASOp);
    }

    /**
     * Generalized atomicMax for all supported ncarray types T.
     * When possible, falls back onto the provide atomic primitives. Otherwise, it
     * will attempt component-wise atomics for vector types and complex numbers. The
     * final fallback is a CAS loop via int casts for anything else up to 8 bytes.
     */
    template <typename T>
    __device__ inline void nca_atomic_max(T* addr, T val) {
      auto nativeOp = [] __device__ (auto* a, auto v) {
        return atomicMax(a, v);
      };

      using ValT = typename op_traits<T>::value_type;
      auto CASOp = [] __device__ (ValT old_v, ValT v) {
        return old_v > v ? old_v : v;
      };

      impl::nca_atomic_apply<impl::SupportsAtomicMinMax<T>>(addr,
                                                            val,
                                                            nativeOp,
                                                            CASOp);
    }

    /**
     * Generalized atomicMin for all supported ncarray types T.
     * When possible, falls back onto the provide atomic primitives. Otherwise, it
     * will attempt component-wise atomics for vector types and complex numbers. The
     * final fallback is a CAS loop via int casts for anything else up to 8 bytes.
     */
    template <typename T>
    __device__ inline void nca_atomic_min(T* addr, T val) {
      auto nativeOp = [] __device__(auto* a, auto v) {
        return atomicMin(a, v);
      };

      using ValT = typename op_traits<T>::value_type;
      auto CASOp = [] __device__(ValT old_v, ValT v) {
        return old_v < v ? old_v : v;
      };

      impl::nca_atomic_apply<impl::SupportsAtomicMinMax<T>>(addr,
                                                            val,
                                                            nativeOp,
                                                            CASOp);
    }
  } // namespace device
} // namespace ncarray

#endif // NCARRAY_DEVICE_ATOMIC_CUH
