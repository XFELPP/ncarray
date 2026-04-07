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

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace ncarray {
  namespace device {
    namespace impl {
      // Setup a small table of locks to reduce collisions
      __device__ int _nca_atomic_locks[1024];

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

      /**
       * A generic foundation for building atomic operations.
       *
       * A pair of functions can be provided, a native atomic operation as well as
       * a lambda for fallback in a CAS loop should the datatype not support the native
       * operation. The first template parameter indicates whether the type supports
       * the native operation (for faster compile-time branching).
       *
       * Operations in order of preference are:
       * 1. Use the native operation if possible (e.g. atomicAdd)
       * 2. For vector types (float3, etc.) use the atomic on each component.
       * 3. For types <= 8 bytes use a CAS loop of either int, or unsigned long long.
       * 4. Finally, fall back to a locking approach with a striped mutex. This is
       *    slower, but supports arbitrarily sized types (whereas native atomics are
       *    only available up to 128 bit).
       *
       * @tparam NativeOp Whether the native operation can be used (for compile-time branching.)
       * @tparam T The datatype.
       * @tparam AtomicFn The type for the native atomic operation.
       * @tparam CASFn The type for the comparison lambda in the CAS loop.
       * @param addr The address to put the final result.
       * @param val The value for this thread.
       * @param atomic The native atomic (in lambda generally). E.g. atomicAdd.
       * @param fn The comparison function. E.g. for add: [](T o, T v) { return o + v; }
       */
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
          using IntT = typename std::conditional_t<sizeof(T) == 4,
                                                   int,
                                                   unsigned long long>;
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
        } else {
          std::uintptr_t addr_v = reinterpret_cast<std::uintptr_t>(addr);
          int& lock = _nca_atomic_locks[(addr_v >> 4) % 1024];

          while (atomicCAS(&lock, 0, 1) != 0); // Acquire spin-lock

          // This is a pessimistic approach with the lock
          // No need for the do-while, but slower because of lock acquisition
          T old_val = *addr;
          T new_val = fn(old_val, val);
          if (new_val != old_val) {
            *addr = new_val;
          }

          atomicExch(&lock, 0); // Release the lock
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
     * Generalized atomic to perform an argmax.
     *
     * There is no native operation for this. Use a CAS loop always.
     */
    template <typename T>
    __device__ inline void nca_atomic_argmax(KeyValPair<ssize_t, T>* addr,
                                             KeyValPair<ssize_t, T> val) {
      using Pair = KeyValPair<ssize_t, T>;

      auto CASOp = [] __device__(Pair old_v, Pair v) {
        if (op_traits<T>::greater(old_v.val, v.val)) {
          return old_v;
        }
        return v;
      };

      impl::nca_atomic_apply<false>(addr, val, /*nativeOp=*/nullptr, CASOp);
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

    /**
     * Generalized atomic to perform an argmin.
     *
     * There is no native operation for this. Use a CAS loop always.
     */
    template <typename T>
    __device__ inline void nca_atomic_argmin(KeyValPair<ssize_t, T>* addr,
                                             KeyValPair<ssize_t, T> val) {
      using Pair = KeyValPair<ssize_t, T>;

      auto CASOp = [] __device__(Pair old_v, Pair v) {
        if (op_traits<T>::less(old_v.val, v.val)) {
          return old_v;
        }
        return v;
      };

      impl::nca_atomic_apply<false>(addr, val, /*nativeOp=*/nullptr, CASOp);
    }

    /**
     * Atomic accumlator merge for variance calculations.
     */
    template <typename T>
    __device__ inline void nca_atomic_accumulator_merge(VarAccumulator<T>* addr,
                                                        VarAccumulator<T> val) {
      auto CASOp = [] __device__ (VarAccumulator<T> old_v, VarAccumulator<T> v) {
        return VarAccumulator<T>::merge(old_v, v);
      };

      impl::nca_atomic_apply<false>(addr, val, /*nativeOp=*/nullptr, CASOp);
    }

    /**
     * Atomic grid-wide logical and operation.
     */
    __device__ inline void nca_atomic_logical_and(bool* addr, bool val) {
      if (!val) {
        // Any false value means we are now false.
        atomicExch(reinterpret_cast<int*>(addr), 0);
      }
    }

    /**
     * Atomic grid-wide logical or operation.
     */
    __device__ inline void nca_atomic_logical_or(bool* addr, bool val) {
      if (val) {
        // Any true value means we are now true.
        atomicExch(reinterpret_cast<int*>(addr), 1);
      }
    }
  } // namespace device
} // namespace ncarray

#endif // NCARRAY_DEVICE_ATOMIC_CUH
