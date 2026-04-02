/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_DEVICE_POOL_CUH
#define NCARRAY_DEVICE_POOL_CUH

#ifdef NCA_HAS_CUDA
#include "ncarray/device/utilities.cuh"

#include "cuda_runtime_api.h"
#endif

#include <atomic>
#include <cstddef>

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

#ifndef NCA_H
#ifdef __CUDACC__
#define NCA_H __host__
#else
#define NCA_H
#endif
#endif

namespace ncarray {
  /**
   * A simple circular buffer pool for small device objects.
   * Provides rapid "allocations" via a simple index wrap-around.
   * The pool is intended to provide a loction for retrieving small scalar
   * results, e.g. from a reduction operation such as ncarr.sum().
   *
   * Pinned memory is used - this is not intended for high performance array
   * wide operations.
   */
  template <typename T, std::size_t Capacity = 4096>
  class CircularDevicePool {
  public:
    struct MemEntry {
      T* h_ptr;
      T* d_ptr;
    };
    NCA_H static CircularDevicePool& instance() {
      static CircularDevicePool<T, Capacity> pool;

      return pool;
    }

    NCA_H ~CircularDevicePool() {
#ifdef NCA_HAS_CUDA
      CHECK_CUDA_ERROR(cudaFreeHost(m_h_data));
#endif
    }

    NCA_H MemEntry next() {
      std::size_t idx { m_idx };

      m_idx = (m_idx + 1) % Capacity;
      return { &m_h_data[idx], &m_d_data[idx] };
    }

  protected:
    NCA_H CircularDevicePool() {
#ifdef NCA_HAS_CUDA
      CHECK_CUDA_ERROR(cudaHostAlloc(&m_h_data,
                                     Capacity * sizeof(T),
                                     cudaHostAllocMapped));
      CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_d_data, m_h_data, 0));
#endif
    }

  private:
    T* m_h_data;
    T* m_d_data;
    std::size_t m_idx { 0 };
  };
} // namespace ncarray

#endif // NCARRAY_DEVICE_POOL_CUH
