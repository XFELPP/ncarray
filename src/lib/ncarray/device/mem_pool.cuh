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
      if (m_pool == nullptr) {
        m_pool = new CircularDevicePool<T, Capacity>();
      }
      return *m_pool;
    }

    CircularDevicePool(CircularDevicePool&) = delete;
    void operator=(const CircularDevicePool&) = delete;

    NCA_H ~CircularDevicePool() {
#ifdef NCA_HAS_CUDA
      if (m_h_data != nullptr) {
        CHECK_CUDA_ERROR(cudaFreeHost(m_h_data));
        m_h_data = nullptr;
      }
#endif
    }

    NCA_H MemEntry next() {
      std::size_t idx { m_idx };

      m_idx = (m_idx + 1) % Capacity;

      if (m_idx == 0) {
#ifdef NCA_HAS_CUDA
        // Synchronize on wrap to prevent overwriting memory from out-standing ops
        cudaDeviceSynchronize();
#endif
      }

      return { &m_h_data[idx], &m_d_data[idx] };
    }

    NCA_H MemEntry get_block(std::size_t items, std::size_t alignment = 16) {
      std::size_t aligned_items = (items + (alignment - 1)) & ~(alignment - 1);
      if (aligned_items > Capacity) {
        return { nullptr, nullptr };
      }

      if (m_idx + aligned_items >= Capacity) {
        // Wrap back to 0
        m_idx = 0;
#ifdef NCA_HAS_CUDA
        // Synchronize on wrap to prevent overwriting memory from out-standing ops
        cudaDeviceSynchronize();
#endif
      }

      std::size_t idx { m_idx };
      m_idx = (m_idx + aligned_items) % Capacity;

      return { &m_h_data[idx], &m_d_data[idx] };
    }

  private:
    NCA_H CircularDevicePool() {
#ifdef NCA_HAS_CUDA
      CHECK_CUDA_ERROR(cudaHostAlloc(&m_h_data,
                                     Capacity * sizeof(T),
                                     cudaHostAllocMapped));
      CHECK_CUDA_ERROR(cudaHostGetDevicePointer(&m_d_data, m_h_data, 0));
#endif
    }

    T* m_h_data { nullptr };
    T* m_d_data { nullptr };
    std::size_t m_idx { 0 };

    inline static CircularDevicePool* m_pool { nullptr };
  };
} // namespace ncarray

#endif // NCARRAY_DEVICE_POOL_CUH
