/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/device/atomic.cuh"

#include <cstddef>

namespace ncarray {
  namespace device {
    /**
     * Atomic grid-wide logical and operation.
     */
    __device__ inline void nca_atomic_logical_and(bool* addr, bool val) {
      if (!val) {
        // Any false value means we are now false.
        std::size_t s_bool { reinterpret_cast<std::size_t>(addr) };

        unsigned* aligned_addr = reinterpret_cast<unsigned*>(s_bool & ~3);

        unsigned shift { (static_cast<unsigned>(s_bool) & 3) * 8 };
        unsigned mask { ~(0xFFu << shift) };

        atomicAnd(aligned_addr, mask);
      }
    }

    /**
     * Atomic grid-wide logical or operation.
     */
    __device__ inline void nca_atomic_logical_or(bool* addr, bool val) {
      if (val) {
        // Any true value means we are now true.
        std::size_t s_bool { reinterpret_cast<std::size_t>(addr) };

        unsigned* aligned_addr = reinterpret_cast<unsigned*>(s_bool & ~3);

        unsigned shift { (static_cast<unsigned>(s_bool) & 3) * 8 };
        unsigned mask { 0x01u << shift };

        atomicOr(aligned_addr, mask);
      }
    }
  } // namespace device
} // namespace ncarray