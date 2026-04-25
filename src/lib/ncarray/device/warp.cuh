/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_DEVICE_WARP_CUH
#define NCARRAY_DEVICE_WARP_CUH

namespace ncarray {
  template <typename T>
  concept SupportsWarpShfl =
    std::is_same_v<std::remove_cvref_t<T>, int>                 ||
    std::is_same_v<std::remove_cvref_t<T>, long>                ||
    std::is_same_v<std::remove_cvref_t<T>, long long>           ||
    std::is_same_v<std::remove_cvref_t<T>, unsigned>            ||
    std::is_same_v<std::remove_cvref_t<T>, unsigned long>       ||
    std::is_same_v<std::remove_cvref_t<T>, unsigned long long>  ||
    std::is_same_v<std::remove_cvref_t<T>, float>               ||
    std::is_same_v<std::remove_cvref_t<T>, double>;
    //std::is_same_v<std::remove_cvref_t<T>, __half>              || // cc 7+
    //std::is_same_v<std::remove_cvref_t<T>, __half2>             || // cc 7+
    //std::is_same_v<std::remove_cvref_t<T>, __nv_bfloat162>      || // cc 8+
    //std::is_same_v<std::remove_cvref_t<T>, __nv_bfloat16>       || // cc 8+


  template <typename T>
  __device__ inline T nca_shfl_down(unsigned int mask, T val, unsigned offset) {
    if constexpr (sizeof(T) == 8) {
        unsigned long long temp = __shfl_down_sync(mask,
                                                   reinterpret_cast<const unsigned long long&>(val),
                                                   offset);
        return reinterpret_cast<T&>(temp);
    } else if constexpr (sizeof(T) == 4) {
        unsigned int temp = __shfl_down_sync(mask,
                                             reinterpret_cast<const unsigned int&>(val),
                                             offset);
        return reinterpret_cast<T&>(temp);
    }

    return val; // Fallback for unsupported sizes
  }

} // namespace ncarray

#endif // NCARRAY_DEVICE_WARP_CUH
