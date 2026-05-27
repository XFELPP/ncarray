/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_DEVICE_UTILITIES_CUH
#define NCARRAY_DEVICE_UTILITIES_CUH

#ifndef __CUDACC_RTC__
#ifdef NCA_HAS_CUDA
#include "cuda_runtime_api.h"
#endif

#include <iostream>

namespace ncarray {
#ifdef NCA_HAS_CUDA
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
  inline void check(cudaError_t err,
                    const char* const func,
                    const char* const file,
                    const int line) {
    if (err != cudaSuccess) {
      std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
      std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
  inline void checkLast(const char* const file, const int line) {
    cudaError_t const err { cudaGetLastError() };

    if (err != cudaSuccess) {
      std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
      std::cerr << cudaGetErrorString(err) << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#endif
} // namespace ncarray

#endif // guard for nvrtc

#endif // NCARRAY_DEVICE_UTILITIES_CUH