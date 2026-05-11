/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/ncarrays.hh"
#ifdef NCA_HAS_CUDA
#include "ncarray/ncdevarrays.cuh"
#endif

#include "gtest/gtest.h"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cstdint>
#include <iostream>
#include <variant>

#ifdef NCA_HAS_CUDA

#ifdef __CUDACC__
template <typename T, ncarray::ViewArrayLike OutT>
__global__ void iota_kernel(OutT out) {
  ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };
  if (idx < out.size()) {
    out[idx] = static_cast<T>(idx);
  }
}
#endif

TEST(NCDevArrayOperationsTest, FullReductions) {
  std::vector<ssize_t> shape { 256 };

  ncarray::NCDevArray dev_arr(shape, ncarray::DType::float32);
#ifdef __CUDACC__
  iota_kernel<float><<<1, 256, 0, ncarray::alloc_stream()>>>(dev_arr.view());
#endif
  cudaDeviceSynchronize();
  // Scalars are stored in a small pool of memory accessible on host and device
  auto dev_sum = dev_arr.sum();
  auto dev_max = dev_arr.max();
  auto dev_min = dev_arr.min();

  EXPECT_EQ(std::get<float>(dev_sum), 32640.0f);
  EXPECT_EQ(std::get<float>(dev_max), 255.0f);
  EXPECT_EQ(std::get<float>(dev_min), 0.0f);
}

TEST(NCArrayOperationsTest, SlicedFullReductions) {
  std::vector<ssize_t> shape { 4, 4 };
  ncarray::NCDevArray dev_arr(shape, ncarray::DType::float32);
  dev_arr.fill(1.0f);

  // Slice the middle 2x2 section and fill with 10.0f
  // NOTE: Sadly, GPU code (nvcc) is limited to C++20, so this nice syntax won't work
  // That can be used in CPU only code when using multi-dim operator[]
  // dev_arr[ncarray::Slice(1, 3), ncarray::Slice(1, 3)].fill(10.0f);
  // We can use multi-dim operator() on GPU/C++20.

  using sl = ncarray::Slice;

  auto view = dev_arr(sl(1,3), sl(1,3)); // Pass number of indices
  view.fill(10.0f);

  auto total_sum = dev_arr.sum();
  // Entire array: (12 * 1.0) + (4 * 10.0) = 52.0
  EXPECT_EQ(std::get<float>(total_sum), 52.0f);
}

TEST(NCArrayOperationsTest, GPUToHostCopy) {
  std::vector<ssize_t> shape { 10 };
  ncarray::NCDevArray dev_arr(shape, ncarray::DType::float32);
  dev_arr.fill(5.7f);

  // Copy float array into integer host array with casts
  std::vector<std::int32_t> host_buffer(10);
  dev_arr.copy_into_astype<std::int32_t>(host_buffer.data());

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(host_buffer[i], 5);
  }
}

TEST(NCArrayOperationsTest, HostToGPUCopyCast) {
  std::vector<ssize_t> shape { 10 };
  ncarray::NCArray host_arr(shape, ncarray::DType::float32);
  host_arr.fill(5.7f);

  // Copy float array into the device array with casts
  ncarray::NCDevArray dev_arr(shape, ncarray::DType::uint32);
  dev_arr.assign(host_arr);

  // Now copy back to see if it survived the round trip
  std::vector<std::uint32_t> result_buffer(10);
  dev_arr.copy_into(result_buffer.data());

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result_buffer[i], 5);
  }
}

TEST(NCArrayOperationsTest, HostToGPUCopyNoCast) {
  std::vector<ssize_t> shape { 10 };
  ncarray::NCArray host_arr(shape, ncarray::DType::float32);
  host_arr.fill(5.7f);

  // Copy float array into the device array without casts
  ncarray::NCDevArray dev_arr(shape, ncarray::DType::float32);
  dev_arr.assign(host_arr);

  // Now copy back to see if it survived the round trip
  std::vector<float> result_buffer(10);
  dev_arr.copy_into(result_buffer.data());

  for (int i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(result_buffer[i], 5.7f);
  }
}

#endif
