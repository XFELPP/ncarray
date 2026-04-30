/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "gtest/gtest.h"

#include "ncarray/array_operations.hh"
#include "ncarray/array_traits.hh"
#include "ncarray/ncarrays.hh"
#ifdef NCA_HAS_CUDA
#include "ncarray/ncdevarrays.cuh"
#endif

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
TEST(NCArrayOperationsTest, GPUBinaryOps) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCDevArray dev_arr1(shape, ncarray::DType::float32);
  ncarray::NCDevArray dev_arr2(shape, ncarray::DType::float32);

  dev_arr1.fill(2.0f);
  dev_arr2.fill(4.0f);

  ncarray::NCDevArray dev_sum = dev_arr1 + dev_arr2; // A kernel!
  cudaDeviceSynchronize();
  ncarray::NCArray host_sum(shape, ncarray::DType::float32);
  dev_sum.copy_into(host_sum.data());

  for (unsigned i = 0; i < dev_sum.size(); ++i) {
    ASSERT_FLOAT_EQ(static_cast<float>(host_sum[{0, 0, i}]), 6.0f)
      << "GPU addition failed at index " << i;
  }
}

#ifdef __CUDACC__
template <typename T, ncarray::ViewArrayLike OutT>
__global__ void iota_kernel(OutT out) {
  ssize_t idx { static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x) };
  if (idx < out.size()) {
    out[idx] = static_cast<T>(idx);
  }
}
#endif

TEST(NCArrayOperationsTest, GPUReductions) {
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

TEST(NCArrayOperationsTest, GPUSlicedReduction) {
  std::vector<ssize_t> shape { 4, 4 };
  ncarray::NCDevArray dev_arr(shape, ncarray::DType::float32);
  dev_arr.fill(1.0f);

  // Slice the middle 2x2 section and fill with 10.0f
  // NOTE: Sadly, GPU code (nvcc) is limited to C++20, so this nice syntax won't work
  // That can be used in CPU only code
  //dev_arr[ncarray::Slice(1, 3), ncarray::Slice(1, 3)].fill(10.0f);
  // We instead use the slightly more verbose version
  using Idx = ncarray::IndexItem;
  using sl = ncarray::Slice;
  Idx region[] = {
    Idx(sl(1, 3)),
    Idx(sl(1, 3))
  };
  auto view = dev_arr.view_from_indices(region, 2); // Pass number of indices
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

TEST(NCArrayOperationsTest, GPUScalarArithmetic) {
  std::vector<ssize_t> shape { 100 };
  ncarray::NCDevArray dev_arr(shape, ncarray::DType::float32);
  dev_arr.fill(10.0f);

  // Test scalar broadcasting
  ncarray::NCDevArray res = dev_arr + 5.0f;

  ncarray::NCArray host_res(shape, ncarray::DType::float32);
  res.copy_into(host_res.data());

  EXPECT_EQ(static_cast<float>(host_res[{0}]), 15.0f);
}

TEST(NCArrayOperationsTest, GPUComparison) {
  std::vector<ssize_t> shape { 10 };
  ncarray::NCDevArray dev_arr(shape, ncarray::DType::float32);
  dev_arr.fill(10.0f);

#ifdef __CUDACC__
  // Fill only the first values - so we slice and send sliced view in
  using sl = ncarray::Slice;
  auto view = dev_arr(sl(0,5));
  iota_kernel<float><<<1, 5, 0, ncarray::alloc_stream()>>>(view);
#endif

  ncarray::NCDevArray mask = dev_arr > 5.0f;

  EXPECT_EQ(mask.dtype(), ncarray::DType::bool_);

  ncarray::NCArray host_res(shape, ncarray::DType::bool_);
  mask.copy_into(host_res.data());
  EXPECT_FALSE(static_cast<bool>(host_res[{0}]));
  EXPECT_FALSE(static_cast<bool>(host_res[{1}]));
  EXPECT_FALSE(static_cast<bool>(host_res[{2}]));
  EXPECT_FALSE(static_cast<bool>(host_res[{3}]));
  EXPECT_FALSE(static_cast<bool>(host_res[{4}]));
  EXPECT_TRUE(static_cast<bool>(host_res[{5}]));
  EXPECT_TRUE(static_cast<bool>(host_res[{6}]));
  EXPECT_TRUE(static_cast<bool>(host_res[{7}]));
  EXPECT_TRUE(static_cast<bool>(host_res[{8}]));
  EXPECT_TRUE(static_cast<bool>(host_res[{9}]));
}

#endif
