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

TEST(NCDevArrayCopies, GPUToHostCopy) {
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

TEST(NCDevArrayCopies, HostToGPUCopyCast) {
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

TEST(NCDevArrayCopies, HostToGPUCopyNoCast) {
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
