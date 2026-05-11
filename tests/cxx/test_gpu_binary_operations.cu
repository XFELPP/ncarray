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
TEST(NCDevArrayBinaryOperationsTest, Addition) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCDevArray dev_arr1(shape, ncarray::DType::float32);
  ncarray::NCDevArray dev_arr2(shape, ncarray::DType::float32);

  dev_arr1.fill(2.0f);
  dev_arr2.fill(4.0f);

  // --- Binary Addition (2.0 + 4.0 = 6.0) --- //
  ncarray::NCDevArray dev_sum = dev_arr1 + dev_arr2;
  ncarray::NCArray host_sum(shape, ncarray::DType::float32);
  dev_sum.copy_into(host_sum.data());

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      for (unsigned k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(static_cast<float>(host_sum[{i, j, k}]), 6.0f)
          << "GPU addition failed at index " << i << ", " << j << ", " << k;
      }
    }
  }
}

TEST(NCDevArrayBinaryOperationsTest, Division) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCDevArray dev_arr1(shape, ncarray::DType::float32);
  ncarray::NCDevArray dev_arr2(shape, ncarray::DType::float32);

  dev_arr1.fill(2.0f);
  dev_arr2.fill(4.0f);

  // --- Binary Division (2.0 / 4.0 = 0.5) --- //
  ncarray::NCDevArray dev_div = dev_arr1 / dev_arr2; // Will be a double!
  ncarray::NCArray host_div(shape, ncarray::DType::float64);
  dev_div.copy_into(host_div.data());

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      for (unsigned k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(static_cast<double>(host_div[{i, j, k}]), 0.5)
          << "Division failed at index " << i << ", " << j << ", " << k;
      }
    }
  }
}

TEST(NCDevArrayBinaryOperationsTest, InplaceAddition) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCDevArray dev_arr1(shape, ncarray::DType::float32);
  ncarray::NCDevArray dev_arr2(shape, ncarray::DType::float32);

  dev_arr1.fill(2.0f);
  dev_arr2.fill(4.0f);

  // --- Binary Addition (2.0 + 4.0 = 6.0) --- //
  dev_arr1 += dev_arr2;
  ncarray::NCArray host_sum(shape, ncarray::DType::float32);
  dev_arr1.copy_into(host_sum.data());

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      for (unsigned k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(static_cast<float>(host_sum[{i, j, k}]), 6.0f)
          << "GPU addition failed at index " << i << ", " << j << ", " << k;
      }
    }
  }
}

TEST(NCDevArrayBinaryOperationsTest, InplaceDivision) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCDevArray dev_arr1(shape, ncarray::DType::float32);
  ncarray::NCDevArray dev_arr2(shape, ncarray::DType::float32);

  dev_arr1.fill(6.0f);
  dev_arr2.fill(4.0f);

  // --- Binary Division (6.0 / 4.0 = 1.5) --- //
  dev_arr1 /= dev_arr2; // Will remain a float!
  ncarray::NCArray host_div(shape, ncarray::DType::float32);
  dev_arr1.copy_into(host_div.data());

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      for (unsigned k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(static_cast<float>(host_div[{i, j, k}]), 1.5)
          << "Division failed at index " << i << ", " << j << ", " << k;
      }
    }
  }
}

TEST(NCDevArrayBinaryOperationsTest, ScalarArithmetic) {
  std::vector<ssize_t> shape { 100 };
  ncarray::NCDevArray dev_arr(shape, ncarray::DType::float32);
  dev_arr.fill(10.0f);

  // Test scalar broadcasting
  ncarray::NCDevArray res = dev_arr + 5.0f;

  ncarray::NCArray host_res(shape, ncarray::DType::float32);
  res.copy_into(host_res.data());

  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_EQ(static_cast<float>(host_res[{i}]), 15.0f);
    EXPECT_EQ(static_cast<float>(host_res[{99 - i}]), 15.0f);
  }
}

TEST(NCDevArrayBinaryOperationsTest, InplaceScalarArithmetic) {
  std::vector<ssize_t> shape { 100 };
  ncarray::NCDevArray dev_arr(shape, ncarray::DType::float32);
  dev_arr.fill(10.0f);

  // Test scalar broadcasting
  dev_arr += 5.0f;

  ncarray::NCArray host_res(shape, ncarray::DType::float32);
  dev_arr.copy_into(host_res.data());

  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_EQ(static_cast<float>(host_res[{i}]), 15.0f);
    EXPECT_EQ(static_cast<float>(host_res[{99 - i}]), 15.0f);
  }
}

TEST(NCDevArrayBinaryOperationsTest, Modulo) {
  std::vector<ssize_t> shape { 5 };
  ncarray::NCArray h_arr1(shape, ncarray::DType::uint32);
  ncarray::NCDevArray d_arr1(shape, ncarray::DType::uint32);
  ncarray::NCDevArray d_arr2(shape, ncarray::DType::uint32);

  for (unsigned i = 0; i < 5; ++i) {
    h_arr1[{i}] = 10 + i; // 10, 11, 12, 13, 14
  }

  d_arr1.assign(h_arr1);
  d_arr2.fill(3);

  ncarray::NCDevArray dev_mod_res = d_arr1 % d_arr2;
  ncarray::NCArray host_res(shape, ncarray::DType::uint32);
  dev_mod_res.copy_into(host_res.data());

  EXPECT_EQ(static_cast<unsigned>(host_res[{0}]), 1); // 10 % 3
  EXPECT_EQ(static_cast<unsigned>(host_res[{1}]), 2); // 11 % 3
  EXPECT_EQ(static_cast<unsigned>(host_res[{2}]), 0); // 12 % 3
}

#ifdef __CUDACC__
template <typename T, ncarray::ViewArrayLike OutT> __global__ void iota_kernel(OutT out) {
  ssize_t idx{static_cast<ssize_t>(blockIdx.x * blockDim.x + threadIdx.x)};
  if (idx < out.size()) {
    out[idx] = static_cast<T>(idx);
  }
}
#endif

TEST(NCDevArrayBinaryOperationsTest, Comparisons) {
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
