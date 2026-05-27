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

#include <vector>

#ifdef NCA_HAS_CUDA
TEST(NCDevArrayUnaryOperationsTest, IncDecOps) {
  std::vector<ssize_t> shape { 10 };
  ncarray::NCDevArray dev_arr(shape, ncarray::DType::float32);
  dev_arr.fill(5.0f);

  // Test Unary Minus
  ncarray::NCDevArray dev_neg = -dev_arr;
  ncarray::NCArray host_neg(shape, ncarray::DType::float32);
  dev_neg.copy_into(host_neg.data());
  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(host_neg[{i}]), -5.0f);
  }

  // Test In-place Increment
  ++dev_arr;
  ncarray::NCArray host_arr(shape, ncarray::DType::float32);
  dev_arr.copy_into(host_arr.data());

  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(host_arr[{i}]), 6.0f);
  }

  // Test In-place Decrement
  --dev_arr;
  dev_arr.copy_into(host_arr.data());
  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(host_arr[{i}]), 5.0f);
  }
}

TEST(NCDevArrayUnaryOperationsTest, Iota) {
  std::vector<ssize_t> shape { 3, 3 };
  ncarray::NCDevArray dev_arr(shape, ncarray::DType::int32);

  // Test Lazy Iota
  ncarray::NCDevArray dev_iota_res = dev_arr.iota();
  ncarray::NCArray host_iota_res(shape, ncarray::DType::int32);
  dev_iota_res.copy_into(host_iota_res.data());
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      EXPECT_EQ(static_cast<int>(host_iota_res[{i, j}]), i * 3 + j);
    }
  }

  // Test immediate evaluation
  ncarray::NCDevArray dev_iota_res2 = ncarray::NCDevArray::iota(shape); // int64 by default
  ncarray::NCArray host_iota_res2(shape, ncarray::DType::int64);
  dev_iota_res2.copy_into(host_iota_res2.data());
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      EXPECT_EQ(static_cast<std::int64_t>(host_iota_res2[{i, j}]), i * 3 + j);
    }
  }

  // Test immediate evaluation with cast
  ncarray::NCDevArray dev_iota_res3 = ncarray::NCDevArray::iota(shape, ncarray::DType::float32);
  ncarray::NCArray host_iota_res3(shape, ncarray::DType::float32);
  dev_iota_res3.copy_into(host_iota_res3.data());
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(static_cast<float>(host_iota_res3[{i, j}]), static_cast<float>(i * 3 + j));
    }
  }

  // Test iota as part of larger expression
  ncarray::NCDevArray dev_iota_res4 = dev_arr.iota() * 2 + 1;
  ncarray::NCArray host_iota_res4(shape, ncarray::DType::int32);
  dev_iota_res4.copy_into(host_iota_res4.data());
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      EXPECT_EQ(static_cast<std::int32_t>(host_iota_res4[{i, j}]), (i * 3 + j) * 2 + 1);
    }
  }

}

#endif
