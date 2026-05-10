/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/ncarrays.hh"

#include "gtest/gtest.h"

#include <vector>

TEST(NCArrayUnaryOperationsTest, IncDecOps) {
  std::vector<ssize_t> shape { 10 };
  ncarray::NCArray arr(shape, ncarray::DType::float32);
  arr.fill(5.0f);

  // Test Unary Minus
  ncarray::NCArray neg_res = -arr;
  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(neg_res[{i}]), -5.0f);
  }

  // Test In-place Increment
  ++arr;
  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(arr[{i}]), 6.0f);
  }

  // Test In-place Decrement
  --arr;
  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(static_cast<float>(arr[{i}]), 5.0f);
  }
}

TEST(NCArrayUnaryOperationsTest, Iota) {
  std::vector<ssize_t> shape { 3, 3 };
  ncarray::NCArray arr(shape, ncarray::DType::int32);

  // Test Lazy Iota
  ncarray::NCArray iota_res = arr.iota();
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      EXPECT_EQ(static_cast<int>(iota_res[{i, j}]), i * 3 + j);
    }
  }
}
