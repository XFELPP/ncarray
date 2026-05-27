/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "gtest/gtest.h"

#include "ncarray/ncarrays.hh"
#include "ncarray/soarrays.hh"
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
#include <vector>

TEST(NCArrayIndexingTest, OwnerCreationAndShape) {
  std::vector<ssize_t> shape { 4, 5 };
  ncarray::NCArray arr(shape, ncarray::dtype_traits<std::int32_t>::value);

  EXPECT_EQ(arr.ndim(), 2);
  EXPECT_EQ(arr.shape(0), 4);
  EXPECT_EQ(arr.shape(1), 5);
  EXPECT_EQ(arr.size(), 20);
  EXPECT_EQ(arr.itemsize(), sizeof(std::int32_t));
}

TEST(NCArrayIndexingTest, RefCreationAndShape) {
  std::vector<std::int32_t> data(10, 42);
  std::vector<ssize_t> shape { 2, 5 };
  std::vector<ssize_t> strides { 5 * sizeof(std::int32_t), sizeof(std::int32_t) };
  std::vector<void*> ptrs { data.data() };

  ncarray::NCArrayRef ref(ptrs,
                          shape,
                          strides,
                          ncarray::dtype_traits<std::int32_t>::value,
                          -1,
                          false);

  EXPECT_EQ(ref.ndim(), 2);
  EXPECT_EQ(ref.shape(0), 2);
  EXPECT_EQ(ref.shape(1), 5);
  EXPECT_EQ(ref.size(), 10);
  EXPECT_EQ(ref.itemsize(), sizeof(std::int32_t));
}

TEST(NCArrayIndexingTest, ElementAccess) {
  std::vector<ssize_t> shape { 2, 2 };
  ncarray::NCArray arr(shape, ncarray::DType::int32);

  arr.fill(1);

  // Test point access
  std::int32_t& item00 = arr[{0, 0}];
  std::int32_t& item01 = arr[{0, 1}];
  std::int32_t& item10 = arr[{1, 0}];
  std::int32_t& item11 = arr[{1, 1}];
  EXPECT_EQ(item00, 1);
  EXPECT_EQ(item01, 1);
  EXPECT_EQ(item10, 1);
  EXPECT_EQ(item11, 1);

  // Test point access and assignment
  arr[{0, 0}] = 10;
  arr[{0, 1}] = 20;
  arr[{1, 0}] = 30;
  arr[{1, 1}] = 40;
  // NOTE: In this case, a cast is needed since it won't deduce automatically
  // A comparison between ArrayElementProxy and integers won't work
  EXPECT_EQ(static_cast<std::int32_t>(arr[{0, 0}]), 10);
  EXPECT_EQ(static_cast<std::int32_t>(arr[{0, 1}]), 20);
  EXPECT_EQ(static_cast<std::int32_t>(arr[{1, 0}]), 30);
  EXPECT_EQ(static_cast<std::int32_t>(arr[{1, 1}]), 40);

  // Test linearized indexing
  // NOTE: MUST be a ssize_t, or wrong operator[] overload will be selected
  // TODO: Work on making this less confusing
  std::int32_t& lin_item = arr[static_cast<ssize_t>(3)];
  EXPECT_EQ(lin_item, 40);
}

TEST(NCArrayIndexingTest, VariadicSlices) {
  std::vector<ssize_t> shape { 4, 4, 4 };
  ncarray::NCArray arr1(shape, ncarray::DType::float32);
  ncarray::NCArray arr2(shape, ncarray::DType::float32);
  arr1.fill(1.0f);
  arr2.fill(2.0f);

  using sl = ncarray::Slice;
  auto view1 = arr1[sl(1,3), sl(1,3), sl(1,3)];
  EXPECT_EQ(view1.ndim(), 3);
  EXPECT_EQ(view1.size(), 8);
  EXPECT_EQ(view1.shape(0), 2);
  EXPECT_EQ(view1.shape(1), 2);
  EXPECT_EQ(view1.shape(2), 2);
}

TEST(SOArrayIndexingTest, OwnerCreationAndShape) {
  std::vector<ssize_t> shape { 3, 3, 3 };
  ncarray::SOArray arr(shape, ncarray::DType::float32);

  EXPECT_EQ(arr.ndim(), 3);
  EXPECT_EQ(arr.size(), 27);
  EXPECT_EQ(arr.itemsize(), sizeof(float));
}
