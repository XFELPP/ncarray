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

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cstdint>
#include <iostream>
#include <variant>
#include <vector>

TEST(NCArrayReductionsTest, FullReductions) {
  std::vector<ssize_t> shape { 10 };
  ncarray::NCArray arr(shape, ncarray::dtype_traits<std::int32_t>::value);

  arr.fill(static_cast<std::int32_t>(2));

  auto a_sum = arr.sum();
  auto a_max = arr.max();
  auto a_min = arr.min();
  EXPECT_EQ(std::get<std::int32_t>(a_sum), 20);
  EXPECT_EQ(std::get<std::int32_t>(a_max), 2);
  EXPECT_EQ(std::get<std::int32_t>(a_min), 2);
}

TEST(NCArrayReductionsTest, AxisAwareReductions) {
  std::vector<ssize_t> shape { 2, 3 };
  ncarray::NCArray arr(shape, ncarray::dtype_traits<std::int32_t>::value);
  // Initialize with [1, 2, 3]
  //                 [4, 5, 6]
  arr = arr.iota() + 1; // iota gives [0, 5], so add 1 to get it to start at 1

  // --- Reduce axis 0 ---

  // Expected sum: [5, 7, 9]
  ncarray::NCArray sum_ax0 = arr.sum({0});
  EXPECT_EQ(sum_ax0.shape()[0], 3);

  EXPECT_EQ(static_cast<std::int32_t>(sum_ax0[{0}]), 5);
  EXPECT_EQ(static_cast<std::int32_t>(sum_ax0[{1}]), 7);
  EXPECT_EQ(static_cast<std::int32_t>(sum_ax0[{2}]), 9);

  // Expected max: [4, 5, 6]

  ncarray::NCArray max_ax0 = arr.max({0});
  EXPECT_EQ(static_cast<std::int32_t>(max_ax0[{0}]), 4);
  EXPECT_EQ(static_cast<std::int32_t>(max_ax0[{1}]), 5);
  EXPECT_EQ(static_cast<std::int32_t>(max_ax0[{2}]), 6);

  // Expected min: [1, 2, 3]
  ncarray::NCArray min_ax0 = arr.min({0});
  EXPECT_EQ(static_cast<std::int32_t>(min_ax0[{0}]), 1);
  EXPECT_EQ(static_cast<std::int32_t>(min_ax0[{1}]), 2);
  EXPECT_EQ(static_cast<std::int32_t>(min_ax0[{2}]), 3);

  // --- Reduce axis 1 ---

  // Expected sum: [6, 15]
  ncarray::NCArray sum_ax1 = arr.sum({1});
  EXPECT_EQ(sum_ax1.shape()[0], 2);

  EXPECT_EQ(static_cast<std::int32_t>(sum_ax1[{0}]), 6);
  EXPECT_EQ(static_cast<std::int32_t>(sum_ax1[{1}]), 15);

  // Expected max: [3, 6]
  ncarray::NCArray max_ax1 = arr.max({1});
  EXPECT_EQ(static_cast<std::int32_t>(max_ax1[{0}]), 3);
  EXPECT_EQ(static_cast<std::int32_t>(max_ax1[{1}]), 6);

  // Expected min: [1, 4]
  ncarray::NCArray min_ax1 = arr.min({1});
  EXPECT_EQ(static_cast<std::int32_t>(min_ax1[{0}]), 1);
  EXPECT_EQ(static_cast<std::int32_t>(min_ax1[{1}]), 4);
}

TEST(SOArrayReductionsTest, FullReductions) {
  std::vector<ssize_t> shape { 5 };
  ncarray::SOArray arr(shape, ncarray::dtype_traits<float>::value);

  arr.fill(1.5f);

  auto a_sum = arr.sum();
  auto a_max = arr.max();
  auto a_min = arr.min();
  EXPECT_EQ(std::get<float>(a_sum), 7.5f);
  EXPECT_EQ(std::get<float>(a_max), 1.5f);
  EXPECT_EQ(std::get<float>(a_min), 1.5f);
}

TEST(SOArrayReductionsTest, AxisAwareReductions) {
  std::vector<ssize_t> shape { 2, 3 };
  ncarray::SOArray arr(shape, ncarray::dtype_traits<std::int32_t>::value);
  // Initialize with [1, 2, 3]
  //                 [4, 5, 6]
  arr = arr.iota() + 1; // iota gives [0, 5], so add 1 to get it to start at 1

  // --- Reduce axis 0 ---

  // Expected sum: [5, 7, 9]
  ncarray::SOArray sum_ax0 = arr.sum({0});
  EXPECT_EQ(sum_ax0.shape()[0], 3);

  EXPECT_EQ(static_cast<std::int32_t>(sum_ax0[{0}]), 5);
  EXPECT_EQ(static_cast<std::int32_t>(sum_ax0[{1}]), 7);
  EXPECT_EQ(static_cast<std::int32_t>(sum_ax0[{2}]), 9);

  // Expected max: [4, 5, 6]

  ncarray::SOArray max_ax0 = arr.max({0});
  EXPECT_EQ(static_cast<std::int32_t>(max_ax0[{0}]), 4);
  EXPECT_EQ(static_cast<std::int32_t>(max_ax0[{1}]), 5);
  EXPECT_EQ(static_cast<std::int32_t>(max_ax0[{2}]), 6);

  // Expected min: [1, 2, 3]
  ncarray::SOArray min_ax0 = arr.min({0});
  EXPECT_EQ(static_cast<std::int32_t>(min_ax0[{0}]), 1);
  EXPECT_EQ(static_cast<std::int32_t>(min_ax0[{1}]), 2);
  EXPECT_EQ(static_cast<std::int32_t>(min_ax0[{2}]), 3);

  // --- Reduce axis 1 ---

  // Expected sum: [6, 15]
  ncarray::SOArray sum_ax1 = arr.sum({1});
  EXPECT_EQ(sum_ax1.shape()[0], 2);

  EXPECT_EQ(static_cast<std::int32_t>(sum_ax1[{0}]), 6);
  EXPECT_EQ(static_cast<std::int32_t>(sum_ax1[{1}]), 15);

  // Expected max: [3, 6]
  ncarray::SOArray max_ax1 = arr.max({1});
  EXPECT_EQ(static_cast<std::int32_t>(max_ax1[{0}]), 3);
  EXPECT_EQ(static_cast<std::int32_t>(max_ax1[{1}]), 6);

  // Expected min: [1, 4]
  ncarray::SOArray min_ax1 = arr.min({1});
  EXPECT_EQ(static_cast<std::int32_t>(min_ax1[{0}]), 1);
  EXPECT_EQ(static_cast<std::int32_t>(min_ax1[{1}]), 4);
}
