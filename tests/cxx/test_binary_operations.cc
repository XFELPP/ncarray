/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/ncarrays.hh"
#include "ncarray/soarrays.hh"
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
#include <variant>
#include <vector>

TEST(NCArrayBinaryOperationsTest, Addition) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCArray arr1(shape, ncarray::DType::float32);
  ncarray::NCArray arr2(shape, ncarray::DType::float32);

  arr1.fill(2.0f);
  arr2.fill(4.0f);

  // --- Binary Addition (2.0 + 4.0 = 6.0) --- //
  ncarray::NCArray sum_res = arr1 + arr2;

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      for (unsigned k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(static_cast<float>(sum_res[{i, j, k}]), 6.0f)
          << "Sum failed at index " << i << ", " << j << ", " << k;
      }
    }
  }

  // --- Binary Division (2.0 / 4.0 = 0.5) --- //
  ncarray::NCArray div_res = arr1 / arr2; // op_traits mean this will be double!

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      for (unsigned k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(static_cast<double>(div_res[{i, j, k}]), 0.5)
          << "Division failed at index " << i << ", " << j << ", " << k;
      }
    }
  }
}

TEST(NCArrayBinaryOperationsTest, Division) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCArray arr1(shape, ncarray::DType::float32);
  ncarray::NCArray arr2(shape, ncarray::DType::float32);

  arr1.fill(2.0f);
  arr2.fill(4.0f);

  // --- Binary Division (2.0 / 4.0 = 0.5) --- //
  ncarray::NCArray div_res = arr1 / arr2; // op_traits mean this will be double!

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      for (unsigned k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(static_cast<double>(div_res[{i, j, k}]), 0.5)
          << "Division failed at index " << i << ", " << j << ", " << k;
      }
    }
  }
}

TEST(NCArrayBinaryOperationsTest, InplaceAddition) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCArray arr1(shape, ncarray::DType::float32);
  ncarray::NCArray arr2(shape, ncarray::DType::float32);

  arr1.fill(2.0f);
  arr2.fill(4.0f);

  // --- Binary Addition (2.0 + 4.0 = 6.0) --- //
  arr1 += arr2;

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      for (unsigned k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(static_cast<float>(arr1[{i, j, k}]), 6.0f)
            << "Sum failed at index " << i;
      }
    }
  }
}

TEST(NCArrayBinaryOperationsTest, InplaceDivision) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCArray arr1(shape, ncarray::DType::float32);
  ncarray::NCArray arr2(shape, ncarray::DType::float32);

  arr1.fill(6.0f);
  arr2.fill(4.0f);

  // --- Binary Division (6.0 / 4.0 = 1.5) --- //
  arr1 /= arr2; // Unlike above, in-place ops won't convert type, for obvious reasons.

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      for (unsigned k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(static_cast<float>(arr1[{i, j, k}]), 1.5f)
            << "Division failed at index " << i;
      }
    }
  }
}

TEST(NCArrayBinaryOperationsTest, ScalarArithmetic) {
  std::vector<ssize_t> shape { 100 };
  ncarray::NCArray arr(shape, ncarray::DType::float32);
  arr.fill(10.0f);

  // Test scalar broadcasting
  ncarray::NCArray res = arr + 5.0f;

  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_EQ(static_cast<float>(res[{i}]), 15.0f);
    EXPECT_EQ(static_cast<float>(res[{99 - i}]), 15.0f);
  }
}

TEST(NCArrayBinaryOperationsTest, InplaceScalarArithmetic) {
  std::vector<ssize_t> shape { 100 };
  ncarray::NCArray arr(shape, ncarray::DType::float32);
  arr.fill(10.0f);

  // Test scalar broadcasting
  arr += 5.0f;

  for (unsigned i = 0; i < 10; ++i) {
    EXPECT_EQ(static_cast<float>(arr[{i}]), 15.0f);
    EXPECT_EQ(static_cast<float>(arr[{99 - i}]), 15.0f);
  }
}

TEST(NCArrayBinaryOperationsTest, Comparisons) {
  std::vector<ssize_t> shape { 5 };
  ncarray::NCArray arr1(shape, ncarray::DType::float32);
  ncarray::NCArray arr2(shape, ncarray::DType::float32);

  for (unsigned i = 0; i < 5; ++i) {
    arr1[{i}] = static_cast<float>(i);
  }
  arr2.fill(2.0f);

  ncarray::NCArray res_eq = (arr1 == arr2); // [F, F, T, F, F]
  EXPECT_FALSE(res_eq[{0}]);
  EXPECT_FALSE(res_eq[{1}]);
  EXPECT_TRUE(res_eq[{2}]);
  EXPECT_FALSE(res_eq[{3}]);
  EXPECT_FALSE(res_eq[{4}]);

  ncarray::NCArray res_lt = (arr1 < arr2); // [T, T, F, F, F]
  EXPECT_TRUE(res_lt[{0}]);
  EXPECT_TRUE(res_lt[{1}]);
  EXPECT_FALSE(res_lt[{2}]);
  EXPECT_FALSE(res_lt[{3}]);
  EXPECT_FALSE(res_lt[{4}]);
}

TEST(NCArrayBinaryOperationsTest, SlicedBinaryOps) {
  std::vector<ssize_t> shape { 4, 4, 4 };
  ncarray::NCArray arr1(shape, ncarray::DType::float32);
  ncarray::NCArray arr2(shape, ncarray::DType::float32);
  arr1.fill(1.0f);
  arr2.fill(2.0f);

  using sl = ncarray::Slice;
  using Idx = ncarray::IndexItem;
  Idx region[] = {
    Idx(sl(1, 3)),
    Idx(sl(1, 3)),
    Idx(sl(1, 3))
  };

  auto view1 = arr1[sl(1, 3), sl(1, 3), sl(1, 3)];
  auto view2 = arr2.view_from_indices(region, 3);
  ncarray::NCArray res = view1 + view2;

  EXPECT_EQ(res.ndim(), 3);
  EXPECT_EQ(res.size(), 8);
  EXPECT_EQ(res.shape(0), 2);
  EXPECT_EQ(res.shape(1), 2);
  EXPECT_EQ(res.shape(2), 2);

  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      for (unsigned k = 0; k < 2; ++k) {
        ASSERT_FLOAT_EQ(static_cast<float>(res[{i, j, k}]), 3.0f)
          << "Slice addition failed at index " << i;
      }
    }
  }
}

TEST(NCArrayUnaryOperationsTest, Modulo) {
  std::vector<ssize_t> shape { 5 };
  ncarray::NCArray arr1(shape, ncarray::DType::uint32);
  ncarray::NCArray arr2(shape, ncarray::DType::uint32);

  for (unsigned i = 0; i < 5; ++i) {
    arr1[{i}] = 10 + i; // 10, 11, 12, 13, 14
    arr2[{i}] = 3;
  }

  ncarray::NCArray mod_res = arr1 % arr2;
  EXPECT_EQ(static_cast<unsigned>(mod_res[{0}]), 1); // 10 % 3
  EXPECT_EQ(static_cast<unsigned>(mod_res[{1}]), 2); // 11 % 3
  EXPECT_EQ(static_cast<unsigned>(mod_res[{2}]), 0); // 12 % 3
}

TEST(NCArrayBinaryOperationsTest, VectorArrayAddition) {
  std::vector<ssize_t> shape { 2 };
  ncarray::NCArray a(shape, ncarray::DType::vfloat2);
  ncarray::NCArray b(shape, ncarray::DType::vfloat2);

  auto* a_ptr = reinterpret_cast<ncarray::Float2*>(a.data());
  a_ptr[0] = { 1.0f, 2.0f };
  a_ptr[1] = { 5.0f, 5.0f };

  auto* b_ptr = reinterpret_cast<ncarray::Float2*>(b.data());
  b_ptr[0] = { 10.0f, 10.0f };
  b_ptr[1] = { 1.0f, 2.0f };

  ncarray::NCArray res = a + b;

  ncarray::Float2& f2_0 = res[{0}];
  ncarray::Float2& f2_1 = res[{1}];
  EXPECT_FLOAT_EQ(f2_0.x, 11.0f);
  EXPECT_FLOAT_EQ(f2_0.y, 12.0f);
  EXPECT_FLOAT_EQ(f2_1.x, 6.0f);
  EXPECT_FLOAT_EQ(f2_1.y, 7.0f);
}
