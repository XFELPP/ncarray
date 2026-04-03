#include "gtest/gtest.h"

#include "ncarray/array_operations.hh"
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

TEST(NCArrayOperationsTest, BinaryOps) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCArray arr1(shape, ncarray::DType::float32);
  ncarray::NCArray arr2(shape, ncarray::DType::float32);

  arr1.fill(2.0f);
  arr2.fill(4.0f);

  // --- Binary Addition (2.0 + 4.0 = 6.0) --- //
  auto sum_res = arr1 + arr2;

  for (ssize_t i = 0; i < 4; ++i) {
    for (ssize_t j = 0; j < 4; ++j) {
      for (ssize_t k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(static_cast<float>(sum_res(i, j, k)), 6.0f)
          << "Sum failed at index " << i;
      }
    }
  }

  // --- Binary Division (2.0 / 4.0 = 0.5) --- //
  auto div_res = arr1 / arr2; // op_traits mean this will be double!

  for (ssize_t i = 0; i < 4; ++i) {
    for (ssize_t j = 0; j < 4; ++j) {
      for (ssize_t k = 0; k < 4; ++k) {
        ASSERT_FLOAT_EQ(static_cast<double>(div_res(i, j, k)), 0.5)
          << "Division failed at index " << i;
      }
    }
  }
}

TEST(NCArrayOperationsTest, Comparisons) {
  std::vector<ssize_t> shape { 5 };
  ncarray::NCArray arr1(shape, ncarray::DType::float32);
  ncarray::NCArray arr2(shape, ncarray::DType::float32);

  for (ssize_t i = 0; i < 5; ++i) {
    arr1(i) = i;
  }
  arr2.fill(2.0f);

  auto res_eq = (arr1 == arr2); // [F, F, T, F, F]
  EXPECT_FALSE(res_eq(0));
  EXPECT_FALSE(res_eq(1));
  EXPECT_TRUE(res_eq(2));
  EXPECT_FALSE(res_eq(3));
  EXPECT_FALSE(res_eq(4));

  auto res_lt = (arr1 < arr2); // [T, T, F, F, F]
  EXPECT_TRUE(res_lt(0));
  EXPECT_TRUE(res_lt(1));
  EXPECT_FALSE(res_lt(2));
  EXPECT_FALSE(res_lt(3));
  EXPECT_FALSE(res_lt(4));
}

TEST(NCArrayOperationsTest, SlicedBinaryOps) {
  std::vector<ssize_t> shape { 4, 4, 4 };
  ncarray::NCArray arr1(shape, ncarray::DType::float32);
  ncarray::NCArray arr2(shape, ncarray::DType::float32);
  arr1.fill(1.0f);
  arr2.fill(2.0f);

  ncarray::IndexItem region[] = {
    ncarray::IndexItem(ncarray::Slice(1, 3)),
    ncarray::IndexItem(ncarray::Slice(1, 3)),
    ncarray::IndexItem(ncarray::Slice(1, 3))
  };

  auto view1 = arr1.view_from_indices(region, 3);
  auto view2 = arr2.view_from_indices(region, 3);
  auto res = view1 + view2;

  EXPECT_EQ(res.ndim(), 3);
  EXPECT_EQ(res.size(), 8);
  EXPECT_EQ(res.shape(0), 2);
  EXPECT_EQ(res.shape(1), 2);
  EXPECT_EQ(res.shape(2), 2);

  for (ssize_t i = 0; i < res.size(); ++i) {
    ASSERT_FLOAT_EQ(static_cast<float>(res(0, 0, i)), 3.0f)
      << "Slice addition failed at index " << i;
  }
}

TEST(NCArray, VectorArrayAddition) {
  std::vector<ssize_t> shape { 2 };
  ncarray::NCArray a(shape, ncarray::DType::vfloat2);
  ncarray::NCArray b(shape, ncarray::DType::vfloat2);

  auto* a_ptr = reinterpret_cast<ncarray::Float2*>(a.data());
  a_ptr[0] = { 1.0f, 2.0f };
  a_ptr[1] = { 5.0f, 5.0f };

  auto* b_ptr = reinterpret_cast<ncarray::Float2*>(b.data());
  b_ptr[0] = { 10.0f, 10.0f };
  b_ptr[1] = { 1.0f, 2.0f };

  auto res = a + b;

  ncarray::Float2& f2_0 = res(0);
  ncarray::Float2& f2_1 = res(1);
  EXPECT_FLOAT_EQ(f2_0.x, 11.0f);
  EXPECT_FLOAT_EQ(f2_0.y, 12.0f);
  EXPECT_FLOAT_EQ(f2_1.x, 6.0f);
  EXPECT_FLOAT_EQ(f2_1.y, 7.0f);
}

TEST(NCArrayOperationsTest, Reductions) {
  std::vector<ssize_t> shape { 10 };
  ncarray::NCArray arr(shape, ncarray::dtype_traits<int32_t>::value);

  arr.fill(static_cast<int32_t>(2));

  auto a_sum = arr.sum();
  auto a_max = arr.max();
  auto a_min = arr.min();
  EXPECT_EQ(std::get<int32_t>(a_sum), 20);
  EXPECT_EQ(std::get<int32_t>(a_max), 2);
  EXPECT_EQ(std::get<int32_t>(a_min), 2);
}

TEST(SOArrayOperationsTest, Reductions) {
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
