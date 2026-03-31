#include "gtest/gtest.h"

#include "ncarray/array_operations.hh"
#include "ncarray/ncarrays.hh"
#include "ncarray/soarrays.hh"
#ifdef NCA_HAS_CUDA
#include "ncarray/ncdevarrays.cuh"
#endif

#include <cstdint>
#include <vector>

TEST(NCArrayOperationsTest, BinaryOps) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCArray arr1(shape, ncarray::dtype_traits<float>::value);
  ncarray::NCArray arr2(shape, ncarray::dtype_traits<float>::value);

  arr1.fill(ncarray::Scalar { 2.0f });
  arr2.fill(ncarray::Scalar { 4.0f });

  // --- Binary Addition (2.0 + 4.0 = 6.0) --- //
  auto sum_res = arr1 + arr2;

  const float* sum_ptr = reinterpret_cast<const float*>(sum_res.data());
  for (ssize_t i = 0; i < sum_res.size(); ++i) {
    ASSERT_FLOAT_EQ(sum_ptr[i], 6.0f) << "Sum failed at index " << i;
  }

  // --- Binary Division (2.0 / 4.0 = 0.5) --- //
  auto div_res = arr1 / arr2;

  const float* div_ptr = reinterpret_cast<const float*>(div_res.data());
  for (ssize_t i = 0; i < div_res.size(); ++i) {
    ASSERT_FLOAT_EQ(div_ptr[i], 0.5f) << "Division failed at index " << i;
  }
}


TEST(NCArrayOperationsTest, SlicedBinaryOps) {
  std::vector<ssize_t> shape { 4, 4, 4 };
  ncarray::NCArray arr1(shape, ncarray::dtype_traits<float>::value);
  ncarray::NCArray arr2(shape, ncarray::dtype_traits<float>::value);
  arr1.fill(ncarray::Scalar { 1.0f });
  arr2.fill(ncarray::Scalar { 2.0f });

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

  const float* res_ptr = reinterpret_cast<const float*>(res.data());
  for (ssize_t i = 0; i < res.size(); ++i) {
    ASSERT_FLOAT_EQ(res_ptr[i], 3.0f) << "Slice addition failed at index " << i;
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

  auto* res_ptr = reinterpret_cast<ncarray::Float2*>(res.data());
  EXPECT_FLOAT_EQ(res_ptr[0].x, 11.0f);
  EXPECT_FLOAT_EQ(res_ptr[0].y, 12.0f);
  EXPECT_FLOAT_EQ(res_ptr[1].x, 6.0f);
  EXPECT_FLOAT_EQ(res_ptr[1].y, 7.0f);
}

TEST(NCArrayOperationsTest, Reductions) {
  std::vector<ssize_t> shape { 10 };
  ncarray::NCArray arr(shape, ncarray::dtype_traits<int32_t>::value);

  arr.fill(ncarray::Scalar { static_cast<int32_t>(2) });

  auto a_sum = arr.sum();
  auto a_max = arr.max();
  auto a_min = arr.min();
  EXPECT_EQ(std::get<int64_t>(a_sum), 20);
  EXPECT_EQ(std::get<int32_t>(a_max), 2);
  EXPECT_EQ(std::get<int32_t>(a_min), 2);
}

TEST(SOArrayOperationsTest, Reductions) {
  std::vector<ssize_t> shape { 5 };
  ncarray::SOArray arr(shape, ncarray::dtype_traits<float>::value);

  arr.fill(ncarray::Scalar { 1.5f });

  auto a_sum = arr.sum();
  auto a_max = arr.max();
  auto a_min = arr.min();
  EXPECT_EQ(std::get<double>(a_sum), 7.5);
  EXPECT_EQ(std::get<float>(a_max), 1.5f);
  EXPECT_EQ(std::get<float>(a_min), 1.5f);
}

#ifdef NCA_HAS_CUDA
TEST(NCArrayOperationsTest, GPUBinaryOps) {
  std::vector<ssize_t> shape { 4, 4, 4 };

  ncarray::NCDevArray dev_arr1(shape, ncarray::dtype_traits<float>::value);
  ncarray::NCDevArray dev_arr2(shape, ncarray::dtype_traits<float>::value);

  dev_arr1.fill(ncarray::Scalar { 2.0f });
  dev_arr2.fill(ncarray::Scalar { 4.0f });

  auto dev_sum = dev_arr1 + dev_arr2;

  ncarray::NCArray host_sum(shape, ncarray::dtype_traits<float>::value);
  dev_sum.copy_into(host_sum.data());

  const float* host_ptr = reinterpret_cast<const float*>(host_sum.data());
  for (ssize_t i = 0; i < host_sum.size(); ++i) {
    ASSERT_FLOAT_EQ(host_ptr[i], 6.0f) << "GPU addition failed at index " << i;
  }
}
#endif
