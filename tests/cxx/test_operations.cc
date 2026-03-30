#include "gtest/gtest.h"

#include "ncarray/array_operations.hh"
#include "ncarray/ncarrays.hh"
#include "ncarray/soarrays.hh"

#include <cstdint>
#include <vector>

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
