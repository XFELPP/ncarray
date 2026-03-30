#include "gtest/gtest.h"

#include "ncarray/ncarrays.hh"
#include "ncarray/soarrays.hh"
#ifdef NCA_HAS_CUDA
#include "ncarray/ncdevarrays.cuh"
#endif

#include <cstdint>
#include <vector>

TEST(NCArrayIndexingTest, OwnerCreationAndShape) {
  std::vector<ssize_t> shape { 4, 5 };
  ncarray::NCArray arr(shape, ncarray::dtype_traits<int32_t>::value);

  EXPECT_EQ(arr.ndim(), 2);
  EXPECT_EQ(arr.shape(0), 4);
  EXPECT_EQ(arr.shape(1), 5);
  EXPECT_EQ(arr.size(), 20);
  EXPECT_EQ(arr.itemsize(), sizeof(int32_t));
}

TEST(NCArrayIndexingTest, RefCreationAndShape) {
  std::vector<int32_t> data(10, 42);
  std::vector<ssize_t> shape { 2, 5 };
  std::vector<ssize_t> strides { 5 * sizeof(int32_t), sizeof(int32_t) };
  std::vector<void*> ptrs { data.data() };

  ncarray::NCArrayRef ref(ptrs,
                          shape,
                          strides,
                          ncarray::dtype_traits<int32_t>::value,
                          -1,
                          false);

  EXPECT_EQ(ref.ndim(), 2);
  EXPECT_EQ(ref.shape(0), 2);
  EXPECT_EQ(ref.shape(1), 5);
  EXPECT_EQ(ref.size(), 10);
  EXPECT_EQ(ref.itemsize(), sizeof(int32_t));
}

TEST(SOArrayIndexingTest, OwnerCreationAndShape) {
  std::vector<ssize_t> shape { 3, 3, 3 };
  ncarray::SOArray arr(shape, ncarray::dtype_traits<float>::value);

  EXPECT_EQ(arr.ndim(), 3);
  EXPECT_EQ(arr.size(), 27);
  EXPECT_EQ(arr.itemsize(), sizeof(float));
}

#ifdef NCA_HAS_CUDA
TEST(NCDevArrayIndexingTest, DevOwnerCreationAndShape) {
  std::vector<ssize_t> shape { 4 };
  ncarray::NCDevArray arr(shape, ncarray::dtype_traits<int32_t>::value);

  EXPECT_EQ(arr.ndim(), 1);
  EXPECT_EQ(arr.size(), 4);
}
#endif
