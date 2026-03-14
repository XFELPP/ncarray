#include "ncarray.hh"

#include "dtype.hh"
#include "ncarrayview.hh"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace {
  std::vector<ssize_t> calculate_c_order_strides(const std::vector<ssize_t>& shape,
                                                 const ssize_t itemsize) {
    size_t ndim { shape.size() };
    std::vector<ssize_t> strides(ndim, itemsize);
    for (ssize_t dim = ndim - 2; ndim >= 0; --dim) {
      strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    return strides;
  }

  std::vector<ssize_t> calculate_c_order_strides(const ssize_t ndim,
                                                 const ssize_t* shape,
                                                 const ssize_t itemsize) {
    std::vector<ssize_t> strides(ndim, itemsize);
    for (ssize_t dim = ndim - 2; ndim >= 0; --dim) {
      strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    return strides;
  }
} // anonymous namespace

namespace ncarray {
  NCArray::NCArray(const std::vector<ssize_t>& shape_, const DType& dtype_)
    : NCArrayView(nullptr,
                  shape_,
                  calculate_c_order_strides(shape_, ncarray::itemsize(dtype_)),
                  dtype_,
                  -1) // No pointer axis == -1
  {
    m_storage = std::make_unique<std::uint8_t[]>(nbytes());
    m_data = reinterpret_cast<void**>(m_storage.get());
  }

  NCArray::NCArray(const ssize_t ndim, const ssize_t* shape_, const DType& dtype_)
    : NCArrayView(nullptr,
                  ndim,
                  shape_,
                  calculate_c_order_strides(ndim, shape_, ncarray::itemsize(dtype_)).data(),
                  dtype_,
                  -1) // No pointer axis == -1
  {
    m_storage = std::make_unique<std::uint8_t[]>(nbytes());
    m_data = reinterpret_cast<void**>(m_storage.get());
  }


  NCArray::NCArray(const NCArray& other)
    : NCArrayView(other)
  {
    // Copy data into a new buffer
    m_storage = std::make_unique<std::uint8_t[]>(nbytes());
    auto* other_ptr = other.m_storage.get();
    std::copy(other_ptr, other_ptr + nbytes(), m_storage.get());
    m_data = reinterpret_cast<void**>(m_storage.get());
  }

  NCArray::NCArray(NCArray&& other) noexcept
    : NCArrayView(std::move(other))
  {
    m_storage = std::move(other.m_storage);
    m_data = reinterpret_cast<void**>(m_storage.get());
  }

  NCArray& NCArray::operator=(const NCArray& other) {
    if (this != &other) {
      NCArrayView::operator=(other);
      // Copy data into a new buffer
      m_storage = std::make_unique<std::uint8_t[]>(nbytes());
      auto* other_ptr = other.m_storage.get();
      std::copy(other_ptr, other_ptr + nbytes(), m_storage.get());
      m_data = reinterpret_cast<void**>(m_storage.get());
    }
    return *this;
  }

  NCArray& NCArray::operator=(NCArray&& other) noexcept {
    if (this != &other) {
      NCArrayView::operator=(std::move(other));
      m_storage = std::move(other.m_storage);
      m_data = reinterpret_cast<void**>(m_storage.get());
    }
    return *this;
  }

  NCArrayView NCArray::new_sub_view(void** data,
                                    std::vector<ssize_t>& shape,
                                    std::vector<ssize_t>& strides,
                                    std::vector<ssize_t>& offsets,
                                    DType dtype,
                                    ssize_t ptr_axis) const {
    return NCArrayView(data, shape, strides, offsets, dtype, ptr_axis);
  }
} // namespace ncarray
