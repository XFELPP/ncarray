#include "ncarrayref.hh"

#include "ncarrayview.hh"

#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace ncarray {
  NCArrayRef::NCArrayRef(std::vector<void*>& data_,
                         std::vector<ssize_t>& shape_,
                         std::vector<ssize_t>& strides_,
                         DType dtype_,
                         ssize_t ptr_axis)
    : NCArrayView(nullptr, shape_, strides_, dtype_, ptr_axis)
    , m_ref_ptrs(data_)
  {
    m_data = m_ref_ptrs.data();
  }

  // Need to make sure to re-sync the pointer to self during copy/move
  NCArrayRef::NCArrayRef(const NCArrayRef& other)
    : NCArrayView(other)
    , m_ref_ptrs(other.m_ref_ptrs)
  {
    m_data = m_ref_ptrs.data();
  }

  NCArrayRef::NCArrayRef(NCArrayRef&& other) noexcept
    : NCArrayView(std::move(other))
    , m_ref_ptrs(std::move(other.m_ref_ptrs))
  {
    m_data = m_ref_ptrs.data();
  }

  NCArrayRef& NCArrayRef::operator=(const NCArrayRef& other) {
    if (this != &other) {
      NCArrayView::operator=(other);
      m_ref_ptrs = other.m_ref_ptrs;
      m_data = m_ref_ptrs.data();
    }
    return *this;
  }

  NCArrayRef& NCArrayRef::operator=(NCArrayRef&& other) noexcept {
    if (this != &other) {
      NCArrayView::operator=(std::move(other));
      m_ref_ptrs = std::move(other.m_ref_ptrs);
      m_data = m_ref_ptrs.data();
    }
    return *this;
  }

  NCArrayView NCArrayRef::new_sub_view(void** data,
                                       std::vector<ssize_t>& shape,
                                       std::vector<ssize_t>& strides,
                                       std::vector<ssize_t>& offsets,
                                       DType dtype,
                                       ssize_t ptr_axis) const {
    return NCArrayView(data, shape, strides, offsets, dtype, ptr_axis);
  }
} // namespace ncarray
