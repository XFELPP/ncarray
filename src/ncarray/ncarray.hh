#ifndef NCARRAY_NCARRAY_HH
#define NCARRAY_NCARRAY_HH

#include "dtype.hh"
#include "ncarrayview.hh"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace ncarray {
  class NCArray : public NCArrayView {
  public:
    NCArray(const std::vector<ssize_t>& shape_, const DType& dtype_);
    NCArray(const ssize_t ndim, const ssize_t* shape_, const DType& dtype_);

    NCArray(const NCArray& other);
    NCArray(NCArray&& other) noexcept;
    NCArray& operator=(const NCArray& other);
    NCArray& operator=(NCArray&& other) noexcept;

  protected:
    virtual std::string class_name() const override { return std::string("NCArray"); }

    virtual NCArrayView new_sub_view(void** data,
                                     std::vector<ssize_t>& shape,
                                     std::vector<ssize_t>& strides,
                                     std::vector<ssize_t>& offsets,
                                     DType dtype,
                                     ssize_t ptr_axis = -1) const override;

  private:
    std::unique_ptr<std::uint8_t[]> m_storage;
  };

  namespace impl {
    template <> struct default_owner<void> {
      using type = ncarray::NCArray;
    };
  }
} // namespace ncarray
#endif // NCARRAY_NCARRAY_HH
