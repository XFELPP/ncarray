#ifndef NCARRAY_NCARRAYREF_HH
#define NCARRAY_NCARRAYREF_HH

#include "ncarrayview.hh"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace ncarray {
  class NCArrayRef : public NCArrayView {
  public:
    NCArrayRef(std::vector<void*>& data_,
               std::vector<ssize_t>& shape_,
               std::vector<ssize_t>& strides_,
               DType dtype_,
               ssize_t ptr_axis = 0,
               bool read_only = true);

    NCArrayRef(const NCArrayRef& other);
    NCArrayRef(NCArrayRef&& other) noexcept;
    NCArrayRef& operator=(const NCArrayRef& other);
    NCArrayRef& operator=(NCArrayRef&& other) noexcept;

  protected:
    virtual std::string class_name() const override { return std::string("NCArrayRef"); }

    virtual NCArrayView new_sub_view(void** data,
                                     std::vector<ssize_t>& shape,
                                     std::vector<ssize_t>& strides,
                                     std::vector<ssize_t>& offsets,
                                     DType dtype,
                                     ssize_t ptr_axis = 0) const override;

  private:
    std::vector<void*> m_ref_ptrs;
  };
} // namespace ncarray
#endif // NCARRAY_NCARRAYREF_HH
