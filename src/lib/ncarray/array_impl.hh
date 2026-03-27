/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ARRAY_IMPL_HH
#define NCARRAY_ARRAY_IMPL_HH

#include "ncarray/array_traits.hh"
#include "ncarray/dtype.hh"
#include "ncarray/indexing.hh"
#include "ncarray/layout.hh"
#include "ncarray/storage.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif

#ifndef NCARRAY_MAX_NDIM
#define NCARRAY_MAX_NDIM 10
#endif

namespace {
  std::vector<ssize_t> calculate_c_order_strides(const std::vector<ssize_t>& shape,
                                                 const ssize_t itemsize) {
    size_t ndim { shape.size() };
    std::vector<ssize_t> strides(ndim, itemsize);
    for (ssize_t dim = ndim - 2; dim >= 0; --dim) {
      strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    return strides;
  }

  std::vector<ssize_t>
  calculate_c_order_strides(const ssize_t ndim,
                            const ssize_t* shape,
                            const ssize_t itemsize) {
    std::vector<ssize_t> strides(ndim, itemsize);
    for (ssize_t dim = ndim - 2; dim >= 0; --dim) {
      strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    return strides;
  }
} // anonymous namespace

namespace ncarray {
  // --- Array Implementations --- //
  /**
   * The ArrayImpl provides the mean entry point for all array types.
   * It is a templated class inheriting from Layout and Storage classes, allowing
   * it to be customized to different kinds of array setups. E.g. suboffsets and
   * NCArray style classes are both accessed via the ArrayImpl.
   *
   * The class provides indexing operations as well as utilities such as repr.
   * Class member functions are implemented in `array_operations` and the two
   * headers must be combined, in the include order, array_impl then array_operations.
   */
  template <typename Layout, typename Storage>
  class ArrayImpl : public Layout, public Storage {
  public:
    using ViewType = ArrayImpl<Layout, ViewPolicy>;
    using OwnerType = ArrayImpl<Layout, OwnerPolicy>;

    NCA_HD ArrayImpl() = default;

    // Standard copy and move semantics - views are shallow copies, owners deep
    // NOTE: The Storage(Storage&) constructor cannot be used since some
    // policies (e.g. Owner) have no default - it cannot be defined since
    // knowing how much memory to allocate relies on information from Layout
    // This means that these constructors MUST be correct and initialize everything
    NCA_HD ArrayImpl(const ArrayImpl& other)
      : Layout(static_cast<const Layout&>(other))
    {
      // Handle the attributes coming from Storage
      this->m_dtype = other.dtype();
      this->m_data = other.data();
      this->m_read_only = other.read_only();

      // Specializations for the Reference and Owning classes
      if constexpr (std::is_same_v<Storage, RefPolicy>) {
        for (ssize_t i = 0; i < this->ndim(); ++i) {
          this->m_ref_ptrs[i] = other.m_ref_ptrs[i];
        }
        this->m_data = &(this->m_ref_ptrs[0]);
      } else if constexpr (std::is_same_v<Storage, OwnerPolicy>) {
        this->m_storage = std::make_unique<std::uint8_t[]>(this->nbytes());
        std::copy(reinterpret_cast<std::uint8_t*>(other.data()),
                  reinterpret_cast<std::uint8_t*>(other.data()) + this->nbytes(),
                  this->m_storage.get());
        this->m_data = reinterpret_cast<void*>(this->m_storage.get());
      }
    }

    NCA_HD ArrayImpl(ArrayImpl&& other) noexcept
      : Layout(std::move(static_cast<Layout&>(other)))
      , Storage(std::move(static_cast<Storage&>(other)))
    {
      // Owner needs to move buffer
      if constexpr (std::is_same_v<Storage, OwnerPolicy>) {
        this->m_data = reinterpret_cast<void*>(this->m_storage.get());
      } else if constexpr (std::is_same_v<Storage, RefPolicy>) {
        for (ssize_t i = 0; i < this->ndim(); ++i) {
          this->m_ref_ptrs[i] = other.m_ref_ptrs[i];
        }
        this->m_data = &(this->m_ref_ptrs[0]);
      }
    }

    // Universal interconversion - any storage type can become a view
    template <class OtherStorage>
    requires std::is_same_v<Storage, ViewPolicy>
    ArrayImpl(const ArrayImpl<Layout, OtherStorage>& other)
      : Layout(static_cast<const Layout&>(other))
      , Storage(static_cast<const OtherStorage&>(other))
    {
      this->m_data = other.data();
    }

    template <class OtherStorage>
    requires std::is_same_v<Storage, ViewPolicy>
    NCA_HD ArrayImpl(ArrayImpl&& other) noexcept
      : Layout(std::move(static_cast<const Layout&>(other)))
      , Storage(std::move(static_cast<Storage&>(other)))
    {}

    // Assignment operators just re-use above
    NCA_HD ArrayImpl& operator=(const ArrayImpl& other) {
      if (this != &other) {
        *this = ArrayImpl(other);
      }

      return *this;
    }

    NCA_HD ArrayImpl& operator=(ArrayImpl&& other) noexcept {
      if (this != &other) {
        Layout::operator=(std::move(static_cast<Layout&>(other)));
        Storage::operator=(std::move(static_cast<Storage&>(other)));

        if constexpr (std::is_same_v<Storage, OwnerPolicy>) {
          this->m_data = this->m_storage.get();
        } else if constexpr (std::is_same_v<Storage, RefPolicy>) {
          for (ssize_t i = 0; i < this->ndim(); ++i) {
            this->m_ref_ptrs[i] = other.m_ref_ptrs[i];
          }
          this->m_data = &(this->m_ref_ptrs[0]);
        }
      }

      return *this;
    }

    NCA_HD ArrayImpl(void* data_,
                     const Metadata shape_,
                     const Metadata strides_,
                     const Metadata offsets_,
                     DType dtype_,
                     ssize_t pointer_axis_,
                     bool read_only_) {
      this->m_dtype = dtype_;
      this->m_data = data_;
      this->m_pointer_axis = pointer_axis_;

      auto ndim { shape_.ndim };
      this->m_shape.set(shape_.data, ndim);
      this->m_strides.set(strides_.data, ndim);

      if constexpr (requires { this->m_offsets; }) {
        this->m_offsets.set(offsets_.data, ndim);
      } else if constexpr (requires { this->m_suboffsets; }) {
        this->m_suboffsets.set(offsets_.data, ndim);
      }

      this->m_read_only = read_only_;
    }

    NCA_HD ArrayImpl(void* data_,
                     const Metadata::value_type ndim,
                     const Metadata::value_type* shape_,
                     const Metadata::value_type* strides_,
                     DType dtype_,
                     Metadata::value_type pointer_axis_,
                     bool read_only_) {
      this->m_shape.set(shape_, ndim);
      this->m_strides.set(strides_, ndim);
      this->m_dtype = dtype_;
      this->m_pointer_axis = pointer_axis_;
      this->m_read_only = read_only_;
    }

    // --- Ref classes.... --- //
    NCA_HD ArrayImpl(const std::vector<void*>& data_ptrs,
                     const std::vector<ssize_t>& shape_,
                     const std::vector<ssize_t>& strides_,
                     DType dtype_,
                     Metadata::value_type pointer_axis_,
                     bool read_only_) {
      if constexpr (std::is_same_v<Storage, RefPolicy>) {
        for (std::size_t i = 0; i < data_ptrs.size(); ++i) {
          this->m_ref_ptrs[i] = data_ptrs[i];
        }
        this->m_data = &this->m_ref_ptrs[0];
        this->m_dtype = dtype_;
        this->m_pointer_axis = pointer_axis_;
        this->m_read_only = read_only_;
        this->m_shape.set(shape_.data(), shape_.size());
        this->m_strides.set(strides_.data(), strides_.size());
      }
    }

    // --- Owner classes.... --- //
    NCA_HD ArrayImpl(const std::vector<ssize_t>& shape_,
                     DType dtype_) {
      if constexpr (std::is_same_v<Storage, OwnerPolicy>) {
        this->m_pointer_axis = -1;
        this->m_dtype = dtype_;
        this->m_shape.set(shape_.data(), shape_.size());
        this->allocate(this->nbytes());
        this->m_data = this->m_storage.get();
        auto new_strides = calculate_c_order_strides(shape_,
                                                     ncarray::itemsize(dtype_));
        this->m_strides.set(new_strides.data(), new_strides.size());
      }
    }

    NCA_HD ArrayImpl(const Metadata& shape_, DType dtype_) {
      if constexpr (std::is_same_v<Storage, OwnerPolicy>) {
        this->m_pointer_axis = -1;
        this->m_shape.set(shape_.data, shape_.ndim);
        this->m_dtype = dtype_;
        this->allocate(this->nbytes());
        this->m_data = this->m_storage.get();
        auto new_strides = calculate_c_order_strides(this->ndim(),
                                                     shape_.data,
                                                     ncarray::itemsize(dtype_));
        this->m_strides.set(new_strides.data(), new_strides.size());
      }
    }

    NCA_HD ArrayImpl(ssize_t ndim, const ssize_t* shape_, DType dtype_) {
      if constexpr (std::is_same_v<Storage, OwnerPolicy>) {
        this->m_pointer_axis = -1;
        this->m_dtype = dtype_;
        this->m_shape.set(shape_, ndim);
        this->allocate(this->nbytes());
        this->m_data = this->m_storage.get();
        auto new_strides = calculate_c_order_strides(ndim,
                                                     shape_,
                                                     ncarray::itemsize(dtype_));
        this->m_strides.set(new_strides.data(), ndim);
      }
    }

    NCA_HD inline ssize_t nbytes() const {
      return this->size() * this->itemsize();
    }

    template <typename... Args>
      requires(sizeof...(Args) >= 0 && (IndexArg<Args> && ...))
    NCA_HD ViewType operator[](Args&&... idx_args) const {
      AxisDescr axes[NCARRAY_MAX_NDIM];

      build_new_axes(axes, 0, std::forward<Args>(idx_args)...);

      return this->template out_from_axes_ptr<ViewType>(this->m_data, axes);
    }

    template <ssize_t Axis = 0, ssize_t TotalArgs = 0>
    NCA_HD void build_new_axes(AxisDescr* new_axes, ssize_t axis) const {
      for (ssize_t a = axis; a < this->ndim(); ++a) {
        new_axes[a] = { this->shape(a), 0, this->stride(a) };
      }
    }

    template <ssize_t Axis = 0, ssize_t TotalArgs = 0, typename Arg, typename... RemainingArgs>
    NCA_HD void build_new_axes(AxisDescr* new_axes,
                               ssize_t axis,
                               Arg&& arg,
                               RemainingArgs&&... remaining) const {
      if constexpr (std::is_same_v<std::decay_t<Arg>, Ellipsis>) {
        ssize_t jump = this->ndim() - TotalArgs + 1;

        for (ssize_t a = 0; a < jump; ++a) {
          ssize_t dim = axis + a;

          ssize_t off_val { 0 };
          if constexpr (requires { this->m_offsets; }) {
            off_val = this->m_offsets[dim];
          } else if constexpr (requires { this->m_suboffsets; }) {
            off_val = this->m_suboffsets[dim];
          }

          new_axes[a] = {
            dim,
            this->m_shape[dim],
            this->m_strides[dim],
            off_val,
            this->is_pointer_axis(dim),
            false
          };
        }

        if constexpr (sizeof...(RemainingArgs) > 0) {
          build_new_axes<TotalArgs>(new_axes + jump,
                                    axis + jump,
                                    std::forward<RemainingArgs>(remaining)...);
        }
      } else {
        ssize_t length { 1 };
        ssize_t stride { this->m_strides[axis] };
        ssize_t offset { 0 };
        bool is_pointer { this->is_pointer_axis(axis) };
        bool collapsed { false };

        if constexpr (std::is_integral_v<std::decay_t<Arg>>) {
          collapsed = true;
          ssize_t idx = arg;
          if (idx < 0) {
            idx += this->m_shape[axis];
          }

          // The advance function for the layout will handle strides and offsets
          // This allows different layouts to adjust appropriately.
          offset = idx;
        } else if constexpr (std::is_same_v<std::decay_t<Arg>, Slice>) {
          ssize_t start = arg.start;
          ssize_t stop = arg.stop;
          ssize_t step = arg.step;
          length = arg.length;

          if (start < 0) {
            start += this->m_shape[axis];
          }
          if (stop < 0) {
            stop += this->m_shape[axis];
          }

          stride *= step;
          if constexpr (requires { this->m_offsets; }) {
            offset = this->m_offsets[axis] + start * this->m_strides[axis];
          } else if constexpr (requires { this->m_suboffsets; }) {
            offset = this->m_suboffsets[axis];
          }
        }

        new_axes[0] = {
          axis,
          length,
          stride,
          offset,
          is_pointer,
          collapsed
        };

        if constexpr (sizeof...(RemainingArgs) > 0) {
          build_new_axes<TotalArgs>(new_axes + 1,
                                    axis + 1,
                                    std::forward<RemainingArgs>(remaining)...);
        } else if (axis + 1 < static_cast<ssize_t>(NCARRAY_MAX_NDIM)) {
          ssize_t remaining_dims = this->ndim() - (axis + 1);
          for (ssize_t a = 0; a < remaining_dims; ++a) {
            ssize_t dim = axis + 1 + a;
            ssize_t off_val { 0 };
            if constexpr (requires { this->m_offsets; }) {
              off_val = this->m_offsets[dim];
            } else if constexpr (requires { this->m_suboffsets; }) {
              off_val = this->m_suboffsets[dim];
            }
            new_axes[a + 1] = {
              dim,
              this->m_shape[dim],
              this->m_strides[dim],
              off_val,
              this->is_pointer_axis(dim),
              false
            };
          }
        }
      }
    }

    template <class VT>
    NCA_HD VT out_from_axes_ptr(void* data_ptr, const AxisDescr* axes) const {
      Metadata new_shape;
      Metadata new_strides;
      Metadata new_offsets;

      ssize_t n_dim { 0 };
      ssize_t pointer_axis { -1 };

      // NOTE: Passing a length 1 slice does NOT collapse/remove the axis.
      // E.g. A 3-D NCArray* ncarr indexed as ncarr[:1] will have shape (1, ...)
      // The `squeeze` function can be used to remove this extra length 1 axis.
      for (ssize_t i = 0; i < this->ndim(); ++i) {
        const auto& d = axes[i];

        if (!d.collapsed) {
          if (d.is_pointer) {
            pointer_axis = n_dim;
          }
          new_shape[n_dim] = d.length;
          new_strides[n_dim] = d.stride;
          new_offsets[n_dim] = d.offset;
          n_dim++;
        } else {
          // NOTE: This call is critical! It makes that correct dereferncing and
          // offset accumulation occurs, regardless of subtype
          data_ptr = this->advance(data_ptr, i, d.offset);
        }
      }

      new_shape.ndim = n_dim;
      new_strides.ndim = n_dim;
      new_offsets.ndim = n_dim;

      return VT(data_ptr,
                new_shape,
                new_strides,
                new_offsets,
                this->m_dtype,
                pointer_axis,
                this->m_read_only);
    }

    // --- Copy, cast, modification and buffer helpers/utilities --- //
    void copy_into(void* dest_buffer) const;

    template <typename OutT>
    void copy_into_astype(OutT* dest_buffer) const;

    // TODO: Perhaps this should have some smarter logic to avoid a copy if already
    //       contiguous?
    OwnerType to_contiguous() const;

    OwnerType astype(DType& dtype_out) const;

    void fill(Scalar val);

    void assign(ArrayLike auto arr);

    ViewType squeeze() const {
      void* new_data = this->m_data;

      AxisDescr new_axes[this->ndim()];
      for (ssize_t dim = 0; dim < this->ndim(); ++dim) {
        ssize_t length { this->m_shape[dim] };
        ssize_t stride { this->m_strides[dim] };
        ssize_t offset;
        if constexpr (requires { this->m_offsets; }) {
          offset = this->m_offsets[dim];
        } else {
          offset = this->m_suboffsets[dim];
        }
        bool is_pointer { this->is_pointer_axis(dim) };
        bool collapsed { length == 1 ? true : false };
        new_axes[dim] = {
          dim,
          length,
          stride,
          offset,
          is_pointer,
          collapsed
        };
      }

      return out_from_axes_ptr<ViewType>(new_data, new_axes);
    }

    // --- Reduction operations --- //
    Scalar sum() const;

    Scalar max() const;

    Scalar min() const;

    Scalar mean() const;

    Scalar get_scalar(void* ptr) const {
      auto reduce = [&]<typename T>() -> Scalar {
        return Scalar{*reinterpret_cast<T*>(ptr)};
      };

      return dispatch(this->m_dtype, reduce);
    }

    // --- Binary operations --- //
    template <ArrayLike OtherType>
    OwnerType add(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator+(const OtherType& other) const;

    template <ArrayLike OtherType>
    OwnerType sub(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator-(const OtherType& other) const;

    template <ArrayLike OtherType>
    OwnerType mul(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator*(const OtherType& other) const;

    template <ArrayLike OtherType>
    OwnerType truediv(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator/(const OtherType& other) const;

    // --- Binary Operations Overloads for Scalar Broadcasts --- //
    OwnerType add(const Scalar& other) const;
    OwnerType operator+(const Scalar& other) const;

    OwnerType sub(const Scalar& other) const;
    OwnerType operator-(const Scalar& other) const;

    OwnerType mul(const Scalar& other) const;
    OwnerType operator*(const Scalar& other) const;

    OwnerType truediv(const Scalar& other) const;
    OwnerType operator/(const Scalar& other) const;

    // -- Iterators --- //

    //template <ViewLike VT>
    template <class VT>
    class IteratorImpl {
    public:
      // Values generated on the fly so reference type is really value type
      using iterator_category = std::input_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = ViewType;
      // using pointer = value_type*;
      using pointer = void;
      using reference = value_type;

      IteratorImpl(VT view, ssize_t idx)
        : m_view(view)
        , m_idx(idx)
      {}

      reference operator*() const { return m_view[m_idx]; }

      // pointer operator->() { return m_view; } // Not immediately sure how to do this
      IteratorImpl& operator++() {
        ++m_idx;
        return *this;
      }
      IteratorImpl operator++(int) {
        IteratorImpl tmp = *this;
        ++(*this);
        return tmp;
      }

      friend bool operator==(const IteratorImpl& a, const IteratorImpl& b) {
        return (a.m_view.data() == b.m_view.data()) && (a.m_idx == b.m_idx);
      }

      friend bool operator!=(const IteratorImpl& a, const IteratorImpl& b) {
        return !(a == b);
      }

    private:
      // Storing by value is easier and safer... The ViewType is fairly light
      VT m_view;
      ssize_t m_idx;
    };

    using Iterator = IteratorImpl<ViewType>;
    using ConstIterator = IteratorImpl<const ViewType>;

    Iterator begin();
    Iterator end();

    ConstIterator begin() const;
    ConstIterator end() const;

    std::string repr() const {
      if (this->m_shape.ndim == 0) {
        return class_name() + "([], dtype=" + ncarray::to_string(this->m_dtype) + ")";
      }

      std::string prefix { class_name() + "(" };
      std::ostringstream oss;
      oss << prefix;

      // We'll indent subsequent lines to match the opening bracket of the first axis
      size_t indent = prefix.size();
      constexpr size_t edge_items = 3;

      repr_recursive(oss, this->m_data, 0, indent, edge_items);

      oss << ", dtype=" << ncarray::to_string(this->m_dtype) << ")";
      return oss.str();
    }

  protected:
    std::string class_name() const {
      return std::string(this->layout_repr()) + std::string(this->storage_repr());
    }

    /**
     * Recursive helper for repr() that handles arbitrary dimensions.
     */
    void repr_recursive(std::ostringstream& oss,
                        void* current_data,
                        ssize_t axis,
                        ssize_t indent,
                        ssize_t edge_items) const {
      auto internal = [&]<typename T> {
        repr_recursive_dispatched<T>(oss, current_data, axis, indent, edge_items);
      };
      dispatch(this->m_dtype, internal);
    }

    template <class T>
    void repr_recursive_dispatched(std::ostringstream& oss,
                                   void* current_data,
                                   ssize_t axis,
                                   ssize_t indent,
                                   ssize_t edge_items) const {
      ssize_t dim = this->m_shape[axis];
      bool is_last_axis = (axis == static_cast<ssize_t>(this->m_shape.ndim) - 1);
      bool should_truncate = (dim > 2 * edge_items);

      oss << "[";

      auto format_element = [&](size_t i) {
        void* next_ptr = this->advance(current_data, axis, i);
        if (is_last_axis) {
          T val = *reinterpret_cast<T*>(next_ptr);
          if constexpr (std::is_integral_v<T> && sizeof(T) == 1) {
            if constexpr (std::is_unsigned_v<T>) {
              oss << static_cast<unsigned long>(val);
            } else {
              oss << static_cast<long>(val);
            }
          } else {
            oss << val;
          }
        } else {
          repr_recursive_dispatched<T>(oss, next_ptr, axis + 1, indent + 1, edge_items);
        }
      };

      if (!should_truncate) {
        for (ssize_t i = 0; i < dim; ++i) {
          format_element(i);
          if (i < dim - 1) {
            if (is_last_axis) {
              oss << ", ";
            } else {
              oss << ",\n" << std::string(indent + 1, ' ');

              // Extra newline for higher dimensions
              for (ssize_t j = 0; j < this->ndim() - axis - 2; ++j) {
                oss << "\n" << std::string(indent + 1, ' ');
              }
            }
          }
        }
      } else {
        // Truncate: show first edge_items and last edge_items
        for (ssize_t i = 0; i < edge_items; ++i) {
          format_element(i);
          oss << ", ";
          if (!is_last_axis) {
            oss << "\n" << std::string(indent + 1, ' ');
          }
        }

        oss << "...";

        if (is_last_axis) {
          oss << ", ";
        } else {
          oss << "\n" << std::string(indent + 1, ' ');
        }

        for (ssize_t i = dim - edge_items; i < dim; ++i) {
          format_element(i);
          if (i < dim - 1) {
            if (is_last_axis) {
              oss << ", ";
            } else {
              oss << ",\n" << std::string(indent + 1, ' ');
            }
          }
        }
      }

      oss << "]";
    }
  };
  // --- End Array Implementations --- //
} // namespace ncarray

#endif // NCARRAY_POLICIES_HH
