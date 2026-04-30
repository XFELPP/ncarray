/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ARRAY_IMPL_HH
#define NCARRAY_ARRAY_IMPL_HH

#include "ncarray/array_element_proxy.hh"
#include "ncarray/array_traits.hh"
#include "ncarray/dtype.hh"
#include "ncarray/expression.hh"
#include "ncarray/indexing.hh"
#include "ncarray/layout.hh"
#include "ncarray/op_traits.hh"
#include "ncarray/mvnode.hh"
#include "ncarray/storage.hh"

#ifdef __CUDACC_RTC__

#include <cuda/std/cstdint>
#include <cuda/std/initializer_list>
#include <cuda/std/type_traits>

using cuda::std::initializer_list;
using cuda::std::is_base_of_v;
using cuda::std::is_same_v;
using cuda::std::move;
using cuda::std::uint64_t;

#else

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using std::initializer_list;
using std::is_base_of_v;
using std::move;
using std::is_same_v;
using std::uint64_t;

#endif // nvrtc guard

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

#ifndef NCARRAY_MAX_NDIM
#define NCARRAY_MAX_NDIM 10
#endif

namespace ncarray {

#ifndef __CUDACC_RTC__

  // --- Helper Functions For Metadata --- //

  inline std::vector<ssize_t> calculate_c_order_strides(const std::vector<ssize_t>& shape,
                                                        const ssize_t itemsize) {
    size_t ndim { shape.size() };
    std::vector<ssize_t> strides(ndim, itemsize);
    for (ssize_t dim = ndim - 2; dim >= 0; --dim) {
      strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    return strides;
  }

  inline std::vector<ssize_t> calculate_c_order_strides(const ssize_t ndim,
                                                        const ssize_t* shape,
                                                        const ssize_t itemsize) {
    std::vector<ssize_t> strides(ndim, itemsize);
    for (ssize_t dim = ndim - 2; dim >= 0; --dim) {
      strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    return strides;
  }

  NCA_HD inline void calculate_c_order_strides(const ssize_t ndim,
                                               const ssize_t* shape,
                                               const ssize_t itemsize,
                                               ssize_t* new_strides) {
    new_strides[ndim - 1] = itemsize;
    for (ssize_t dim = ndim - 2; dim >= 0; --dim) {
      new_strides[dim] = new_strides[dim + 1] * shape[dim + 1];
    }
  }

#endif

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
    // Exposed for concepts and traits
    using StoragePolicy = Storage;
    using LayoutPolicy = Layout;

    // --- Global type aliases --- //
    // The following are used so that for a given Layout, you can access what
    // the appropriate ViewType and OwnerType would be.
    using MemType = typename Storage::MemType;
    using VPolicy = typename StoragePolicyTraits<MemType>::View;
    using OPolicy = typename StoragePolicyTraits<MemType>::Owner;

    using ViewType = ArrayImpl<Layout, VPolicy>;
    using OwnerType = ArrayImpl<Layout, OPolicy>;

    ArrayImpl() = default;

    // Standard copy and move semantics - views are shallow copies, owners deep
    // NOTE: Copys can only be done for Owner types on the host.
    // NOTE: The Storage(Storage&) constructor cannot be used since some
    // policies (e.g. Owner) have no default - it cannot be defined since
    // knowing how much memory to allocate relies on information from Layout
    // This means that these constructors MUST be correct and initialize everything

    // --- Host-only copy constructor for owner types --- //
#ifndef __CUDACC_RTC__
    ArrayImpl(const ArrayImpl& other)
      requires std::is_base_of_v<OwnerTag, Storage>
      : Layout(static_cast<const Layout&>(other))
    {
      this->m_dtype = other.dtype();
      this->m_read_only = other.read_only();

      this->allocate(this->nbytes());
      this->copy(other.data(), this->nbytes());
      this->m_data = reinterpret_cast<void*>(this->m_storage.get());
    }

    ArrayImpl(ArrayImpl&& other) noexcept
      requires std::is_base_of_v<OwnerTag, Storage>
      : Layout(std::move(static_cast<Layout&>(other)))
      , Storage(std::move(static_cast<Storage&>(other)))
    {
      // After move make sure to reset the data pointer
      this->m_data = reinterpret_cast<void*>(this->m_storage.get());
    }
#endif
    // --- View/Ref copy/move can be done on host or device --- //
    NCA_HD ArrayImpl(const ArrayImpl& other)
      requires (!is_base_of_v<OwnerTag, Storage>)
      : Layout(static_cast<const Layout&>(other))
    {
      // Handle the attributes coming from Storage
      this->m_dtype = other.dtype();
      this->m_data = other.data();
      this->m_read_only = other.read_only();

      // Specializations for the Reference and Owning classes
      if constexpr (is_base_of_v<RefTag, Storage>) {
        for (ssize_t i = 0; i < this->ndim(); ++i) {
          this->m_ref_ptrs[i] = other.m_ref_ptrs[i];
        }
        this->m_data = &(this->m_ref_ptrs[0]);
      }
    }

    NCA_HD ArrayImpl(ArrayImpl&& other) noexcept
      requires (!is_base_of_v<OwnerTag, Storage>)
      : Layout(move(static_cast<Layout&>(other)))
      , Storage(move(static_cast<Storage&>(other)))
    {
      if constexpr (is_base_of_v<RefTag, Storage>) {
        for (ssize_t i = 0; i < this->ndim(); ++i) {
          this->m_ref_ptrs[i] = other.m_ref_ptrs[i];
        }
        this->m_data = &(this->m_ref_ptrs[0]);
      }
    }

    // Universal interconversion - any storage type can become a view
    template <class OtherStorage>
    requires is_base_of_v<ViewTag, Storage>
    NCA_HD ArrayImpl(const ArrayImpl<Layout, OtherStorage>& other)
      : Layout(static_cast<const Layout&>(other))
      , Storage()
    {
      this->m_data = other.data();
      this->m_dtype = other.dtype();
      this->m_read_only = other.read_only();
    }

    template <class OtherStorage>
    requires is_base_of_v<ViewTag, Storage>
    NCA_HD ArrayImpl(ArrayImpl<Layout, OtherStorage>&& other) noexcept
      : Layout(move(static_cast<const Layout&>(other)))
      , Storage()
    {
      this->m_data = other.data();
      this->m_dtype = other.dtype();
      this->m_read_only = other.read_only();
    }

#ifndef __CUDACC_RTC__

    // --- Host only move/copy assignment for owner types --- //

    ArrayImpl& operator=(const ArrayImpl& other)
      requires is_base_of_v<OwnerTag, Storage>
    {
      if (this != &other) {
        *this = ArrayImpl(other);
      }
      return *this;
    }

    ArrayImpl& operator=(ArrayImpl&& other) noexcept
      requires is_base_of_v<OwnerTag, Storage>
    {
      if (this != &other) {
        Layout::operator=(move(static_cast<Layout&>(other)));
        Storage::operator=(move(static_cast<Storage&>(other)));

        this->m_data = this->m_storage.get();
      }

      return *this;
    }

#endif

    // --- View/Ref copy/move assignment can be done on host or device --- //
    NCA_HD ArrayImpl& operator=(const ArrayImpl& other)
      requires (!is_base_of_v<OwnerTag, Storage>)
    {
      if (this != &other) {
        *this = ArrayImpl(other);
      }

      return *this;
    }

    NCA_HD ArrayImpl& operator=(ArrayImpl&& other) noexcept
      requires (!is_base_of_v<OwnerTag, Storage>)
    {
      if (this != &other) {
        Layout::operator=(move(static_cast<Layout&>(other)));
        Storage::operator=(move(static_cast<Storage&>(other)));

        if constexpr (is_base_of_v<RefTag, Storage>) {
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

      if constexpr (is_same_v<Layout, NCOffsetsPolicy>) {
        this->m_offsets.ndim = this->ndim();
        this->m_offsets.set(offsets_.data, ndim);
      } else if constexpr (is_same_v<Layout, SOArrayPolicy>) {
        this->m_suboffsets.ndim = this->ndim();
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
      this->m_data = data_;
      this->m_shape.set(shape_, ndim);
      this->m_strides.set(strides_, ndim);
      this->m_dtype = dtype_;
      this->m_pointer_axis = pointer_axis_;
      this->m_read_only = read_only_;
      for (ssize_t i = 0; i < this->ndim(); ++i) {
        if constexpr (is_same_v<Layout, NCOffsetsPolicy>) {
          this->m_offsets.ndim = ndim;
          this->m_offsets[i] = 0;
        } else if constexpr (is_same_v<Layout, SOArrayPolicy>) {
          this->m_suboffsets.ndim = ndim;
          this->m_suboffsets[i] = -1;
        }
      }
    }

#ifndef __CUDACC_RTC__

    // --- Ref classes.... --- //

    NCA_HD ArrayImpl(const std::vector<void*>& data_ptrs,
                     const std::vector<ssize_t>& shape_,
                     const std::vector<ssize_t>& strides_,
                     DType dtype_,
                     Metadata::value_type pointer_axis_,
                     bool read_only_) {
      if constexpr (std::is_base_of_v<RefTag, Storage>) {
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

      for (ssize_t i = 0; i < this->ndim(); ++i) {
        if constexpr (std::is_same_v<Layout, SOArrayPolicy>) {
          this->m_suboffsets.ndim = this->ndim();
          this->m_suboffsets[i] = -1;
        } else if constexpr (std::is_same_v<Layout, NCOffsetsPolicy>) {
          this->m_offsets.ndim = this->ndim();
          this->m_offsets[i] = 0;
        }
      }

      if constexpr (std::is_same_v<Layout, SOArrayPolicy>) {
        if (this->m_pointer_axis >= 0) {
          this->m_suboffsets[this->m_pointer_axis] = 0;
        }
      }
    }

    // --- Owner classes.... --- //

    /**
     * Construct a new array given a shape and datatype. This will generally be
     * used for Owner type arrays.
     *
     * NOTE: At least for the foreseeable future, this constructor cannot be
     * called from device code! A storage policy may allocate device memory for an
     * owner-type array; however, it must be constructed host-side, using host APIs.
     * This avoids complications with managing concurrent allocation in device code,
     * or kernels, the small heap for device-side malloc, lack of STL compatibility
     * device-side, and so on.
     *
     * @param[in] shape_ The shape of the new array (using a std::vector).
     * @param[in] dtype_ The datatype of the elements of the new array.
     */
    ArrayImpl(const std::vector<ssize_t>& shape_, DType dtype_) {
      if constexpr (std::is_base_of_v<OwnerTag, Storage>) {
        ssize_t ndim { static_cast<ssize_t>(shape_.size()) };
        this->m_pointer_axis = -1;
        this->m_dtype = dtype_;
        this->m_shape.ndim = ndim;
        this->m_strides.ndim = ndim;

        if constexpr (std::is_same_v<Layout, SOArrayPolicy>) {
          this->m_suboffsets.ndim = ndim;
        } else if constexpr (std::is_same_v<Layout, NCOffsetsPolicy>) {
          this->m_offsets.ndim = ndim;
        }
        auto new_strides = calculate_c_order_strides(shape_,
                                                     ncarray::itemsize(dtype_));

        // Cannot use .set function from Metadata since it is host/device
        for (ssize_t i = 0; i < ndim; ++i) {
          this->m_shape[i] = shape_[i];
          this->m_strides[i] = new_strides[i];
          if constexpr (std::is_same_v<Layout, SOArrayPolicy>) {
            this->m_suboffsets[i] = -1;
          } else if constexpr (std::is_same_v<Layout, NCOffsetsPolicy>) {
            this->m_offsets[i] = 0;
          }
        }
        this->allocate(this->nbytes());
        this->m_data = this->m_storage.get();
      }
    }

    /**
     * Construct a new array given a shape and datatype. This will generally be
     * used for Owner type arrays.
     *
     * NOTE: At least for the foreseeable future, this constructor cannot be
     * called from device code! A storage policy may allocate device memory for an
     * owner-type array; however, it must be constructed host-side, using host APIs.
     * This avoids complications with managing concurrent allocation in device code,
     * or kernels, the small heap for device-side malloc, lack of STL compatibility
     * device-side, and so on.
     *
     * @param[in] shape_ The shape of the new array (using the metadata struct).
     * @param[in] dtype_ The datatype of the elements of the new array.
     */
    ArrayImpl(const Metadata& shape_, DType dtype_) {
      if constexpr (std::is_base_of_v<OwnerTag, Storage>) {
        ssize_t ndim { shape_.ndim };
        this->m_pointer_axis = -1;
        this->m_dtype = dtype_;
        this->m_shape.ndim = ndim;
        this->m_strides.ndim = ndim;

        if constexpr (std::is_same_v<Layout, SOArrayPolicy>) {
          this->m_suboffsets.ndim = ndim;
        } else if constexpr (std::is_same_v<Layout, NCOffsetsPolicy>) {
          this->m_offsets.ndim = ndim;
        }
        auto new_strides = calculate_c_order_strides(ndim,
                                                     shape_.data,
                                                     ncarray::itemsize(dtype_));

        // Cannot use .set function from Metadata since it is host/device
        for (ssize_t i = 0; i < ndim; ++i) {
          this->m_shape[i] = shape_[i];
          this->m_strides[i] = new_strides[i];
          if constexpr (std::is_same_v<Layout, SOArrayPolicy>) {
            this->m_suboffsets[i] = -1;
          } else if constexpr (std::is_same_v<Layout, NCOffsetsPolicy>) {
            this->m_offsets[i] = 0;
          }
        }
        this->allocate(this->nbytes());
        this->m_data = this->m_storage.get();
      }
    }

    /**
     * Construct a new array given a shape and datatype. This will generally be
     * used for Owner type arrays.
     *
     * NOTE: At least for the foreseeable future, this constructor cannot be
     * called from device code! A storage policy may allocate device memory for an
     * owner-type array; however, it must be constructed host-side, using host APIs.
     * This avoids complications with managing concurrent allocation in device code,
     * or kernels, the small heap for device-side malloc, lack of STL compatibility
     * device-side, and so on.
     *
     * @param[in] ndim The number of dimensions (to properly increment the pointer).
     * @param[in] shape_ The shape of the new array (using a raw pointer).
     * @param[in] dtype_ The datatype of the elements of the new array.
     */
    ArrayImpl(ssize_t ndim, const ssize_t* shape_, DType dtype_) {
      if constexpr (std::is_base_of_v<OwnerTag, Storage>) {
        this->m_pointer_axis = -1;
        this->m_dtype = dtype_;
        this->m_shape.ndim = ndim;
        this->m_strides.ndim = ndim;

        if constexpr (std::is_same_v<Layout, SOArrayPolicy>) {
          this->m_suboffsets.ndim = ndim;
        } else if constexpr (std::is_same_v<Layout, NCOffsetsPolicy>) {
          this->m_offsets.ndim = ndim;
        }
        auto new_strides = calculate_c_order_strides(ndim,
                                                     shape_,
                                                     ncarray::itemsize(dtype_));

        // Cannot use .set function from Metadata since it is host/device
        for (ssize_t i = 0; i < ndim; ++i) {
          this->m_shape[i] = shape_[i];
          this->m_strides[i] = new_strides[i];
          if constexpr (std::is_same_v<Layout, SOArrayPolicy>) {
            this->m_suboffsets[i] = -1;
          } else if constexpr (std::is_same_v<Layout, NCOffsetsPolicy>) {
            this->m_offsets[i] = 0;
          }
        }
        this->allocate(this->nbytes());
        this->m_data = this->m_storage.get();
      }
    }

    template <Expression Expr>
    ArrayImpl(const Expr& expr)
      : ArrayImpl(expr.ndim(), expr.shape(), expr.dtype())
    {
      *this = expr;
    }

    NCA_HD ssize_t nbytes() const {
      return this->size() * this->itemsize();
    }

    // --- Int/slice/ellipsis variadic indexing to view --- //

#if __cplusplus > 202002L

    template <typename... Args>
    requires(sizeof...(Args) >= 0 && (IndexArg<Args> && ...))
    NCA_HD ViewType operator[](Args&&... idx_args) const {
      AxisDescr axes[NCARRAY_MAX_NDIM];

      if constexpr (sizeof...(Args) > 0) {
        IndexItem indices[] = { IndexItem(std::forward<Args>(idx_args))... };

        this->build_new_axes(axes, indices, sizeof...(Args));
      } else {
        this->build_new_axes(axes, nullptr, 0);
      }
      return this->template out_from_axes_ptr<ViewType>(this->m_data, axes);
    }

#endif

    template <typename... Args>
    requires(sizeof...(Args) >= 0 && (IndexArg<Args> && ...))
    NCA_HD ViewType operator()(Args&&... idx_args) const {
      AxisDescr axes[NCARRAY_MAX_NDIM];

      if constexpr (sizeof...(Args) > 0) {
        IndexItem indices[] = { IndexItem(std::forward<Args>(idx_args))... };

        this->build_new_axes(axes, indices, sizeof...(Args));
      } else {
        this->build_new_axes(axes, nullptr, 0);
      }
      return this->template out_from_axes_ptr<ViewType>(this->m_data, axes);
    }
#endif // nvrtc guard

    // --- Linearized indexing to reference (non-const and const) --- //

    NCA_HD ArrayElementProxy operator[](ssize_t idx) {
      void* out_data = const_cast<void*>(this->data());
      ssize_t lin_idx { idx };

#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC unroll 10
#elif defined(__clang__)
#pragma clang loop unroll(full)
#endif
      for (ssize_t dim = this->ndim() - 1; dim >= 0; --dim) {
        ssize_t dim_shape = this->shape(dim);
        ssize_t local_idx = lin_idx % dim_shape;

        lin_idx /= this->shape(dim);

        out_data = this->advance(out_data, dim, local_idx);
      }

      return { out_data, this->dtype() };
    }

    NCA_HD const ArrayElementProxy operator[](ssize_t idx) const {
      const void* out_data = this->data();
      ssize_t lin_idx { idx };

#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC unroll 10
#elif defined(__clang__)
#pragma clang loop unroll(full)
#endif
      for (ssize_t dim = this->ndim() - 1; dim >= 0; --dim) {
        ssize_t dim_shape = this->shape(dim);
        ssize_t local_idx = lin_idx % dim_shape;
        lin_idx /= dim_shape;
        out_data = this->advance(out_data, dim, local_idx);
      }

      return { const_cast<void*>(out_data), this->dtype() };
    }

    template <typename Coords>
    NCA_HD void lin_to_md(ssize_t idx, Coords& coords) {
      ssize_t lin_idx { idx };

#if defined(__CUDACC__)
#pragma unroll
#elif defined(__GNUC__) || defined(__GNUG__)
#pragma GCC unroll 10
#elif defined(__clang__)
#pragma clang loop unroll(full)
#endif
      for (auto dim = coords.size() - 1; dim >= 1; --dim) {
        auto dim_shape = this->shape(dim);
        coords[dim] = lin_idx % dim_shape;

        lin_idx /= dim_shape;
      }
      coords[0] = lin_idx % this->shape(0);
    }

    template <typename Coords>
    NCA_HD ArrayElementProxy operator[](const Coords& coords) {
      void* out_data = const_cast<void*>(this->data());

      for (auto dim = 0; dim < coords.size(); ++dim) {
        auto local_idx { coords[dim] };
        out_data = this->advance(out_data, dim, local_idx);
      }

      return { out_data, this->dtype() };
    }

    template <typename Coords>
    NCA_HD const ArrayElementProxy operator[](const Coords& coords) const {
      const void* out_data = this->data();

      for (auto dim = 0; dim < coords.size(); ++dim) {
        auto local_idx { coords[dim] };
        out_data = this->advance(out_data, dim, local_idx);
      }

      return { const_cast<void*>(out_data), this->dtype() };
    }

    // --- Indexing to reference --- //

    /**
     * The initializer list overloader are for indexing down to a single point reference.
     * For semantics, you can make use of the variadic operator[] (C++23) or for NVCC,
     * C++20, code, the variadic operator().
     */
    NCA_HD ArrayElementProxy operator[](initializer_list<uint64_t> coords) {
      assert(coords.size() == this->ndim());
      void* out_data = const_cast<void*>(this->data());

      ssize_t axis { 0 };
      for (auto l_idx : coords) {
        out_data = this->advance(out_data, axis++, l_idx);
      }

      return { out_data, this->dtype() };
    }

    /**
     * The initializer list overloader are for indexing down to a single point reference.
     * For semantics, you can make use of the variadic operator[] (C++23) or for NVCC,
     * C++20, code, the variadic operator().
     */
    NCA_HD const ArrayElementProxy operator[](initializer_list<uint64_t> coords) const {
      const void* out_data = this->data();

      ssize_t axis { 0 };
      for (auto l_idx : coords) {
        out_data = this->advance(out_data, axis++, l_idx);
      }

      return { const_cast<void*>(out_data), this->dtype() };
    }

#ifndef __CUDACC_RTC__

    // --- Utilities for building new views --- //

    /**
     * Provided a set of indices, construct a new ViewType of array.
     *
     * @param[in] indices The index specification. Each item contains a type
     *            (like Slice, or integer) and the axis it belongs to.
     * @param[in] num_indices The number of indices that were provided for
     *            traversing the pointer.
     * @returns The new view of the data.
     */
    NCA_HD ViewType view_from_indices(const IndexItem* indices,
                                      ssize_t num_indices) const {
      AxisDescr axes[NCARRAY_MAX_NDIM];
      this->build_new_axes(axes, indices, num_indices);

      return this->template out_from_axes_ptr<ViewType>(this->m_data, axes);
    }

    /**
     * Traverse a data pointer using per axis specifications for shapes, strides,
     * and offsets/suboffsets etc.
     *
     * This function in general is not likely to be used directly. It correctly
     * traverses the data pointer, accumulating offsets as needed, to produce a
     * final view that matches the input axes specifications.
     *
     * Using the view_from_indices function is more user friendly.
     *
     * @param[in] data_ptr The data to traverse.
     * @param[in] axes The specification for each axis in the new view.
     * @returns The new view of the data.
     */
    template <class VT>
    NCA_HD VT out_from_axes_ptr(void* data_ptr,
                                const AxisDescr* axes,
                                ssize_t ndim = -1) const {
      Metadata new_shape;
      Metadata new_strides;
      Metadata new_offsets;

      ssize_t total_dim = ndim == -1 ? this->ndim() : ndim;

      ssize_t n_dim { 0 };
      ssize_t pointer_axis { -1 };
      ssize_t shift_ptr_axis { -1 }; // Track the most recent ptr axis for shift accumulation

      // NOTE: Passing a length 1 slice does NOT collapse/remove the axis.
      // E.g. A 3-D NCArray* ncarr indexed as ncarr[:1] will have shape (1, ...)
      // The `squeeze` function can be used to remove this extra length 1 axis.
      for (ssize_t i = 0; i < total_dim; ++i) {
        const auto& d = axes[i];

        if (!d.collapsed) {
          new_shape[n_dim] = d.length;
          new_strides[n_dim] = d.stride;
          new_offsets[n_dim] = d.offset;
          if (d.data_shift != 0) {
            if (shift_ptr_axis == -1) {
              data_ptr = reinterpret_cast<std::uint8_t*>(data_ptr) + d.data_shift;
            } else {
              new_offsets[shift_ptr_axis] += d.data_shift;
            }
          }

          if (d.is_pointer) {
            pointer_axis = n_dim;
            shift_ptr_axis = n_dim;
          }

          n_dim++;
        } else {
          // NOTE: This call is critical! It makes that correct dereferncing and
          // offset accumulation occurs, regardless of subtype
          if (d.is_pointer) {
            data_ptr = this->advance(data_ptr, i, d.offset);

            // Reset the shift accumulator tracker if the axis was collapsed
            shift_ptr_axis = -1;
          } else {
            if (shift_ptr_axis == -1) {
              data_ptr = this->advance(data_ptr, i, d.offset);
            } else {
              new_offsets[shift_ptr_axis] += d.offset * this->m_strides[i];
            }
          }
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
    /**
     * Construct a view of any array.
     *
     * Some functions, and in particular device code, require views only, to
     * avoid allocations when dealing with OwnerType arrays. This function will
     * provide a view over the data from any array, so it can be used easily with
     * all APIs.
     *
     * E.g.:
     *
     * @code{.cpp}
     * gpu_kernel<<<blocks, TPB>>>(my_array.view());
     * @endcode
     */
    NCA_HD ViewType view() const { return ViewType(*this); }

    /**
     * Copy the array's data into the provided buffer.
     *
     * This will copy data using the array's datatype.
     *
     * @param[in] dest_buffer The destination for the copied data.
     */
    void copy_into(void* dest_buffer) const;

    /**
     * Copy the array's data into the provided buffer while casting.
     *
     * This will perform the appropriate casts from the array's datatype to the
     * requested datatype.
     *
     * @tparam OutT The output/destination datatype.
     * @param[in] dest_buffer The destination for the copied data.
     */
    template <typename OutT>
    void copy_into_astype(OutT* dest_buffer) const;

    // TODO: Perhaps this should have some smarter logic to avoid a copy if already
    //       contiguous?
    OwnerType to_contiguous() const;

    /**
     * Check whether the array's current data is a contiguous block.
     *
     * @returns is_contiguous Whether the data is contiguous.
     */
    NCA_HD inline bool is_contiguous() const {
      ssize_t expected_stride { this->itemsize() };
      for (ssize_t axis = this->ndim() - 1; axis >= 0; --axis) {
        if (this->is_pointer_axis(axis)) {
          // Check for pointer axes
          return false;
        }

        auto dim_shape { this->m_shape[axis] };
        if (dim_shape > 1 && this->m_strides[axis] != expected_stride) {
          // Check for sliced views with steps (e.g. ncarr[::4])
          return false;
        }
        expected_stride *= dim_shape;
      }

      // Finally, check layout classes helper (which looks at offsets etc.)
      return this->is_contiguous_impl();
    }

    OwnerType astype(DType& dtype_out) const;

    void fill(Scalar val);

    void assign(const ArrayLike auto& arr);

    ViewType squeeze() const {
      void* new_data = this->m_data;

      AxisDescr new_axes[NCARRAY_MAX_NDIM];
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
        new_axes[dim] = {
          dim,
          length,
          stride,
          // For all arrays, this is correct - out_from_axes_ptr will handle
          length == 1 ? 0 : offset,
          is_pointer,
          /*collapsed=*/length == 1
        };
      }

      return out_from_axes_ptr<ViewType>(new_data, new_axes);
    }

    /**
     * This function for reshaping assumes contiguity. It is user responsibility
     * to ensure that this is true!
     *
     * @param[in] new_shape The new shape that is requested.
     * @param[in] ndim The number of dimensions in the new shape.
     * @returns new_view The reshaped view.
     */
    NCA_HD ViewType reshape(const ssize_t* new_shape, ssize_t ndim) const {
      ssize_t new_strides[NCARRAY_MAX_NDIM] { 0 };

      calculate_c_order_strides(ndim, new_shape, this->itemsize(), new_strides);

      AxisDescr new_axes[NCARRAY_MAX_NDIM];
      for (ssize_t dim = 0; dim < ndim; ++dim) {
        ssize_t length { new_shape[dim] };
        ssize_t stride { new_strides[dim] };
        ssize_t offset { 0 };
        if constexpr (requires { this->m_suboffsets; }) {
          offset = -1;
        }
        bool is_pointer { false };
        new_axes[dim] = {
          dim,
          length,
          stride,
          offset,
          is_pointer,
          false
        };
      }

      return out_from_axes_ptr<ViewType>(this->m_data, new_axes, ndim);
    }

    /**
     * For non-contiguous arrays, there is no mathematical formula to reshape in
     * the general case. This function provides a copy and reshape utility to
     * return a new array of the specified shape.
     *
     * Because this function allocates an OwnerType, it can only be called from the
     * host.
     *
     * @param[in] new_shape The new shape that is requested.
     * @param[in] ndim The number of dimensions in the new shape.
     * @returns new_owner A new contiguous owning array of the specified shape.
     */
    OwnerType copy_as_shape(const ssize_t* new_shape, ssize_t ndim) const {
      ssize_t new_size {
        std::accumulate(new_shape, new_shape + ndim, 1, std::multiplies<ssize_t>())
      };

      if (new_size != this->size()) {
        throw std::runtime_error("Cannot reshape into the requested shape!");
      }
      OwnerType copy(ndim, new_shape, this->dtype());

      this->copy_into(copy.data());

      return copy;
    }

    // --- Axis-Aware Reductions --- //

    OwnerType sum(const std::vector<ssize_t>& axes) const;

    OwnerType max(const std::vector<ssize_t>& axes) const;
    OwnerType argmax(const std::vector<ssize_t>& axes) const;

    OwnerType min(const std::vector<ssize_t>& axes) const;
    OwnerType argmin(const std::vector<ssize_t>& axes) const;

    OwnerType mean(const std::vector<ssize_t>& axes) const;
    OwnerType var(const std::vector<ssize_t>& axes, ssize_t ddof = 0) const;
    OwnerType std(const std::vector<ssize_t>& axes, ssize_t ddof = 0) const;

    OwnerType all(const std::vector<ssize_t>& axes) const;
    OwnerType any(const std::vector<ssize_t>& axes) const;

    // --- Full Reductions (To Scalar) --- //

    Scalar sum() const;

    Scalar max() const;
    Scalar argmax() const;

    Scalar min() const;
    Scalar argmin() const;

    Scalar mean() const;
    Scalar var(ssize_t ddof = 0) const;
    Scalar std(ssize_t ddof = 0) const;

    Scalar all() const;
    Scalar any() const;

    Scalar get_scalar(void* ptr) const {
      auto reduce = [&]<typename T>() -> Scalar {
        return Scalar { *reinterpret_cast<T*>(ptr) };
      };

      return dispatch(this->m_dtype, reduce);
    }

    template <Expression Expr>
    ArrayImpl& operator=(const Expr& expr);

    // --- Binary operations --- //

    using ExprResult = ExprMVNode<MemType>;

    template <class RHS>
    ExprResult operator+(const RHS& right) const;

    template <class RHS>
    ExprResult operator-(const RHS& right) const;

    template <class RHS>
    ExprResult operator*(const RHS& right) const;

    template <class RHS>
    ExprResult operator/(const RHS& right) const;

    // --- Binary Inplace Operations --- //

    template <class RHS>
    ArrayImpl& operator+=(const RHS& right);

    template <class RHS>
    ArrayImpl& operator-=(const RHS& right);

    template <class RHS>
    ArrayImpl& operator*=(const RHS& right);

    template <class RHS>
    ArrayImpl& operator/=(const RHS& right);

    // --- Comparisons --- //

    template <class RHS>
    ExprResult operator==(const RHS& right) const;

    template <class RHS>
    ExprResult operator!=(const RHS& right) const;

    template <class RHS>
    ExprResult operator<(const RHS& right) const;

    template <class RHS>
    ExprResult operator<=(const RHS& right) const;

    template <class RHS>
    ExprResult operator>(const RHS& right) const;

    template <class RHS>
    ExprResult operator>=(const RHS& right) const;

    // --- Logical Operations --- //

    template <class RHS>
    ExprResult operator&&(const RHS& right) const;

    template <class RHS>
    ExprResult operator||(const RHS& right) const;

    ExprResult operator!() const;

    // --- Logical Inplace Operations --- //

    template <class RHS>
    ArrayImpl& operator&=(const RHS& right);

    template <class RHS>
    ArrayImpl& operator|=(const RHS& right);

    // -- Iterators --- //

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
      if constexpr (std::is_same_v<MemType, DevTag>) {
        std::ostringstream oss;
        oss << "(";
        for (ssize_t i = 0; i < this->ndim(); ++i) {
          oss << this->m_shape[i];
          if (i < this->ndim() - 1) {
            oss << ", ";
          }
        }
        oss << ")";
        return
          class_name() + "([<...(on device)...>], shape=" + oss.str() +
          ", dtype=" + ncarray::to_string(this->m_dtype) + ")";
      }
      if (this->m_shape.ndim == 0) {
        return class_name() + "([], dtype=" + ncarray::to_string(this->m_dtype) + ")";
      }

      std::string prefix { class_name() + "(" };
      std::ostringstream oss;
      oss << prefix;

      // We'll indent subsequent lines to match the opening bracket of the first axis
      size_t indent = prefix.size();
      constexpr size_t edge_items = 3;

      this->repr_recursive(oss, this->m_data, 0, indent, edge_items);

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
      auto internal = [&]<typename T>() {
        this->template repr_recursive_dispatched<T>(oss, current_data, axis, indent, edge_items);
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
          this->template repr_recursive_dispatched<T>(oss, next_ptr, axis + 1, indent + 1, edge_items);
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

#endif // nvrtc guard

  };

#ifndef __CUDACC_RTC__

  // --- End Array Implementations --- //

  /**
   * Convert an NCOffsets array to an SOArray.
   *
   * For convenience the function will consume all offsets in the NCOffsetsArray.
   * All suboffsets should then be -1 or 0.
   */
  template <typename MemTag>
  auto promote_to_so(const ArrayImpl<NCOffsetsPolicy,
                                     typename StoragePolicyTraits<MemTag>::View>& nc) {
    using SOView = ArrayImpl<SOArrayPolicy, typename StoragePolicyTraits<MemTag>::View>;

    Metadata shape;
    Metadata strides;
    Metadata suboffsets;
    auto ndim { nc.ndim() };

    void* data_ptr { const_cast<void*>(nc.data()) };

    ssize_t pointer_axis { -1 };
    shape.set(nc.shape(), ndim);
    for (ssize_t i = 0; i < ndim; ++i) {
      if (nc.is_pointer_axis(i)) {
        pointer_axis = i;
        data_ptr = reinterpret_cast<void**>(data_ptr) + nc.offset(i);

        strides[i] = sizeof(void*);
        suboffsets[i] = 0; // We just handled this for the SOArray
      } else {
        data_ptr = reinterpret_cast<std::uint8_t*>(data_ptr) + nc.offset(i);

        strides[i] = nc.stride(i);
        suboffsets[i] = -1;
      }
    }
    strides.ndim = ndim;
    suboffsets.ndim = ndim;
    SOView so(data_ptr,
              shape,
              strides,
              suboffsets,
              nc.dtype(),
              pointer_axis,
              nc.read_only());

    return so;
  }

#endif

  // --- Alias Helpers --- //

  template <typename MemType>
  using NCViewFor = ArrayImpl<NCOffsetsPolicy, typename StoragePolicyTraits<MemType>::View>;

  template <typename MemType>
  using NCRefFor = ArrayImpl<NCOffsetsPolicy, typename StoragePolicyTraits<MemType>::Ref>;

  template <typename MemType>
  using NCOwnerFor = ArrayImpl<NCOffsetsPolicy, typename StoragePolicyTraits<MemType>::Owner>;

  template <typename MemType>
  using SOViewFor = ArrayImpl<SOArrayPolicy, typename StoragePolicyTraits<MemType>::View>;

  template <typename MemType>
  using SORefFor = ArrayImpl<SOArrayPolicy, typename StoragePolicyTraits<MemType>::Ref>;

  template <typename MemType>
  using SOOwnerFor = ArrayImpl<SOArrayPolicy, typename StoragePolicyTraits<MemType>::Owner>;

  // --- Indexing Helpers --- //

  template <class Array>
  NCA_HD inline ArrayElementProxy static_index(Array arr, unsigned idx) {
    switch (arr.ndim()) {
    case 1: {
      StaticCoords<1, unsigned> coords;

      arr.lin_to_md(idx, coords);

      return arr[coords];
    }
    case 2: {
      StaticCoords<2, unsigned> coords;

      arr.lin_to_md(idx, coords);

      return arr[coords];
    }
    case 3: {
      StaticCoords<3, unsigned> coords;

      arr.lin_to_md(idx, coords);

      return arr[coords];
    }
    case 4: {
      StaticCoords<4, unsigned> coords;

      arr.lin_to_md(idx, coords);

      return arr[coords];
    }
    case 5: {
      StaticCoords<5, unsigned> coords;

      arr.lin_to_md(idx, coords);

      return arr[coords];
    }
    case 6: {
      StaticCoords<6, unsigned> coords;

      arr.lin_to_md(idx, coords);

      return arr[coords];
    }
    case 7: {
      StaticCoords<7, unsigned> coords;

      arr.lin_to_md(idx, coords);

      return arr[coords];
    }
    case 8: {
      StaticCoords<8, unsigned> coords;

      arr.lin_to_md(idx, coords);

      return arr[coords];
    }
    case 9: {
      StaticCoords<9, unsigned> coords;

      arr.lin_to_md(idx, coords);

      return arr[coords];
    }
    case 10:
    default: {
      StaticCoords<10, unsigned> coords;

      arr.lin_to_md(idx, coords);

      return arr[coords];
    }
    }
  }
} // namespace ncarray

#endif // NCARRAY_ARRAY_IMPL_HH
