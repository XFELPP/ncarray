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
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

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
  // --- Array Element Proxy --- //
  /**
   * The element proxy can be returned by arrays during indexing to provide
   * reference based access to the underlying data.
   *
   * This mostly provides improved ergonomics.
   *
   * operator T& style functions do NOT type check. If you require type checking
   * an `at<T>` function is provided.
   */
  struct ArrayElementProxy {
    void* m_data;
    DType m_dtype;

    template <typename T>
    NCA_HD inline operator T&() const {
      return *reinterpret_cast<T*>(m_data);
    }

    template <typename T>
    NCA_HD inline operator const T&() const {
      return *reinterpret_cast<const T*>(m_data);
    }

    template <typename T>
    NCA_HD inline ArrayElementProxy& operator=(const T& val) {
      *reinterpret_cast<T*>(m_data) = val;
      return *this;
    }

    // --- Type checking versions --- //
    template <typename T>
    NCA_HD inline T& at() {
      assert(m_dtype == dtype_traits<T>::value);
      return *reinterpret_cast<T*>(m_data);
    }

    template <typename T>
    NCA_HD inline const T& at() const {
      assert(m_dtype == dtype_traits<T>::value);
      return *reinterpret_cast<const T*>(m_data);
    }

    template <typename T>
    NCA_HD inline void set(const T& val) {
      assert(m_dtype == dtype_traits<T>::value);
      *reinterpret_cast<T*>(m_data) = val;
    }
  };

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

    // --- View/Ref copy/move can be done on host or device --- //
    NCA_HD ArrayImpl(const ArrayImpl& other)
      requires (!std::is_base_of_v<OwnerTag, Storage>)
      : Layout(static_cast<const Layout&>(other))
    {
      // Handle the attributes coming from Storage
      this->m_dtype = other.dtype();
      this->m_data = other.data();
      this->m_read_only = other.read_only();

      // Specializations for the Reference and Owning classes
      if constexpr (std::is_base_of_v<RefTag, Storage>) {
        for (ssize_t i = 0; i < this->ndim(); ++i) {
          this->m_ref_ptrs[i] = other.m_ref_ptrs[i];
        }
        this->m_data = &(this->m_ref_ptrs[0]);
      }
    }

    NCA_HD ArrayImpl(ArrayImpl&& other) noexcept
      requires (!std::is_base_of_v<OwnerTag, Storage>)
      : Layout(std::move(static_cast<Layout&>(other)))
      , Storage(std::move(static_cast<Storage&>(other)))
    {
      if constexpr (std::is_base_of_v<RefTag, Storage>) {
        for (ssize_t i = 0; i < this->ndim(); ++i) {
          this->m_ref_ptrs[i] = other.m_ref_ptrs[i];
        }
        this->m_data = &(this->m_ref_ptrs[0]);
      }
    }

    // Universal interconversion - any storage type can become a view
    template <class OtherStorage>
    requires std::is_base_of_v<ViewTag, Storage>
    NCA_HD ArrayImpl(const ArrayImpl<Layout, OtherStorage>& other)
      : Layout(static_cast<const Layout&>(other))
      , Storage()
    {
      this->m_data = other.data();
      this->m_dtype = other.dtype();
      this->m_read_only = other.read_only();
    }

    template <class OtherStorage>
    requires std::is_base_of_v<ViewTag, Storage>
    NCA_HD ArrayImpl(ArrayImpl<Layout, OtherStorage>&& other) noexcept
      : Layout(std::move(static_cast<const Layout&>(other)))
      , Storage()
    {
      this->m_data = other.data();
      this->m_dtype = other.dtype();
      this->m_read_only = other.read_only();
    }

    // --- Host only move/copy assignment for owner types --- //
    ArrayImpl& operator=(const ArrayImpl& other)
      requires std::is_base_of_v<OwnerTag, Storage>
    {
      if (this != &other) {
        *this = ArrayImpl(other);
      }
      return *this;
    }

    ArrayImpl& operator=(ArrayImpl&& other) noexcept
      requires std::is_base_of_v<OwnerTag, Storage>
    {
      if (this != &other) {
        Layout::operator=(std::move(static_cast<Layout&>(other)));
        Storage::operator=(std::move(static_cast<Storage&>(other)));

        this->m_data = this->m_storage.get();
      }

      return *this;
    }

    // --- View/Ref copy/move assignment can be done on host or device --- //
    NCA_HD ArrayImpl& operator=(const ArrayImpl& other)
      requires (!std::is_base_of_v<OwnerTag, Storage>)
    {
      if (this != &other) {
        *this = ArrayImpl(other);
      }

      return *this;
    }

    NCA_HD ArrayImpl& operator=(ArrayImpl&& other) noexcept
      requires (!std::is_base_of_v<OwnerTag, Storage>)
    {
      if (this != &other) {
        Layout::operator=(std::move(static_cast<Layout&>(other)));
        Storage::operator=(std::move(static_cast<Storage&>(other)));

        if constexpr (std::is_base_of_v<RefTag, Storage>) {
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

      if constexpr (std::is_same_v<Layout, NCOffsetsPolicy>) {
        this->m_offsets.ndim = this->ndim();
        this->m_offsets.set(offsets_.data, ndim);
      } else if constexpr (std::is_same_v<Layout, SOArrayPolicy>) {
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
        if constexpr (std::is_same_v<Layout, NCOffsetsPolicy>) {
          this->m_offsets.ndim = ndim;
          this->m_offsets[i] = 0;
        } else if constexpr (std::is_same_v<Layout, SOArrayPolicy>) {
          this->m_suboffsets.ndim = ndim;
          this->m_suboffsets[i] = -1;
        }
      }
    }

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

    NCA_HD inline ssize_t nbytes() const {
      return this->size() * this->itemsize();
    }

    // --- Int/slice/ellipsis variadic indexing to view --- //

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

    // --- Linearized indexing to reference (non-const and const) --- //

    NCA_HD inline ArrayElementProxy operator[](ssize_t idx) {
      void* out_data = const_cast<void*>(this->data());
      ssize_t lin_idx { idx };

#ifdef __CUDACC__
#pragma unroll
#endif
      for (ssize_t dim = this->ndim() - 1; dim >= 0; --dim) {
        ssize_t dim_shape = this->shape(dim);
        ssize_t local_idx = lin_idx % dim_shape;

        lin_idx /= this->shape(dim);

        out_data = this->advance(out_data, dim, local_idx);
      }

      return { out_data, this->dtype() };
    }

    NCA_HD inline const ArrayElementProxy operator[](ssize_t idx) const {
      const void* out_data = this->data();
      ssize_t lin_idx { idx };

#ifdef __CUDACC__
#pragma unroll
#endif
      for (ssize_t dim = this->ndim() - 1; dim >= 0; --dim) {
        ssize_t dim_shape = this->shape(dim);
        ssize_t local_idx = lin_idx % dim_shape;
        lin_idx /= dim_shape;
        out_data = this->advance(out_data, dim, local_idx);
      }

      return { const_cast<void*>(out_data), this->dtype() };
    }

    // --- Variadic indexing to reference --- //
    // operator() overloads are intended for indexing down to a single point reference.
    // For view semantics, use the variadic operator[] - a specialized version of that
    // operator also accepts a "linearized" index to traverse the entire N-D array
    // with one index.
    template <typename... Args>
    requires (sizeof...(Args) > 0 && (std::integral<std::decay_t<Args>> && ...))
    NCA_HD inline ArrayElementProxy operator()(Args... args) {
      assert(sizeof...(Args) == this->ndim());

      void* ptr = const_cast<void*>(this->data());
      ssize_t axis { 0 };

      ((ptr = this->advance(ptr, axis++, static_cast<ssize_t>(args))), ...);

      return { ptr, this->dtype() };
    }

    template <typename... Args>
    requires (sizeof...(Args) > 0 && (std::integral<std::decay_t<Args>> && ...))
    NCA_HD inline const ArrayElementProxy operator()(Args... args) const {
      assert(sizeof...(Args) == this->ndim());

      const void* ptr = this->data();
      ssize_t axis { 0 };

      ((ptr = this->advance(ptr, axis++, static_cast<ssize_t>(args))), ...);

      return { const_cast<void*>(ptr), this->dtype() };
    }

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
    NCA_HD VT out_from_axes_ptr(void* data_ptr, const AxisDescr* axes) const {
      Metadata new_shape;
      Metadata new_strides;
      Metadata new_offsets;

      ssize_t n_dim { 0 };
      ssize_t pointer_axis { -1 };
      ssize_t shift_ptr_axis { -1 }; // Track the most recent ptr axis for shift accumulation

      // NOTE: Passing a length 1 slice does NOT collapse/remove the axis.
      // E.g. A 3-D NCArray* ncarr indexed as ncarr[:1] will have shape (1, ...)
      // The `squeeze` function can be used to remove this extra length 1 axis.
      for (ssize_t i = 0; i < this->ndim(); ++i) {
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
    ViewType view() const { return ViewType(*this); }

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

    // --- Binary inplace operations --- //
    template <ArrayLike OtherType>
    inline ArrayImpl& iadd(const OtherType& other);
    template <ArrayLike OtherType>
    inline ArrayImpl& operator+=(const OtherType& other);

    template <ArrayLike OtherType>
    inline ArrayImpl& isub(const OtherType& other);
    template <ArrayLike OtherType>
    inline ArrayImpl& operator-=(const OtherType& other);

    template <ArrayLike OtherType>
    inline ArrayImpl& imul(const OtherType& other);
    template <ArrayLike OtherType>
    inline ArrayImpl& operator*=(const OtherType& other);

    template <ArrayLike OtherType>
    inline ArrayImpl& itruediv(const OtherType& other);
    template <ArrayLike OtherType>
    inline ArrayImpl& operator/=(const OtherType& other);

    // --- Binary Operations Overloads for Scalar Broadcasts --- //
    OwnerType add(const Scalar& other) const;
    OwnerType operator+(const Scalar& other) const;

    OwnerType sub(const Scalar& other) const;
    OwnerType operator-(const Scalar& other) const;

    OwnerType mul(const Scalar& other) const;
    OwnerType operator*(const Scalar& other) const;

    OwnerType truediv(const Scalar& other) const;
    OwnerType operator/(const Scalar& other) const;

    // --- Inplace binary operations with scalar broadcasts --- //

    inline ArrayImpl& iadd(const Scalar& other);
    inline ArrayImpl& operator+=(const Scalar& other);

    inline ArrayImpl& isub(const Scalar& other);
    inline ArrayImpl& operator-=(const Scalar& other);

    inline ArrayImpl& imul(const Scalar& other);
    inline ArrayImpl& operator*=(const Scalar& other);

    inline ArrayImpl& itruediv(const Scalar& other);
    inline ArrayImpl& operator/=(const Scalar& other);

    // --- Logical and boolean operators --- //

    template <ArrayLike OtherType>
    OwnerType is_equal(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator==(const OtherType& other) const;

    template <ArrayLike OtherType>
    OwnerType is_not_equal(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator!=(const OtherType& other) const;

    template <ArrayLike OtherType>
    OwnerType is_less_than(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator<(const OtherType& other) const;

    template <ArrayLike OtherType>
    OwnerType is_less_equal_than(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator<=(const OtherType& other) const;

    template <ArrayLike OtherType>
    OwnerType is_greater_than(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator>(const OtherType& other) const;

    template <ArrayLike OtherType>
    OwnerType is_greater_equal_than(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator>=(const OtherType& other) const;

    template <ArrayLike OtherType>
    OwnerType logical_and(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator&&(const OtherType& other) const;

    template <ArrayLike OtherType>
    OwnerType logical_or(const OtherType& other) const;
    template <ArrayLike OtherType>
    OwnerType operator||(const OtherType& other) const;

    OwnerType logical_not() const;
    OwnerType operator!() const;

    // --- Inplace logical operators --- //

    template <ArrayLike OtherType>
    inline ArrayImpl& ilogical_and(const OtherType& other);
    template <ArrayLike OtherType>
    inline ArrayImpl& operator&=(const OtherType& other);

    template <ArrayLike OtherType>
    inline ArrayImpl& ilogical_or(const OtherType& other);
    template <ArrayLike OtherType>
    inline ArrayImpl& operator|=(const OtherType& other);

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
  };
  // --- End Array Implementations --- //
} // namespace ncarray

#endif // NCARRAY_POLICIES_HH
