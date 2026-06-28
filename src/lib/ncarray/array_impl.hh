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
#include "ncarray/expression/interface.hh"
#include "ncarray/expression/mvnode.hh"
#include "ncarray/indexing.hh"
#include "ncarray/layout.hh"
#include "ncarray/op_traits.hh"
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

  /**
   * Calculate the strides for a standard C-order (row-major) contiguous array of given shape.
   *
   * @note Strides are always calculated and stored as bytes.
   *
   * @param[in] shape The shape to calculate strides for.
   * @param[in] itemsize The size in bytes of a single element of the array.
   * @returns The C-order calculated strides.
   */
  inline std::vector<ssize_t> calculate_c_order_strides(const std::vector<ssize_t>& shape,
                                                        const ssize_t itemsize) {
    size_t ndim { shape.size() };
    std::vector<ssize_t> strides(ndim, itemsize);
    for (ssize_t dim = ndim - 2; dim >= 0; --dim) {
      strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    return strides;
  }

  /**
   * Calculate the strides for a standard C-order (row-major) contiguous array of given shape.
   *
   * @note Strides are always calculated and stored as bytes.
   *
   * @param[in] The number of dimensions in the array.
   * @param[in] shape The shape to calculate strides for (pointer to first element).
   * @param[in] itemsize The size in bytes of a single element of the array.
   * @returns The C-order calculated strides.
   */
  inline std::vector<ssize_t> calculate_c_order_strides(const ssize_t ndim,
                                                        const ssize_t* shape,
                                                        const ssize_t itemsize) {
    std::vector<ssize_t> strides(ndim, itemsize);
    for (ssize_t dim = ndim - 2; dim >= 0; --dim) {
      strides[dim] = strides[dim + 1] * shape[dim + 1];
    }
    return strides;
  }

  /**
   * Calculate the strides for a standard C-order (row-major) contiguous array of given shape.
   *
   * @note Strides are always calculated and stored as bytes.
   *
   * @param[in] The number of dimensions in the array.
   * @param[in] shape The shape to calculate strides for (pointer to first element).
   * @param[in] itemsize The size in bytes of a single element of the array.
   * @param[out] strides The C-order calculated strides will be output in this pointer.
   */
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
    // @note Copys can only be done for Owner types on the host.
    // @note The Storage(Storage&) constructor cannot be used since some
    // policies (e.g. Owner) have no default - it cannot be defined since
    // knowing how much memory to allocate relies on information from Layout
    // This means that these constructors MUST be correct and initialize everything

    // --- Host-only copy constructor for owner types --- //
#ifndef __CUDACC_RTC__
    /**
     * Copy constructor for owner type arrays.
     *
     * Unlike the constructors that follow, this is a deep copy. The actual data
     * will be copied!
     *
     * @note This constructor is NOT compatible with device code.
     *       This is a host-only constructor.
     */
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

    /**
     * Move constructor for owner type arrays.
     *
     * @note This constructor is NOT compatible with device code.
     *       This is a host-only constructor.
     */
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
    /**
     * Copy constructor for view/reference type interconversion.
     *
     * @note If copying a reference type, the underlying data is not copied; however,
     * the hosted reference pointers are.
     *
     * @note This constructor is compatible with device code.
     */
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

    /**
     * Move constructor for view/reference type interconversion.
     *
     * @note This constructor is compatible with device code.
     */
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
    /**
     * Universal copy to view constructor.
     *
     * All array's can be copy constructed as their corresponding view.
     *
     * @note As the destination is a constructed view, this constructor does NOT
     *       copy the underlying data. Only the array metadata.
     *
     * @note This constructor is compatible with device code.
     */
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

    /**
     * Universal move to view constructor.
     *
     * All array's can be move constructed as their corresponding view.
     *
     * @note This constructor is compatible with device code.
     */
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

    /**
     * Copy constructor for Owner type arrays.
     *
     * @note This constructor is NOT compatible with device code.
     *       This is a host-only constructor.
     */
    ArrayImpl& operator=(const ArrayImpl& other)
      requires is_base_of_v<OwnerTag, Storage>
    {
      if (this != &other) {
        *this = ArrayImpl(other);
      }
      return *this;
    }

    /**
     * Move constructor for Owner type arrays.
     *
     * @note This constructor is NOT compatible with device code.
     *       This is a host-only constructor.
     */
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
    /**
     * Copy constructor for non-Owner type arrays.
     *
     * @note This constructor is compatible with device code.
     */
    NCA_HD ArrayImpl& operator=(const ArrayImpl& other)
      requires (!is_base_of_v<OwnerTag, Storage>)
    {
      if (this != &other) {
        *this = ArrayImpl(other);
      }

      return *this;
    }

    /**
     * Move constructor for non-Owner type arrays.
     *
     * @note This constructor is compatible with device code.
     */
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

    /**
     * Construct a new array (generally view type). This constructor allows passing
     * explicit offsets/suboffsets.
     *
     * @note This constructor is compatible with device code.
     *
     * @param[in] data_ The underlying array data.
     * @param[in] ndim The number of dimensions in the array.
     * @param[in] shape_ A pointer to the shape. Should be valid through `ndim` derefernces.
     * @param[in] strides_ A pointer to the strides. Should be valid through `ndim` derefernces.
     * @param[in] offsets_ A pointer to the offsets. Should be valid through `ndim` derefernces.
     * @param[in] dtype_ The datatype of the elements of the new array.
     * @param[in] pointer_axis_ Which if any axis is a pointer axis.
     * @param[in] read_only_ Whether the underlying data is read-only.
     */
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

    /**
     * Construct a new array (generally view type).
     *
     * @note This constructor is compatible with device code.
     *
     * @param[in] data_ The underlying array data.
     * @param[in] ndim The number of dimensions in the array.
     * @param[in] shape_ A pointer to the shape. Should be valid through `ndim` derefernces.
     * @param[in] strides_ A pointer to the strides. Should be valid through `ndim` derefernces.
     * @param[in] dtype_ The datatype of the elements of the new array.
     * @param[in] pointer_axis_ Which if any axis is a pointer axis.
     * @param[in] read_only_ Whether the underlying data is read-only.
     */
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

    /**
     * Construct a new reference-style array given a shape and datatype.
     *
     * @note This constructor CANNOT be called from device code.
     *
     * @param[in] data_ptrs A vector of data pointers that makeup the array.
     * @param[in] shape_ The shape of the new ref array (using a std::vector).
     * @param[in] strides_ The strides of the new ref array (using a std::vector).
     * @param[in] dtype_ The datatype of the elements of the new array.
     * @param[in] pointer_axis_ Which if any axis is a pointer axis.
     * @param[in] read_only_ Whether the underlying data is read-only.
     */
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
     * @note At least for the foreseeable future, this constructor cannot be
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
     * @note At least for the foreseeable future, this constructor cannot be
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
     * @note At least for the foreseeable future, this constructor cannot be
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

    /**
     * Expression-materializing constructor.
     *
     * An array constructed via an expression object will have shape, and metadata
     * determined by the expression. The expression will be evaluated into the array.
     *
     * @note This constructor is NOT compatible with device code.
     *       This is a host only constructor as it implies allocation.
     *
     * @tparam Expr The type of the expression to be evaluated.
     * @param expr The expression to be evaluated.
     */
    template <Expression Expr>
    ArrayImpl(const Expr& expr)
      : ArrayImpl(expr.ndim(), expr.shape(), expr.dtype())
    {
      *this = expr;
    }

    /**
     * The total number of bytes contained in the array (shape * itemsize).
     *
     * @returns The number of bytes.
     */
    NCA_HD ssize_t nbytes() const {
      return this->size() * this->itemsize();
    }

    // --- Int/slice/ellipsis variadic indexing to view --- //

#if __cplusplus > 202002L

    /**
     * Variadic, multi-element operator[] for indexing to an array view.
     *
     * @note This overload requires C++23 multi-argument operator[] support and is
     *       thus currently host-only.
     *
     * Index arguments must be either:
     * - Integers
     * - Slice elements
     * - Ellipsis place holders.
     *
     * @returns The subview array determined by the provided indices.
     */
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

    /**
     * Variadic, multi-element operator() for indexing to an array view.
     *
     * @note This overload is supported in both host and device code.
     *
     * Index arguments must be either:
     * - Integers
     * - Slice elements
     * - Ellipsis place holders.
     *
     * @returns The subview array determined by the provided indices.
     */
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

    // --- Indexing to reference (non-const and const) --- //

    NCA_HD ArrayElementProxy operator[](ssize_t idx) {
      void* out_data = const_cast<void*>(this->data());
      ssize_t lin_idx { idx };

#if defined(__CUDA_ARCH__)
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

#if defined(__CUDA_ARCH__)
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

    /**
     * Convert a linearized ravel index into a multi-dimensional coords struct.
     *
     * @note The coords object must be correctly sized for the array! Dimensionality
     * is not checked in this function, as this allows for efficient compiler optimizations
     * in hot loops.
     *
     * @tparam Coords A specialization of the StaticCoords object.
     * @param[in] idx The linearized input index.
     * @param[out] coords The pre-created coords struct to be populated.
     */
    template <typename Coords>
    NCA_HD void lin_to_md(ssize_t idx, Coords& coords) {
      ssize_t lin_idx { idx };

#if defined(__CUDA_ARCH__)
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

    /**
     * Convert a multi-dimensional coords struct to a linearized index.
     *
     * @note The coords object must be correctly sized for the array! Dimensionality
     * is not checked in this function, as this allows for efficient compiler optimizations
     * in hot loops.
     *
     * @tparam Coords A specialization of the StaticCoords object.
     * @param[in] coords The pre-created coords struct with multi-dimensional indices.
     * @returns idx The linearized index.
     */
    template <typename Coords>
    NCA_HD ssize_t md_to_lin(Coords& coords) {
      ssize_t lin_idx { 0 };
      ssize_t cum_stride { 1 };

      for (auto dim = coords.size() - 1; dim >= 0; --dim) {
        lin_idx += static_cast<ssize_t>(coords[dim]) * cum_stride;
        cum_stride *= this->shape(dim);
      }

      return lin_idx;
    }

    /**
     * A single Coords struct operator[] overload to index to proxy reference.
     *
     * @note This overload is supported in both host and device code.
     *
     * The StaticCoords object should be the correct dimensionality to match the array's
     * dimensionality. This is NOT checked in this function for performance reasons!
     *
     * @tparam Coords The type of the StaticCoords object (templated on dimension and integer width.)
     * @param coords The coords to use to index to proxy reference.
     * @returns The proxy reference pointed to by the provided coords.
     */
    template <typename Coords>
    NCA_HD ArrayElementProxy operator[](const Coords& coords) {
      void* out_data = const_cast<void*>(this->data());

      for (auto dim = 0; dim < coords.size(); ++dim) {
        auto local_idx { coords[dim] };
        out_data = this->advance(out_data, dim, local_idx);
      }

      return { out_data, this->dtype() };
    }

    /**
     * A single Coords struct operator[] overload to index to proxy reference.
     *
     * @note This overload is supported in both host and device code.
     *
     * The StaticCoords object should be the correct dimensionality to match the array's
     * dimensionality. This is NOT checked in this function for performance reasons!
     *
     * @tparam Coords The type of the StaticCoords object (templated on dimension and integer width.)
     * @param coords The coords to use to index to proxy reference.
     * @returns The const proxy reference pointed to by the provided coords.
     */
    template <typename Coords>
    NCA_HD const ArrayElementProxy operator[](const Coords& coords) const {
      const void* out_data = this->data();

      for (auto dim = 0; dim < coords.size(); ++dim) {
        auto local_idx { coords[dim] };
        out_data = this->advance(out_data, dim, local_idx);
      }

      return { const_cast<void*>(out_data), this->dtype() };
    }

    /**
     * The initializer list overloader are for indexing down to a proxy reference.
     *
     * @note This overload is supported in both host and device code.
     *
     * For view semantics, you can make use of the variadic operator[] (C++23) or for NVCC,
     * C++20, code, the variadic operator().
     *
     * @param coords The coords to use to index to proxy reference.
     * @returns The proxy reference pointed to by the provided coords.
     */
    NCA_HD ArrayElementProxy operator[](initializer_list<uint64_t> coords) {
      assert(coords.size() == static_cast<size_t>(this->ndim()));
      void* out_data = const_cast<void*>(this->data());

      ssize_t axis { 0 };
      for (auto l_idx : coords) {
        out_data = this->advance(out_data, axis++, l_idx);
      }

      return { out_data, this->dtype() };
    }

    /**
     * The initializer list overloader are for indexing down to a proxy reference.
     *
     * @note This overload is supported in both host and device code.
     *
     * For view semantics, you can make use of the variadic operator[] (C++23) or for NVCC,
     * C++20, code, the variadic operator().
     *
     * @param coords The coords to use to index to proxy reference.
     * @returns The proxy reference pointed to by the provided coords.
     */
    NCA_HD const ArrayElementProxy operator[](initializer_list<uint64_t> coords) const {
      assert(coords.size() == static_cast<size_t>(this->ndim()));
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
     * @tparam VT The type of the array view.
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

      // @note Passing a length 1 slice does NOT collapse/remove the axis.
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
          // @note This call is critical! It makes that correct dereferncing and
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
     *
     * @returns The view of the array.
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

    /**
     * Convert an array to a new contiguous array.
     *
     * @note This function currently ALWAYS implies a copy, and is thus inefficient if
     *       the array is already contiguous.
     *
     * @todo Consider the logic in this function to avoid a copy if the array is contiguous.
     *
     * @returns A new contiguous array with the same data.
     */
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

    /**
     * Convert an array to an array of a new DType.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] dtype_out The DType to convert the array to.
     * @returns A new contiguous array with the same data.
     */
    OwnerType astype(DType& dtype_out) const;

    /**
     * Fill all elements of an array with the provided value. The array must not be read-only.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] val The value to assign to the array elements.
     */
    void fill(Scalar val);

    /**
     * Fill all elements of an array with the elements of the input array.
     *
     * @note This function currently only supports arrays of equal shape.
     *       No broadcasting is performed.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] arr The array to use for assignment.
     */
    void assign(const ArrayLike auto& arr);

    /**
     * Return a new array view with axes of size 1 "squeezed" (removed).
     *
     * @returns The squeezed output array view.
     */
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
     * @returns The reshaped view.
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
     * @returns A new contiguous owning array of the specified shape.
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

    /**
     * Perform a sum along the set of provided axes.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] axes The axes to sum along.
     * @returns An owning type array with the per-axis sums.
     */
    OwnerType sum(const std::vector<ssize_t>& axes) const;

    /**
     * Calculate the maximum along the set of provided axes.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] axes The axes to find the maximum along.
     * @returns An owning type array with the per-axis maxima.
     */
    OwnerType max(const std::vector<ssize_t>& axes) const;
    /**
     * Determine the index of the maximum along the set of provided axes.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] axes The axes to find the index of the maximum along.
     * @returns An owning type array with the indices of the per-axis maxima.
     */
    OwnerType argmax(const std::vector<ssize_t>& axes) const;

    /**
     * Calculate the minimum along the set of provided axes.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] axes The axes to find the minimum along.
     * @returns An owning type array with the per-axis maxima.
     */
    OwnerType min(const std::vector<ssize_t>& axes) const;
    /**
     * Determine the index of the minima along the set of provided axes.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] axes The axes to find the index of the minimum along.
     * @returns An owning type array with the indices of the per-axis minima.
     */
    OwnerType argmin(const std::vector<ssize_t>& axes) const;

    /**
     * Calculate the mean along the set of provided axes.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] axes The axes to find the mean along.
     * @returns An owning type array with the indices of the per-axis means.
     */
    OwnerType mean(const std::vector<ssize_t>& axes) const;
    /**
     * Calculate the variance along the set of provided axes.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] axes The axes to find the variance along.
     * @param[in] ddof The delta degrees of freedom. Divisor in the calculation is N-ddof.
     * @returns An owning type array with the indices of the per-axis variance.
     */
    OwnerType var(const std::vector<ssize_t>& axes, ssize_t ddof = 0) const;
    /**
     * Calculate the standard deviation along the set of provided axes.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] axes The axes to find the standard deviation along.
     * @param[in] ddof The delta degrees of freedom. Divisor in the calculation is N-ddof.
     * @returns An owning type array with the indices of the per-axis standard deviation.
     */
    OwnerType std(const std::vector<ssize_t>& axes, ssize_t ddof = 0) const;

    /**
     * Return true along the provided axes if all values are "truthy."
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] axes The axes to determine truthiness along.
     * @returns An owning type array with a truthy value if all values were truthy.
     */
    OwnerType all(const std::vector<ssize_t>& axes) const;
    /**
     * Return true along the provided axes if any value is "truthy."
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] axes The axes along which to determine if any value is "truthy."
     * @returns An owning type array with a truthy value if any value was truthy.
     */
    OwnerType any(const std::vector<ssize_t>& axes) const;

    // --- Full Reductions (To Scalar) --- //

    /**
     * Perform a sum of all array elements.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @returns The full sum of the array.
     */
    Scalar sum() const;

    /**
     * Find the maximum of all array elements.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @returns The maximum of the array.
     */
    Scalar max() const;
    /**
     * Find the index of the maximum of all array elements.
     *
     * @note This function returns a linearized ravel index.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @returns The linearized ravel index for the maximum of the array.
     */
    Scalar argmax() const;

    /**
     * Find the minimum of all array elements.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @returns The minimum of the array.
     */
    Scalar min() const;
    /**
     * Find the index of the minimum of all array elements.
     *
     * @note This function returns a linearized ravel index.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @returns The linearized ravel index for the minimum of the array.
     */
    Scalar argmin() const;

    /**
     * Find the mean of all array elements.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @returns The mean of the array.
     */
    Scalar mean() const;

    /**
     * Find the variance of all array elements.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] ddof The delta degrees of freedom. Divisor in the calculation is N-ddof.
     * @returns The variance of the array.
     */
    Scalar var(ssize_t ddof = 0) const;
    /**
     * Find the standard deviation of all array elements.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] ddof The delta degrees of freedom. Divisor in the calculation is N-ddof.
     * @returns The standard deviation of the array.
     */
    Scalar std(ssize_t ddof = 0) const;

    /**
     * Return truthy if all elements of the array are truthy.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @returns Truthy if all elements are truthy, otherwise falsey.
     */
    Scalar all() const;
    /**
     * Return truthy if any element of the array is truthy.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @returns Truthy if any element of the array is truthy, otherwise falsey.
     */
    Scalar any() const;

    /**
     * Return the underlying pointer to an array element as a Scalar variant.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] ptr The pointer to an element of the array.
     * @returns A scalar for the provided array element.
     */
    Scalar get_scalar(void* ptr) const {
      auto reduce = [&]<typename T>() -> Scalar {
        return Scalar { *reinterpret_cast<T*>(ptr) };
      };

      return dispatch(this->m_dtype, reduce);
    }

    /**
     * Evaluate an expression object into the array.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @param[in] The expression to evaluate into the array.
     */
    template <Expression Expr>
    ArrayImpl& operator=(const Expr& expr);

    using ExprResult = ExprMVNode<MemType>;

    // --- Generators --- //

    /**
     * Generate an array using the APL-style index generator.
     *
     * Use the ravel index to populate the values of the array based on the shape
     * of the instance it is called on.
     *
     * @returns An expression to be evaluated which will create the ravel index array.
     */
    ExprResult iota() const;

#ifndef __CUDACC_RTC__
    /**
     * Generate an array of the specified type using the APL-style index generator.
     *
     * Use the ravel index to populate the values of the array based on the shape
     * of the instance it is called on.
     *
     * @note This function is NOT compatible with device code. (It requires allocation).
     *       This is a host only function.
     *
     * @param shape The shape of the array to construct.
     * @param dtype The datatype for the final array.
     * @returns An array with elements populated by the index generator.
     */
    static OwnerType iota(const std::vector<ssize_t>& shape, DType dtype = DType::int64) {
      OwnerType res(shape, dtype);
      res = res.iota(); // Triggers the expression engine to evaluate the IDX op
      return res;
    }
#endif // nvrtc guard

    // --- Unary operations --- //

    /**
     * Return an expression for the negation of the array.
     *
     * @returns An expression for the negation of the array.
     */
    ExprResult operator-() const;
    /**
     * Inplace increment each element of the array.
     */
    ArrayImpl& operator++();
    /**
     * Inplace decrement each element of the array.
     */
    ArrayImpl& operator--();
    /**
     * Return an expression for the logical not of the array.
     *
     * @returns An expression for the logical not of the array.
     */
    ExprResult operator!() const;

    // --- Binary operations --- //

    /**
     * Return an expression for the sum of the array and the provided expression.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to sum with the array.
     * @returns An expression for the sum of the array and the provided expression.
     */
    ExprResult operator+(const ExprResult& right) const;

    /**
     * Return an expression for the difference of an expression from the array.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to subtract from the array.
     * @returns An expression for the difference of the provided expression from the array.
     */
    ExprResult operator-(const ExprResult& right) const;

    /**
     * Return an expression for the product of the array and the provided expression.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to multiply with the array.
     * @returns An expression for the product of the array and the provided expression.
     */
    ExprResult operator*(const ExprResult& right) const;

    /**
     * Return an expression for the quotient of array over the provided expression.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression that will be the denominator.
     * @returns An expression for the quotient of array and expression.
     */
    ExprResult operator/(const ExprResult& right) const;

    /**
     * Return an expression for the remainder (modulo) of array over the provided expression.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression that will be the denominator.
     * @returns An expression for the remainder (modulo) of array and expression.
     */
    ExprResult operator%(const ExprResult& right) const;

    // --- Binary Inplace Operations --- //

    /**
     * Add an expression into an array inplace.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to sum with the array.
     */
    ArrayImpl& operator+=(const ExprResult& right);

    /**
     * Subtract an expression from an array inplace.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to subtract from the array.
     */
    ArrayImpl& operator-=(const ExprResult& right);

    /**
     * Calculate the product of an expression into an array inplace.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to multiply with the array.
     */
    ArrayImpl& operator*=(const ExprResult& right);

    /**
     * Divide an expression into an array inplace.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to divide the array by.
     */
    ArrayImpl& operator/=(const ExprResult& right);

    // --- Comparisons --- //

    /**
     * Return an expression for the elementwise equality test between the expression and array.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to compare with the array.
     * @returns An expression for the elementwise equality comparison.
     */
    ExprResult operator==(const ExprResult& right) const;

    /**
     * Return an expression for the elementwise inequality test between the expression and array.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to compare with the array.
     * @returns An expression for the elementwise inequality comparison.
     */
    ExprResult operator!=(const ExprResult& right) const;

    /**
     * Return an expression for the elementwise less than test between the expression and array.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to compare with the array.
     * @returns An expression for the elementwise less than comparison.
     */
    ExprResult operator<(const ExprResult& right) const;

    /**
     * Return an expression for the elementwise less than or equal test between the expression and array.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to compare with the array.
     * @returns An expression for the elementwise less than or equal comparison.
     */
    ExprResult operator<=(const ExprResult& right) const;

    /**
     * Return an expression for the elementwise greater than test between the expression and array.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to compare with the array.
     * @returns An expression for the elementwise greater than comparison.
     */
    ExprResult operator>(const ExprResult& right) const;

    /**
     * Return an expression for the elementwise greater than or equal test between the expression and array.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to compare with the array.
     * @returns An expression for the elementwise greater than or equal comparison.
     */
    ExprResult operator>=(const ExprResult& right) const;

    // --- Logical Operations --- //

    /**
     * Return an expression for the logical AND of the expression and array.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to calculate the logical AND of with the array.
     * @returns An expression for the logical AND.
     */
    ExprResult operator&&(const ExprResult& right) const;

    /**
     * Return an expression for the logical OR of the expression and array.
     *
     * @note This is performed elementwise.
     *
     * @param right The expression to calculate the logical OR of with the array.
     * @returns An expression for the logical OR.
     */
    ExprResult operator||(const ExprResult& right) const;

    // --- Logical Inplace Operations --- //

    /**
     * Calculate the logical AND of an expression and array in place.
     *
     * @note This is performed elementwise.
     * @note This requires datatypes convertible to bool.
     *
     * @param right The expression to calculate the logical AND of with the array.
     */
    ArrayImpl& operator&=(const ExprResult& right);

    /**
     * Calculate the logical OR of an expression and array in place.
     *
     * @note This is performed elementwise.
     * @note This requires datatypes convertible to bool.
     *
     * @param right The expression to calculate the logical OR of with the array.
     */
    ArrayImpl& operator|=(const ExprResult& right);

    // -- Iterators --- //

    /**
     * An value-based iterator for traversing an array along the first dimension.
     *
     * This iterator is in some respects atypical by standard C++ standards. However,
     * it provides a simplified API for a common operation on multi-dimensional arrays,
     * namely, iterating along an axis -- in particular, it iterates the FIRST axis.
     *
     * This iterator implementation generates sub-views by value on the fly. The
     * ArrayImpl class is rather light-weight and the views are non-owning, so this
     * strategy is appropriate. The complexity of managing a true pointer/reference
     * based iterator for array's with varying layouts of suboffsets (pointer axes)
     * outweighs the benefits in reducing the subview construction by value cost.
     *
     * @tparam VT The ArrayImpl view type.
     */
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

    /**
     * Construct a non-const Iterator pointing to the first subview of the array.
     *
     * @returns A non-const iterator pointing to the first subview of the array.
     */
    Iterator begin();
    /**
     * Construct a non-const Iterator pointing to the past the last subview of the array.
     *
     * @returns A non-const iterator pointing past the last subview of the array.
     */
    Iterator end();

    /**
     * Construct a const Iterator pointing to the first subview of the array.
     *
     * @returns A const iterator pointing to the first subview of the array.
     */
    ConstIterator begin() const;
    /**
     * Construct a const Iterator pointing to the past the last subview of the array.
     *
     * @returns A const iterator pointing past the last subview of the array.
     */
    ConstIterator end() const;

    /**
     * Return a string representation of the data elements of the array.
     *
     * This routine returns a string of data elements of the array, formatted
     * in a manner very similar to how NumPy prints its NDArrays. If there are
     * too many elements, the first 3 and last 3 are shown with the truncated
     * elements replaced by an elipsis on all axes this has occurred.
     *
     * @note If the data is resident on GPU, it will NOT be copied over to the
     *       host just for the purposes of string representation. Instead, the
     *       representation will read `[<...(on device)...>]` followed by the
     *       shape and datatype as normal.
     *
     * @note This function is NOT compatible with device code.
     *       This is a host only function.
     *
     * @returns The string representation.
     */
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
    /**
     * Return a class name primarily for the repr routine.
     *
     * This provides a unique name for the class that combines the layout and storage
     * specifications for easy identification when printing to console.
     *
     * @returns The class name.
     */
    std::string class_name() const {
      return std::string(this->layout_repr()) + std::string(this->storage_repr());
    }

    /**
     * Dispatcher for the recursive helper for repr() that handles arbitrary dimensions.
     *
     * This function only dispatches the true recurser based on the datatype.
     *
     * @param[out] oss The stream to write the array representation to.
     * @param[in] current_data A pointer to the current data element of the array.
     * @param[in] axis The current array axis being traversed.
     * @param[in] indent The current indentation level to add to the string representation.
     * @param[in] edge_items The current number of items to be shown when truncating for length.
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

    /**
     * Recursive helper for repr() that handles arbitrary dimensions.
     *
     * @tparam T The type of the array elements.
     * @param[out] oss The stream to write the array representation to.
     * @param[in] current_data A pointer to the current data element of the array.
     * @param[in] axis The current array axis being traversed.
     * @param[in] indent The current indentation level to add to the string representation.
     * @param[in] edge_items The current number of items to be shown when truncating for length.
     */
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
   *
   * @tparam MemTag The tag indicating whether array is host- or device-resident.
   * @param[in] nc The NCArray* to be converted to SOArray*.
   * @returns The SOArray* equivalent to the input `nc` array.
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

  /**
   * Given a linearized index and array, construct a StaticCoords object to index.
   *
   * Because the StaticCoords object has a constexpr size, using it for indexing
   * as opposed to linearized indices can have major performance benefits, particularly
   * for device code applications. E.g., the constexpr size generally allows loops
   * to be fully unrolled, and stack spilling to be avoided in compiled kernels.
   *
   * This routine simply converts the linearized ravel index to a StaticCoords object
   * of the correct dimensionality of the array, using a switch dispatch.
   *
   * @note That as a result of this switch, code using the function will get much
   *       larger.
   *
   * @tparam Array The type of the input array.
   * @param[in] arr The input array.
   * @param[in] idx The linearized ravel index.
   * @returns The ArrayElementProxy to the requested data element.
   */
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
