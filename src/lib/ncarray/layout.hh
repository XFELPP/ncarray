/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_LAYOUT_HH
#define NCARRAY_LAYOUT_HH

#include "ncarray/indexing.hh"
#include "ncarray/storage.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <concepts>
#include <cstdint>

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

  /**
   * For simplifying CPU/GPU compatibility, STL containers are replaced with a
   * fixed struct for expressing metadata surrounding arrays. E.g. shape and
   * strides information. Currently, ncarray supports up to 10 dimensions in an
   * array.
   *
   * The FixedMetadata struct is a small component which is used for the holding of
   * the above metadata. It has a data member (which will be the shape data, e.g.)
   * and also holds its dimensionality in `ndim`. This struct is templated so the
   * underlying type for the metadata arrays can be changed.
   */
  template <typename T, ssize_t MaxNDim = NCARRAY_MAX_NDIM>
  struct FixedMetadata {
  public:
    using value_type = T;

    T data[MaxNDim] { 0 };
    ssize_t ndim { 0 };

  public:
    NCA_HD constexpr T& operator[](ssize_t i) {
      return data[i];
    }

    NCA_HD constexpr const T& operator[](ssize_t i) const {
      return data[i];
    }

    NCA_HD inline void set(const T* vals, ssize_t n) {
      ndim = n;
      for (ssize_t i = 0; i < n; ++i) {
        data[i] = vals[i];
      }
    }
  };

  // --- Layout Policies (NCArray* vs SOArray*) --- //

  /**
   * By default, Metadata will use ssize_t which is compatible with the conventions
   * used in many places for a signed integer to describe shapes, strides, suboffsets
   * and so on.
   */
  using Metadata = FixedMetadata<ssize_t>;

  /**
   * The LayoutPolicy dictates the overall organization of an array object. E.g.
   * if it has offsets or suboffsets.
   *
   * Currently, all types of arrays inherit from the base LayoutPolicy which gives
   * them a shape and stride. Specializations can provide additional features.
   *
   * All subclasses must implement the `advance` function which is the *key*
   * mechanism dictating how the array's layout can be traversed.
   */
  template <typename Derived>
  struct LayoutPolicy {
    /**
     * Helper function to determine whether a given axis contains pointers.
     * If true, this axis is a "double pointer" or involves suboffsets, etc.
     * depending on the specialization.
     */
    NCA_HD inline bool is_pointer_axis(ssize_t axis) const {
      if constexpr (requires { static_cast<const Derived*>(this)->is_pointer_impl(axis); }) {
        return static_cast<const Derived*>(this)->is_pointer_impl(axis);
      }
      return false;
    }

    /**
     * Check if any pointer axes exist.
     *
     * For simplicity a loop is used here. NCOffsets could optimize, but its a fairly
     * fast operation anyway.
     */
    NCA_HD inline bool is_contiguous_impl() const {
      if constexpr (requires { static_cast<const Derived*>(this)->is_contiguous_impl(); }) {
        return static_cast<const Derived*>(this)->is_contiguous_impl();
      }
      return false;
    }

    /**
     * The advance function takes a pointer to the current position in the array
     * as well as an axis and index. Using this information, and the knowledge
     * of the array layout it will move the pointer forward one data unit,
     * accounting for strides, offsets, pointer jumps and so on.
     *
     * @param[in] data The pointer to the current position in the array.
     * @param[in] axis The current axis being traversed.
     * @param[in] index The current index along the traversed axis.
     * @returns The pointer to the next item in the array.
     */
    NCA_HD inline void* advance(void* data, ssize_t axis, ssize_t index) const {
      return static_cast<const Derived*>(this)->advance(data, axis, index);
    }

    NCA_HD inline const void* advance(const void* data, ssize_t axis, ssize_t index) const {
      return static_cast<const Derived*>(this)->advance(data, axis, index);
    }

    /**
     * @returns shape The pointer to the underlying shape metadata.
     */
    NCA_HD inline const ssize_t* shape() const {
      return m_shape.data;
    }

    // TODO: Bounds check?
    /**
     * @param[in] dim A dimension to check the shape of.
     * @returns The shape of that dimension.
     */
    NCA_HD inline ssize_t shape(ssize_t dim) const {
      return m_shape[dim];
    }

    /**
     * @returns The total number of dimensions.
     */
    NCA_HD inline ssize_t ndim() const {
      return m_shape.ndim;
    }

    /**
     * @returns The total number of elements in the array.
     */
    NCA_HD inline ssize_t size() const {
      ssize_t s { 1 };
      for (ssize_t i = 0; i < m_shape.ndim; ++i) {
        s *= m_shape[i];
      }
      return s;
    }

    /**
     * @returns The pointer to the underlying strides metadata.
     */
    NCA_HD inline const ssize_t* strides() const {
      return m_strides.data;
    }

    // TODO: Bounds check?
    /**
     * @param[in] dim A dimension to check the stride for.
     * @returns The stride along that dimension.
     */
    NCA_HD inline ssize_t stride(ssize_t dim) const {
      return m_strides[dim];
    }

    /**
     * The repr functions return a string to identify the storage policy when
     * writing out string representations of the array.
     */
    NCA_HD inline const char* layout_repr() const {
      return static_cast<Derived*>(this)->layout_repr();
    }

    /**
     * Provided a set of indexing arguments construct the description of the new
     * axes given the layout specification.
     *
     * @param[out] new_axes A set of AxisDescr struct length NCARRAY_MAX_NDIM to
     *             build the descriptions into.
     * @param[in] indices The indices, consisting of collections of integers, slices and
     *             ellipsis in any combination.
     * @param[in] num_indices The total number of indices provided.
     */
    NCA_HD void build_new_axes(AxisDescr* new_axes,
                               const IndexItem* indices,
                               ssize_t num_indices) const {
      ssize_t axis { 0 };

      for (ssize_t i = 0; i < num_indices; ++i) {
        const auto& idx_item = indices[i];

        if (idx_item.type == IndexType::Ellipsis) {
          ssize_t jump = static_cast<const Derived*>(this)->ndim() - num_indices + 1;

          for (ssize_t a = 0; a < jump; ++a) {
            ssize_t dim = axis + a;

            ssize_t off_val = static_cast<const Derived*>(this)->get_offset(dim);

            new_axes[dim] = {
              dim,
              static_cast<const Derived*>(this)->m_shape[dim],
              static_cast<const Derived*>(this)->m_strides[dim],
              off_val,
              static_cast<const Derived*>(this)->is_pointer_axis(dim),
              /*collapsed=*/false,
              /*data_shift=*/0
            };
          }
          axis += jump;
        } else {
          ssize_t length { 1 };
          ssize_t stride { static_cast<const Derived*>(this)->m_strides[axis] };
          ssize_t offset { 0 };
          bool is_pointer { static_cast<const Derived*>(this)->is_pointer_axis(axis) };
          bool collapsed { false };
          ssize_t data_shift { 0 };

          if (idx_item.type == IndexType::Integer) {
            collapsed = true;
            ssize_t idx = idx_item.idx;
            if (idx < 0) {
              idx += static_cast<const Derived*>(this)->m_shape[axis];
            }

            // The advance function for the layout will handle strides and offsets
            // This allows different layouts to adjust appropriately.
            offset = idx;
          } else if (idx_item.type == IndexType::Slice) {
            ssize_t start = idx_item.slice.start;
            ssize_t stop = idx_item.slice.stop;
            ssize_t step = idx_item.slice.step;
            length = (stop - start + step - 1) / step;
            if (length < 0) {
              length = 0;
            }

            if (start < 0) {
              start += static_cast<const Derived*>(this)->m_shape[axis];
            }
            if (stop < 0) {
              stop += static_cast<const Derived*>(this)->m_shape[axis];
            }

            stride *= step;

            if constexpr (requires { static_cast<const Derived*>(this)->offsets(); }) {
              offset =
                static_cast<const Derived*>(this)->get_offset(axis) +
                start * static_cast<const Derived*>(this)->m_strides[axis];

              data_shift = 0;
            } else if constexpr (requires { static_cast<const Derived*>(this)->suboffsets(); }) {
              offset = static_cast<const Derived*>(this)->get_offset(axis);
              data_shift = start * static_cast<const Derived*>(this)->m_strides[axis];
            }
          }

          new_axes[axis] = {
            axis,
            length,
            stride,
            offset,
            is_pointer,
            collapsed,
            data_shift
          };

          axis++;
        }
      }

      for (; axis < static_cast<const Derived*>(this)->ndim(); ++axis) {
        ssize_t off_val = static_cast<const Derived*>(this)->get_offset(axis);

        new_axes[axis] = {
          axis,
          static_cast<const Derived*>(this)->m_shape[axis],
          static_cast<const Derived*>(this)->m_strides[axis],
          off_val,
          static_cast<const Derived*>(this)->is_pointer_axis(axis),
          /*collapsed=*/false,
          /*data_shift=*/0
        };
      }
    }

    NCA_HD inline Metadata::value_type get_offset(ssize_t axis) const {
      return static_cast<const Derived*>(this)->get_offset_impl(axis);
    }

  protected:
    Metadata m_shape;
    Metadata m_strides;
    // Pointer axis is provided for simplicity of interfaces only.
    // It may not be used by all subclasses.
    Metadata::value_type m_pointer_axis { -1 };
  };

  /**
   * The NCOffsetsPolicy provides an additional offsets attribute and allows for a
   * single pointer axis.
   */
  struct NCOffsetsPolicy : public LayoutPolicy<NCOffsetsPolicy> {
  public:
    /**
     * For NCArray* type arrays, if the current axis is a pointer axis, then
     * traversal requires interpreting the data as a double pointer and indexing
     * appropriately (with an offset, if slicing has selected a sub-view).
     */
    NCA_HD inline void* advance(void* data, ssize_t axis, ssize_t index) const {
      if (axis == this->m_pointer_axis) {
        return reinterpret_cast<void**>(data)[index + m_offsets[axis]];
      }

      return reinterpret_cast<std::uint8_t*>(data) + index * m_strides[axis] + m_offsets[axis];
    }

    NCA_HD inline const void* advance(const void* data,
                                      ssize_t axis,
                                      ssize_t index) const {
      if (axis == this->m_pointer_axis) {
        return reinterpret_cast<const void* const *>(data)[index + m_offsets[axis]];
      }

      return
        reinterpret_cast<const std::uint8_t*>(data) + index * m_strides[axis] + m_offsets[axis];
    }

    NCA_HD inline const ssize_t* offsets() const {
      return m_offsets.data;
    }
    NCA_HD inline ssize_t offset(ssize_t dim) const {
      return m_offsets[dim];
    }

    NCA_HD inline bool is_pointer_impl(ssize_t axis) const {
      return axis == this->m_pointer_axis;
    }

    NCA_HD inline Metadata::value_type get_offset_impl(ssize_t axis) const {
      return this->m_offsets[axis];
    }

    NCA_HD inline const char* layout_repr() const { return "NCArray"; }

    NCA_HD inline bool is_contiguous_impl() const {
      for (ssize_t i = 0; i < this->ndim(); ++i) {
        if (this->m_offsets[i] != 0) {
          return false;
        }
      }
      return true;
    }

  protected:
    Metadata m_offsets;
  };

  /**
   * The SOArrayPolicy implements PEP3118 compliance with the addition of a
   * suboffsets field.
   */
  struct SOArrayPolicy : public LayoutPolicy<SOArrayPolicy> {
  public:
    /**
     * The Metadata struct defaults values to 0 - this works for NCArray.
     * For suboffsets based classes, we want the value to be -1.
     * Flipping the suboffset back on requires explicitly setting it.
     */
    NCA_HD SOArrayPolicy() {
      for (ssize_t i = 0; i < NCARRAY_MAX_NDIM; ++i) {
        this->m_suboffsets[i] = -1;
      }
    }

    SOArrayPolicy(const SOArrayPolicy& other) = default;
    SOArrayPolicy(SOArrayPolicy&& other) noexcept = default;
    SOArrayPolicy& operator=(const SOArrayPolicy& other) = default;
    SOArrayPolicy& operator=(SOArrayPolicy&& other) = default;

    /**
     * Via the revised buffer protocol specification in PEP3118, a "suboffset"
     * greater than or equal to 0 indicates that the value on that axis is a pointer.
     * The specific value dictates how many bytes to add to it AFTER dereferencing.
     * Negative suboffsets mean no-pointer axis, and no special dereferencing
     */
    NCA_HD inline void* advance(void* data, ssize_t axis, ssize_t index) const {
      std::uint8_t* next = reinterpret_cast<std::uint8_t*>(data) + index * m_strides[axis];
      if (m_suboffsets[axis] >= 0) {
        next = *reinterpret_cast<std::uint8_t**>(next) + m_suboffsets[axis];
      }
      return reinterpret_cast<void*>(next);
    }

    NCA_HD inline const void* advance(const void* data,
                                      ssize_t axis,
                                      ssize_t index) const {
      const std::uint8_t* next =
        reinterpret_cast<const std::uint8_t*>(data) + index * m_strides[axis];
      if (m_suboffsets[axis] >= 0) {
        next = *reinterpret_cast<const std::uint8_t* const*>(next) + m_suboffsets[axis];
      }
      return reinterpret_cast<const void*>(next);
    }

    NCA_HD inline const ssize_t* suboffsets() const {
      return m_suboffsets.data;
    }

    NCA_HD inline ssize_t suboffset(ssize_t dim) const {
      return m_suboffsets[dim];
    }

    NCA_HD inline bool is_pointer_impl(ssize_t axis) const {
      return m_suboffsets[axis] >= 0;
    }

    NCA_HD inline Metadata::value_type get_offset_impl(ssize_t axis) const {
      return this->m_suboffsets[axis];
    }

    NCA_HD inline const char* layout_repr() const { return "SOArray"; }

    NCA_HD inline bool is_contiguous_impl() const {
      for (ssize_t i = 0; i < this->ndim(); ++i) {
        if (this->m_suboffsets[i] >= 0) {
          return false;
        }
      }
      return true;
    }

  protected:
    Metadata m_suboffsets;
  };

  // --- End Layout Policies --- //

  // Setup simple concept to support const and non-const iterators
  template <typename T>
  concept ViewLike = std::is_base_of_v<ViewPolicy, std::remove_const_t<T>>;

} // namespace ncarray

#endif // NCARRAY_LAYOUT_HH
