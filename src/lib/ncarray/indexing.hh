/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_INDEXING_HH
#define NCARRAY_INDEXING_HH

#ifdef __CUDACC_RTC__
typedef long long ssize_t;

#include <cuda/std/type_traits>

namespace hd_std = cuda::std;

#else
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <type_traits>

namespace hd_std = std;

#endif

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
  /**
   * An empty struct indicating skipped axes when indexing an array.
   *
   * This struct serves the same purpose as an ellipsis (...) does when indexing
   * a NumPy array in Python.
   */
  struct Ellipsis {};

  /**
   * A struct representing a slice for indexing by a step between a start and stop index.
   *
   * The Slice represents the indexing parameters used to index the array, not the
   * resultant sub-view produced from the indexing operation.
   */
  struct Slice {
    NCA_HD Slice(ssize_t start_, ssize_t stop_)
      : start(start_)
      , stop(stop_)
      , step(1)
    {}

    NCA_HD Slice(ssize_t start_, ssize_t stop_, ssize_t step_)
      : start(start_)
      , stop(stop_)
      , step(step_)
    {}

    ssize_t start;
    ssize_t stop;
    ssize_t step;
  };

  // Indexing is allowed by integer, slice or ellipsis
  /**
   * Determines whether an object as an allowable argument to array indexing functions.
   */
  template <typename IdxT>
  concept IndexArg = hd_std::integral<hd_std::decay_t<IdxT>> ||
    hd_std::same_as<hd_std::decay_t<IdxT>, Slice>            ||
    hd_std::same_as<hd_std::decay_t<IdxT>, Ellipsis>;

  /**
   * A tag identifier for the type of indexing argument when constructing lists of args.
   */
  enum class IndexType { Integer, Slice, Ellipsis };

  /**
   * A generic indexing argument with a tag indicating the type to be used.
   */
  struct IndexItem {
    IndexType type;
    ssize_t idx;
    Slice slice { 0, 0 };

    NCA_HD IndexItem(ssize_t idx_)
      : type(IndexType::Integer)
      , idx(idx_)
    {}

    NCA_HD IndexItem(Slice s)
      : type(IndexType::Slice)
      , slice(s)
    {}

    NCA_HD IndexItem(Ellipsis)
      : type(IndexType::Ellipsis)
    {}
  };

  /**
   * A description of an axis of an array. This struct can be used for creation
   * of new views.
   */
  struct AxisDescr {
    ssize_t index;             ///< The axis index/dimension. E.g. 0 is the first axis.
    ssize_t length;            ///< The total length of the axis.
    ssize_t stride;            ///< The stride for this axis.
    ssize_t offset;            ///< The offset or suboffset for this axis.
    bool is_pointer { false }; ///< Whether the axis is a pointer axis.
    bool collapsed { false };  ///< Whether the axis has been collapsed (squeezed)
    ssize_t data_shift { 0 };  ///< An accumulator to propagate offsets for new views
  };

  /**
   * A small struct with compile-time constant size for indexing arrays.
   *
   * The constexpr size makes it far easier for the compiler to unroll loops and
   * optimize certain hot paths.
   *
   * @tparam NDIM The dimensionality of the coords object, and by proxy the array itself.
   * @tparam IndexT The indexing type. Can, e.g., change the width of the integer.
   */
  template <int NDim = 1, typename IndexT = ssize_t>
  struct StaticCoords {
    IndexT value[NDim];

    NCA_HD static constexpr int size() {
      return NDim;
    }

    NCA_HD inline IndexT& operator[](int idx) {
      return value[idx];
    }

    NCA_HD inline const IndexT& operator[](int idx) const {
      return value[idx];
    }
  };
} // namespace ncarray
#endif // NCARRAY_INDEXING_HH
