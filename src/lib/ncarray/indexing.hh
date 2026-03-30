/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_INDEXING_HH
#define NCARRAY_INDEXING_HH

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <concepts>

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

namespace ncarray {
  struct Ellipsis {};

  struct Slice {
    Slice(ssize_t start_, ssize_t stop_)
      : start(start_)
      , stop(stop_)
      , step(1)
      , length((stop - start + step - 1) / step)
    {}

    Slice(ssize_t start_, ssize_t stop_, ssize_t step_)
      : start(start_)
      , stop(stop_)
      , step(step_)
      , length((stop - start + step - 1) / step)
    {}

    ssize_t start;
    ssize_t stop;
    ssize_t step;
    ssize_t length;
  };

  // Indexing is allowed by integer, slice or ellipsis
  template <typename IdxT>
  concept IndexArg = std::integral<std::decay_t<IdxT>> ||
    std::same_as<std::decay_t<IdxT>, Slice> ||
    std::same_as<std::decay_t<IdxT>, Ellipsis>;

  enum class IndexType { Integer, Slice, Ellipsis };

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

} // namespace ncarray
#endif // NCARRAY_INDEXING_HH
