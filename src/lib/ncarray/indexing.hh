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
#include <variant>
#include <vector>

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

  using IndexVariant = std::variant<ssize_t, Slice, Ellipsis>;
  using ArrayIndices = std::vector<IndexVariant>;

  struct AxisDescr {
    ssize_t index;
    ssize_t length;
    ssize_t stride;
    ssize_t offset;
    bool is_pointer { false };
    bool collapsed { false };
  };

} // namespace ncarray
#endif // NCARRAY_INDEXING_HH
