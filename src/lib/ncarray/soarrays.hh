/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_SOARRAYS_HH
#define NCARRAY_SOARRAYS_HH

#include "ncarray/array_impl.hh"
#include "ncarray/array_operations.hh"

namespace ncarray {
  using SOArrayView = ArrayImpl<SOArrayPolicy, ViewPolicy>;
  using SOArrayRef = ArrayImpl<SOArrayPolicy, RefPolicy>;
  using SOArray = ArrayImpl<SOArrayPolicy, OwnerPolicy>;

  extern template class ArrayImpl<SOArrayPolicy, ViewPolicy>;
  extern template class ArrayImpl<SOArrayPolicy, RefPolicy>;
  extern template class ArrayImpl<SOArrayPolicy, OwnerPolicy>;
} // namespace ncarray

#endif // NCARRAY_SOARRAYS_HH
