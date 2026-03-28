/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_NCARRAYS_HH
#define NCARRAY_NCARRAYS_HH

#include "ncarray/array_impl.hh"
#include "ncarray/array_operations.hh"

namespace ncarray {
  using NCArrayView = ArrayImpl<NCOffsetsPolicy, ViewPolicy>;
  using NCArrayRef = ArrayImpl<NCOffsetsPolicy, RefPolicy>;
  using NCArray = ArrayImpl<NCOffsetsPolicy, OwnerPolicy>;

  extern template class ArrayImpl<NCOffsetsPolicy, ViewPolicy>;
  extern template class ArrayImpl<NCOffsetsPolicy, RefPolicy>;
  extern template class ArrayImpl<NCOffsetsPolicy, OwnerPolicy>;
} // namespace ncarray

#endif // NCARRAY_NCARRAYS_HH
