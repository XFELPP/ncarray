/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_NCDEVARRAYS_CUH
#define NCARRAY_NCDEVARRAYS_CUH

#include "ncarray/array_impl.hh"
#include "ncarray/array_operations.hh"

namespace ncarray {
  using NCDevArrayView = ArrayImpl<NCOffsetsPolicy, DevViewPolicy>;
  using NCDevArrayRef = ArrayImpl<NCOffsetsPolicy, DevRefPolicy>;
  using NCDevArray = ArrayImpl<NCOffsetsPolicy, DevOwnerPolicy>;

  extern template class ArrayImpl<NCOffsetsPolicy, DevViewPolicy>;
  extern template class ArrayImpl<NCOffsetsPolicy, DevRefPolicy>;
  extern template class ArrayImpl<NCOffsetsPolicy, DevOwnerPolicy>;
} // namespace ncarray

#endif // NCARRAY_NCDEVARRAYS_CUH
