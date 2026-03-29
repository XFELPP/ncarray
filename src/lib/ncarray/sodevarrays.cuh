/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_SODEVARRAYS_HH
#define NCARRAY_SODEVARRAYS_HH

#include "ncarray/array_impl.hh"
#include "ncarray/array_operations.hh"

namespace ncarray {
  using SODevArrayView = ArrayImpl<SOArrayPolicy, DevViewPolicy>;
  using SODevArrayRef = ArrayImpl<SOArrayPolicy, DevRefPolicy>;
  using SODevArray = ArrayImpl<SOArrayPolicy, DevOwnerPolicy>;

  extern template class ArrayImpl<SOArrayPolicy, DevViewPolicy>;
  extern template class ArrayImpl<SOArrayPolicy, DevRefPolicy>;
  extern template class ArrayImpl<SOArrayPolicy, DevOwnerPolicy>;
} // namespace ncarray

#endif // NCARRAY_SODEVARRAYS_HH
