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
#include "ncarray/build_macro.hh"

namespace ncarray {
  using SOArrayView = ArrayImpl<SOArrayPolicy, ViewPolicy>;
  using SOArrayRef = ArrayImpl<SOArrayPolicy, RefPolicy>;
  using SOArray = ArrayImpl<SOArrayPolicy, OwnerPolicy>;

  EXTERN_NC_BASE_OPS(SOArrayPolicy, ViewPolicy)
  EXTERN_NC_BASE_OPS(SOArrayPolicy, RefPolicy)
  EXTERN_NC_BASE_OPS(SOArrayPolicy, OwnerPolicy)

  EXTERN_NC_CROSS_OPS(SOArrayPolicy, RefPolicy,   SOArrayPolicy, ViewPolicy)
  EXTERN_NC_CROSS_OPS(SOArrayPolicy, OwnerPolicy, SOArrayPolicy, ViewPolicy)
  EXTERN_NC_CROSS_OPS(SOArrayPolicy, OwnerPolicy, SOArrayPolicy, RefPolicy)

  EXTERN_NC_TERNARY_OPS(SOArrayPolicy, ViewPolicy,  SOArrayPolicy, ViewPolicy, SOArrayPolicy, ViewPolicy)
  EXTERN_NC_TERNARY_OPS(SOArrayPolicy, RefPolicy,   SOArrayPolicy, ViewPolicy, SOArrayPolicy, ViewPolicy)
  EXTERN_NC_TERNARY_OPS(SOArrayPolicy, OwnerPolicy, SOArrayPolicy, ViewPolicy, SOArrayPolicy, ViewPolicy)
} // namespace ncarray

#endif // NCARRAY_SOARRAYS_HH
