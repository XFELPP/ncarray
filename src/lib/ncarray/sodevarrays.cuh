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
#include "ncarray/build_macro.hh"

namespace ncarray {
  using SODevArrayView = ArrayImpl<SOArrayPolicy, DevViewPolicy>;
  using SODevArrayRef = ArrayImpl<SOArrayPolicy, DevRefPolicy>;
  using SODevArray = ArrayImpl<SOArrayPolicy, DevOwnerPolicy>;

  EXTERN_NC_BASE_OPS(SOArrayPolicy, DevViewPolicy)
  EXTERN_NC_BASE_OPS(SOArrayPolicy, DevRefPolicy)
  EXTERN_NC_BASE_OPS(SOArrayPolicy, DevOwnerPolicy)

  EXTERN_NC_CROSS_OPS(SOArrayPolicy, DevRefPolicy,   SOArrayPolicy, DevViewPolicy)
  EXTERN_NC_CROSS_OPS(SOArrayPolicy, DevOwnerPolicy, SOArrayPolicy, DevViewPolicy)
  EXTERN_NC_CROSS_OPS(SOArrayPolicy, DevOwnerPolicy, SOArrayPolicy, DevRefPolicy)
}

#endif // NCARRAY_SODEVARRAYS_HH
