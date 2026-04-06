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
#include "ncarray/build_macro.hh"

namespace ncarray {

  using NCDevArrayView = ArrayImpl<NCOffsetsPolicy, DevViewPolicy>;
  using NCDevArrayRef = ArrayImpl<NCOffsetsPolicy, DevRefPolicy>;
  using NCDevArray = ArrayImpl<NCOffsetsPolicy, DevOwnerPolicy>;

  EXTERN_NC_BASE_OPS(NCOffsetsPolicy, DevViewPolicy)
  EXTERN_NC_BASE_OPS(NCOffsetsPolicy, DevRefPolicy)
  EXTERN_NC_BASE_OPS(NCOffsetsPolicy, DevOwnerPolicy)

  EXTERN_NC_CROSS_OPS(NCOffsetsPolicy, DevRefPolicy,   NCOffsetsPolicy, DevViewPolicy)
  EXTERN_NC_CROSS_OPS(NCOffsetsPolicy, DevOwnerPolicy, NCOffsetsPolicy, DevViewPolicy)
  EXTERN_NC_CROSS_OPS(NCOffsetsPolicy, DevOwnerPolicy, NCOffsetsPolicy, DevRefPolicy)
}

#endif // NCARRAY_NCDEVARRAYS_CUH
