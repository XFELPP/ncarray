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
#include "ncarray/build_macro.hh"

namespace ncarray {
  using NCArrayView = ArrayImpl<NCOffsetsPolicy, ViewPolicy>;
  using NCArrayRef = ArrayImpl<NCOffsetsPolicy, RefPolicy>;
  using NCArray = ArrayImpl<NCOffsetsPolicy, OwnerPolicy>;

  EXTERN_NC_BASE_OPS(NCOffsetsPolicy, ViewPolicy)
  EXTERN_NC_BASE_OPS(NCOffsetsPolicy, RefPolicy)
  EXTERN_NC_BASE_OPS(NCOffsetsPolicy, OwnerPolicy)

  EXTERN_NC_CROSS_OPS(NCOffsetsPolicy, RefPolicy, NCOffsetsPolicy, ViewPolicy)
  EXTERN_NC_CROSS_OPS(NCOffsetsPolicy, OwnerPolicy, NCOffsetsPolicy, ViewPolicy)
  EXTERN_NC_CROSS_OPS(NCOffsetsPolicy, OwnerPolicy, NCOffsetsPolicy, RefPolicy)

  EXTERN_NC_TERNARY_OPS(NCOffsetsPolicy, ViewPolicy, NCOffsetsPolicy, ViewPolicy, NCOffsetsPolicy, ViewPolicy)
  EXTERN_NC_TERNARY_OPS(NCOffsetsPolicy, RefPolicy, NCOffsetsPolicy, ViewPolicy, NCOffsetsPolicy, ViewPolicy)
  EXTERN_NC_TERNARY_OPS(NCOffsetsPolicy, OwnerPolicy, NCOffsetsPolicy, ViewPolicy, NCOffsetsPolicy, ViewPolicy)
} // namespace ncarray

#endif // NCARRAY_NCARRAYS_HH
