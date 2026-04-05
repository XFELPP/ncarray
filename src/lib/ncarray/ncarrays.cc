/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/ncarrays.hh"
#include "ncarray/array_operations.hh"
#include "ncarray/build_macro.hh"

INSTANTIATE_NC_BASE_OPS(NCOffsetsPolicy, ViewPolicy)
INSTANTIATE_NC_BASE_OPS(NCOffsetsPolicy, RefPolicy)
INSTANTIATE_NC_BASE_OPS(NCOffsetsPolicy, OwnerPolicy)

INSTANTIATE_NC_CROSS_OPS(NCOffsetsPolicy, RefPolicy, NCOffsetsPolicy, ViewPolicy)
INSTANTIATE_NC_CROSS_OPS(NCOffsetsPolicy, OwnerPolicy, NCOffsetsPolicy, ViewPolicy)
INSTANTIATE_NC_CROSS_OPS(NCOffsetsPolicy, OwnerPolicy, NCOffsetsPolicy, RefPolicy)
