/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/ncdevarrays.cuh"

#include "ncarray/array_operations.hh"
#include "ncarray/build_macro.hh"

INSTANTIATE_NC_BASE_OPS(NCOffsetsPolicy, DevRefPolicy)

// --- GPU <-> GPU Ops --- //
INSTANTIATE_NC_CROSS_OPS(NCOffsetsPolicy, DevRefPolicy,   NCOffsetsPolicy, DevViewPolicy)
INSTANTIATE_NC_CROSS_OPS(NCOffsetsPolicy, DevRefPolicy,   NCOffsetsPolicy, DevOwnerPolicy)

// --- GPU -> Host Ops --- //
INSTANTIATE_NC_CROSS_OPS(NCOffsetsPolicy, DevRefPolicy,   NCOffsetsPolicy, ViewPolicy)
INSTANTIATE_NC_CROSS_OPS(NCOffsetsPolicy, DevRefPolicy,   NCOffsetsPolicy, RefPolicy)
INSTANTIATE_NC_CROSS_OPS(NCOffsetsPolicy, DevRefPolicy,   NCOffsetsPolicy, OwnerPolicy)

// --- Host -> GPU Ops --- //
INSTANTIATE_NC_CROSS_OPS(NCOffsetsPolicy, RefPolicy,      NCOffsetsPolicy, DevViewPolicy)
INSTANTIATE_NC_CROSS_OPS(NCOffsetsPolicy, RefPolicy,      NCOffsetsPolicy, DevRefPolicy)
INSTANTIATE_NC_CROSS_OPS(NCOffsetsPolicy, RefPolicy,      NCOffsetsPolicy, DevOwnerPolicy)
