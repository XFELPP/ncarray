/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/sodevarrays.cuh"

#include "ncarray/array_operations.hh"
#include "ncarray/build_macro.hh"

INSTANTIATE_NC_BASE_OPS(SOArrayPolicy, DevOwnerPolicy)

// --- GPU <-> GPU Ops --- //
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, DevOwnerPolicy, SOArrayPolicy, DevViewPolicy)
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, DevOwnerPolicy, SOArrayPolicy, DevRefPolicy)

// --- GPU -> Host Ops --- //
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, DevOwnerPolicy, SOArrayPolicy, ViewPolicy)
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, DevOwnerPolicy, SOArrayPolicy, RefPolicy)
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, DevOwnerPolicy, SOArrayPolicy, OwnerPolicy)

// --- Host -> GPU Ops --- //
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, OwnerPolicy,    SOArrayPolicy, DevViewPolicy)
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, OwnerPolicy,    SOArrayPolicy, DevRefPolicy)
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, OwnerPolicy,    SOArrayPolicy, DevOwnerPolicy)
