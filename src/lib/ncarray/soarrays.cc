/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/soarrays.hh"
#include "ncarray/array_operations.hh"
#include "ncarray/build_macro.hh"

INSTANTIATE_NC_BASE_OPS(SOArrayPolicy, ViewPolicy)
INSTANTIATE_NC_BASE_OPS(SOArrayPolicy, RefPolicy)
INSTANTIATE_NC_BASE_OPS(SOArrayPolicy, OwnerPolicy)

INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, RefPolicy,   SOArrayPolicy, ViewPolicy)
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, OwnerPolicy, SOArrayPolicy, ViewPolicy)
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, OwnerPolicy, SOArrayPolicy, RefPolicy)

INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, ViewPolicy,  SOArrayPolicy, RefPolicy)
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, ViewPolicy,  SOArrayPolicy, OwnerPolicy)
INSTANTIATE_NC_CROSS_OPS(SOArrayPolicy, RefPolicy,   SOArrayPolicy, OwnerPolicy)
