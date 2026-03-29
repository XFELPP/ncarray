/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/ncdevarrays.cuh"

template class ncarray::ArrayImpl<ncarray::NCOffsetsPolicy, ncarray::DevViewPolicy>;
template class ncarray::ArrayImpl<ncarray::NCOffsetsPolicy, ncarray::DevRefPolicy>;
template class ncarray::ArrayImpl<ncarray::NCOffsetsPolicy, ncarray::DevOwnerPolicy>;
