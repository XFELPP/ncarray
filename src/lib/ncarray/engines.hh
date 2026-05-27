/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_ENGINES_HH
#define NCARRAY_ENGINES_HH

#ifdef __CUDACC__
#include "ncarray/engines/gpuengine.hh"
#endif
#include "ncarray/engines/hostengine.hh"

#endif // NCARRAY_ENGINES_HH
