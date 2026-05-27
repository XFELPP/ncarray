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

  EXTERN_DEV_VM_OPS(char, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(bool, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(std::uint8_t, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(std::uint16_t, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(std::uint32_t, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(std::uint64_t, NCOffsetsPolicy)

  EXTERN_DEV_VM_OPS(std::int8_t, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(std::int16_t, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(std::int32_t, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(std::int64_t, NCOffsetsPolicy)

  EXTERN_DEV_VM_OPS(float, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(double, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(long double, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(std::complex<float>, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(std::complex<double>, NCOffsetsPolicy)

  EXTERN_DEV_VM_OPS(Float2, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(Float3, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(Float4, NCOffsetsPolicy)

  EXTERN_DEV_VM_OPS(Double2, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(Double3, NCOffsetsPolicy)
  EXTERN_DEV_VM_OPS(Double4, NCOffsetsPolicy)
}

#endif // NCARRAY_NCDEVARRAYS_CUH
