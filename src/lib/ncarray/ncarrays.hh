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

  EXTERN_HOST_VM_OPS(char, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(bool, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(std::uint8_t, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(std::uint16_t, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(std::uint32_t, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(std::uint64_t, NCOffsetsPolicy)

  EXTERN_HOST_VM_OPS(std::int8_t, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(std::int16_t, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(std::int32_t, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(std::int64_t, NCOffsetsPolicy)

  EXTERN_HOST_VM_OPS(float, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(double, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(long double, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(std::complex<float>, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(std::complex<double>, NCOffsetsPolicy)

  EXTERN_HOST_VM_OPS(Float2, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(Float3, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(Float4, NCOffsetsPolicy)

  EXTERN_HOST_VM_OPS(Double2, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(Double3, NCOffsetsPolicy)
  EXTERN_HOST_VM_OPS(Double4, NCOffsetsPolicy)
} // namespace ncarray

#endif // NCARRAY_NCARRAYS_HH
