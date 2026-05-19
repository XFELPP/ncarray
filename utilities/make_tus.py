# Copyright (c) 2025-2026 Gabriel Dorlhiac
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Make the translation units separated by supported data types."""

import argparse
import os
from typing import Tuple

header: str = """
/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/{include}"
#include "ncarray/array_operations.hh"
#include "ncarray/build_macro.hh"

using namespace ncarray;

"""

ArrayImpl: str = "ArrayImpl<{L}, {S}>&"
OwnerType: str = "typename ArrayImpl<{L}, {S}>::OwnerType"
ViewType: str = "typename ArrayImpl<{L}, {S}>::ViewType"
Iterator: str = "typename ArrayImpl<{L}, {S}>::Iterator"
ConstIterator: str = "typename ArrayImpl<{L}, {S}>::ConstIterator"

cpp_dtypes: Tuple[str, ...] = (
    "char",
    "std::uint8_t",
    "std::uint16_t",
    "std::uint32_t",
    "std::uint64_t",
    "std::int8_t",
    "std::int16_t",
    "std::int32_t",
    "std::int64_t",
    "float",
    "double",
    "long double",
    "bool",
    "std::complex<float>",
    "std::complex<double>",
    "std::complex<long double>",
    "Float2",
    "Float3",
    "Float4",
    "Double2",
    "Double3",
    "Double4",
)

dev_funcs: str = """
#include "ncarray/ncdevarrays.cuh"
#include "ncarray/sodevarrays.cuh"
#include "ncarray/expression/mvnode.hh"

namespace ncarray {{
  // Float32 support for both Layouts on Device
  INSTANTIATE_DEV_VM_OPS({typename}, NCOffsetsPolicy)

  INSTANTIATE_DEV_VM_OPS({typename}, SOArrayPolicy)
}}

"""

dev_red_funcs: str = """
#include "ncarray/ncdevarrays.cuh"
#include "ncarray/sodevarrays.cuh"
#include "ncarray/expression/mvnode.hh"

namespace ncarray {{
  // Float32 support for both Layouts on Device
  INSTANTIATE_DEV_REDUCTIONS({typename}, NCOffsetsPolicy)

  INSTANTIATE_DEV_REDUCTIONS({typename}, SOArrayPolicy)
}}

"""

dev_separate_red_funcs: str = """
#include "ncarray/ncdevarrays.cuh"
#include "ncarray/sodevarrays.cuh"
#include "ncarray/expression/mvnode.hh"

namespace ncarray {{
  // Float32 support for both Layouts on Device
  INSTANTIATE_DEV_VM_FULL_REDUCE({typename}, NCOffsetsPolicy, {traits})
  INSTANTIATE_DEV_VM_REDUCE({typename}, NCOffsetsPolicy, {traits})

  INSTANTIATE_DEV_VM_FULL_REDUCE({typename}, SOArrayPolicy, {traits})
  INSTANTIATE_DEV_VM_REDUCE({typename}, SOArrayPolicy, {traits})
}}

"""

dev_separate_full_red_funcs: str = """
#include "ncarray/ncdevarrays.cuh"
#include "ncarray/sodevarrays.cuh"
#include "ncarray/expression/mvnode.hh"

namespace ncarray {{
  INSTANTIATE_DEV_VM_FULL_REDUCE({typename}, {traits}, {layout})
}}

"""

dev_separate_axes_red_funcs: str = """
#include "ncarray/ncdevarrays.cuh"
#include "ncarray/sodevarrays.cuh"
#include "ncarray/expression/mvnode.hh"

namespace ncarray {{
  INSTANTIATE_DEV_VM_REDUCE({typename}, {traits}, {layout})
}}

"""

host_funcs: str = """
#include "ncarray/ncarrays.hh"
#include "ncarray/soarrays.hh"
#include "ncarray/expression/mvnode.hh"

namespace ncarray {{
  // Float32 support for both Layouts on Device
  INSTANTIATE_HOST_VM_OPS({typename}, NCOffsetsPolicy)

  INSTANTIATE_HOST_VM_OPS({typename}, SOArrayPolicy)
}}

"""

host_red_funcs: str = """
#include "ncarray/ncarrays.hh"
#include "ncarray/soarrays.hh"
#include "ncarray/expression/mvnode.hh"

namespace ncarray {{
  INSTANTIATE_HOST_REDUCTIONS({typename}, NCOffsetsPolicy)

  INSTANTIATE_HOST_REDUCTIONS({typename}, SOArrayPolicy)
}}

"""

type_aliases: Tuple[str, ...] = (
    "ch",
    "u8",
    "u16",
    "u32",
    "u64",
    "i8",
    "i16",
    "i32",
    "i64",
    "f32",
    "f64",
    "f128",
    "bool",
    "cf32",
    "cf64",
    "cf128",
    "vf2",
    "vf3",
    "vf4",
    "vd2",
    "vd3",
    "vd4",
)

reduction_traits: Tuple[str, ...] = (
    "SumTraits",
    "MaxTraits",
    "ArgmaxTraits",
    "MinTraits",
    "ArgminTraits",
    "MeanTraits",
    "VarTraits",
    "StdTraits",
    "AllTraits",
    "AnyTraits",
)

reduction_names: Tuple[str, ...] = (
    "sum",
    "max",
    "argmax",
    "min",
    "argmin",
    "mean",
    "var",
    "std",
    "all",
    "any",
)

layout_names: Tuple[str, str] = ("ncarr", "soarr")

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, help="Output directory")

    args: argparse.Namespace = parser.parse_args()

    base_dir: str = args.output
    write_host_files: bool = True
    if "tus/device" in base_dir:
        write_host_files = False

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    for idx, typename in enumerate(cpp_dtypes):
        if write_host_files:
            h_filename: str = f"vm_{type_aliases[idx]}.cc"
            h_path: str = f"{base_dir}/{h_filename}"
            print(typename)
            with open(h_path, "w") as f:
                new_str: str = host_funcs.format(typename=typename)
                print(new_str)
                f.write(new_str)
            h_red_filename: str = f"vm_{type_aliases[idx]}_reductions.cc"
            h_red_path: str = f"{base_dir}/{h_red_filename}"
            print(typename)
            with open(h_red_path, "w") as f:
                new_str: str = host_red_funcs.format(typename=typename)
                print(new_str)
                f.write(new_str)
        else:
            d_filename: str = f"vm_{type_aliases[idx]}_binops.cu"
            d_path: str = f"{base_dir}/{d_filename}"
            with open(d_path, "w") as f:
                new_str: str = dev_funcs.format(typename=typename)
                print(new_str)
                f.write(new_str)

            d_red_filename: str = f"vm_{type_aliases[idx]}_reductions.cu"
            d_red_path: str = f"{base_dir}/{d_red_filename}"
            with open(d_red_path, "w") as f:
                new_str: str = dev_red_funcs.format(typename=typename)
                print(new_str)
                f.write(new_str)
