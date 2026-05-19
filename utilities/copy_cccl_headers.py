# Copyright (c) 2025-2026 Gabriel Dorlhiac
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Find CUDA toolkit headers and copy into build dir for installation and use by JIT."""

import os
import shutil
import sys
from typing import Optional, Tuple

cuda_path: Optional[str] = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
if not cuda_path:
    # Check standard Unix locations
    std_locs: Tuple[str, str] = ("/usr/local/cuda", "/usr/cuda")
    for p in std_locs:
        if os.path.exists(p):
            cuda_path = p
            break

if not cuda_path:
    # If not found above, try and find it relative to the nvcc compiler location
    nvcc: Optional[str] = shutil.which("nvcc")
    if nvcc:
        cuda_path = os.path.dirname(os.path.dirname(nvcc))

if not cuda_path:
    sys.exit("Error: CUDA Toolkit not found. Cannot extract JIT headers.")

target_dir: str = sys.argv[1]

os.makedirs(target_dir, exist_ok=True)

cccl_subdirs: Tuple[str, str] = ("cuda", "nv")
for subdir in cccl_subdirs:
    src: str = os.path.join(cuda_path, "include", subdir)
    dst: str = os.path.join(target_dir, subdir)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    if os.path.exists(src):
        shutil.copytree(src, dst)
        print(f"Successfully copied JIT headers from {src} to {dst}")
    else:
        print(f"Warning: {src} does not exist. Skipping.")
