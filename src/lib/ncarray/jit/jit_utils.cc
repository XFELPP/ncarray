/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/jit/jit_utils.hh"

#ifdef NCA_HAS_CUDA
#include <cuda.h>
#endif

#include <iomanip>
#include <sstream>
#include <string>

namespace ncarray {
  std::string hash_to_hex(const std::string& input) {
    std::hash<std::string> hasher;
    size_t hash = hasher(input);
    std::stringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << hash;
    return ss.str();
  }

  std::string get_arch_opt() {
#ifdef NCA_HAS_CUDA
    int major_v;
    int minor_v;
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    cuDeviceGetAttribute(&major_v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor_v, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

    std::string arch_opt {
      "-arch=sm_" + std::to_string(major_v) + std::to_string(minor_v)
    };

    return arch_opt;
#else
    std::string arch_opt { "" };
    return arch_opt;
#endif
  }
} // namespace ncarray
