/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_JIT_HOST_JIT_UTILS_HH
#define NCARRAY_JIT_HOST_JIT_UTILS_HH

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace ncarray {
  namespace host {
    ExprKernelFunc load_kernel_from_disk(fs::path file_path);

    void write_kernel_to_disk(fs::path cache_path,
                              std::size_t k_size,
                              const std::uint8_t* k_data);
  } // namespace host
} // namespace ncarray

#endif // NCARRAY_JIT_HOST_JIT_UTILS_HH
