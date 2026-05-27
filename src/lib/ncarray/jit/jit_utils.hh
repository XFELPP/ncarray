/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_JIT_UTILS_HH
#define NCARRAY_JIT_UTILS_HH

#include <string>

namespace ncarray {
  std::string hash_to_hex(const std::string& input);

  std::string get_arch_opt();
} // namespace ncarray

#endif // NCARRAY_JIT_UTILS_HH
