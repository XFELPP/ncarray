/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

namespace ncarray {
  namespace host {
    void* load_kernel_from_disk(fs::path file_path) {
      std::ifstream in_f(file_path, std::ios::binary | std::ios::ate);
      std::streamsize size { in_f.tellg() };
      in_f.seekg(0, std::ios::beg);

      std::vector<char> buffer(size);
      in_f.read(buffer.data(), size);

      void* exec_mem { nullptr };
#ifdef _WIN32
      exec_mem = VirtualAlloc(NULL,
                              size,
                              MEM_COMMIT | MEM_RESERVE,
                              PAGE_EXECUTE_READWRITE);
#else
      exec_mem = mmap(NULL,
                      size,
                      PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_ANONYMOUS | MAP_PRIVATE,
                      -1,
                      0);
#endif

      std::memcpy(exec_mem, buffer.data(), size);

      return exec_mem;
    }

    void write_kernel_to_disk(fs::path file_path,
                              std::size_t k_size,
                              const std::uint8_t* k_data) {
      std::ofstream out_f(file_path, std::ios::binary);

      if (!out_f) {
        std::cerr << "Failed to open stream for writing!\n";

        return;
      }

      out_f.write(reinterpret_cast<const char*>(k_data), k_size);
    }
  } // namespace host
} // namespace ncarray

