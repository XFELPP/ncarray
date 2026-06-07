/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_JIT_PATH_UTILS_HH
#define NCARRAY_JIT_PATH_UTILS_HH

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace ncarray {
  /**
   * Get the cache directory for the system.
   *
   * Uses XDG_CACHE_HOME on Linux if set, or LOCALPPDATA on Windows.
   */
  fs::path get_cache_dir();

  /**
   * Return the path to the currently loaded JIT shared library.
   */
  std::string get_install_library_path();

  /**
   * Return the path to the include headers for ncarray relative to the shared library.
   */
  std::string get_install_include_path();


  /**
   * Read a file at the provided path. The entire contents are returned in a string.
   *
   * @param file_path The path to read.
   * @returns contents The file contents as a string. Empty if there was an error.
   */
  std::string read_file(fs::path file_path);

  /**
   * Read a binary file at the provided path into executable memory.
   *
   * @param file_path The path to read.
   * @returns contents The data loaded into executable memory.
   */
  void* read_bin_file_to_exec_mem(fs::path file_path);

  /**
   * Write a file to the provided path.
   *
   * @param file_path The path to write to.
   * @param contents The file contents to write out.
   */
  void write_file(fs::path file_path, const std::string& contents);

  /**
   * Write a binary object to the provided path.
   *
   * @param file_path The path to write to.
   * @param k_size Number of bytes to write.
   * @param k_data Pointer to the start of the data to write.
   */
  void write_file(fs::path file_path, std::size_t k_size, const std::uint8_t* k_data);
} // namespace ncarray

#endif // NCARRAY_JIT_PATH_UTILS_HH
