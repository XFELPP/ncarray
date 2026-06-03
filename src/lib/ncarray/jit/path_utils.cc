/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "ncarray/jit/path_utils.hh"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace ncarray {
  fs::path get_cache_dir() {
    std::string path;
#ifdef _WIN32
    const char* local_app_data = std::getenv("LOCALAPPDATA");
    if (local_app_data) {
      path = std::string(local_app_data) + "/ncarray/cache";
    } else {
      path = "./ncarray_cache";
    }
#else
    const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
    if (xdg_cache) {
      path = std::string(xdg_cache) + "/ncarray";
    } else {
      const char* home = std::getenv("HOME");
      path = std::string(home ? home : ".") + "/.cache/ncarray";
    }
#endif
    fs::create_directories(path);

    return fs::path(path);
  }

  std::string get_install_library_path() {
    std::string path;
#ifdef _WIN32
    HMODULE hModule { nullptr };


    GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                      GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                      reinterpret_cast<LPCSTR>(&get_install_library_path),
                      &hModule);

    std::vector<char> buffer(MAX_PATH); // MAX_PATH comes from windows.h
    DWORD size { GetModuleFileName(hModule, buffer.data(), buffer.size()) };

    while (size == buffer.size()) {
      buffer.resize(buffer.size() * 2);
      size = GetModuleFileName(hModule, buffer.data(), buffer.size());
    }

    path = std::string(buffer.data(), size);
#else
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(get_install_library_path), &info)) {
      path = std::string(info.dli_fname);
    }
#endif

    return path;
  }

  std::string get_install_include_path() {
    if (const char* env_inc = std::getenv("NCARRAY_INCLUDE_DIR")) {
      // This provides a mechanism to inject an include directory when not installed
      // E.g., the test suite can run from the build dir using this env var
      return std::string(env_inc);
    }

    std::error_code ec;
    fs::path lib_file = fs::canonical(get_install_library_path(), ec);
    if (ec) {
      lib_file = fs::absolute(get_install_library_path());
    }

    // The install hierarchy is something like this, inside the wheel variant:
    // PREFIX/lib/..../ncarray/
    //                    | ----- lib/
    //                    | ----- include/
    // So, get the parent path from the lib, and then append include to get headers
    // This should work for a standard installation as well (e.g. via conda)
    return (lib_file.parent_path().parent_path() / "include").string();
  }

  std::string read_file(fs::path file_path) {
    std::ifstream in_f(file_path, std::ios::binary);

    if (!in_f) {
      std::cerr << "Failed to open file!\n";

      return "";
    }
    std::string contents((std::istreambuf_iterator<char>(in_f)),
                         std::istreambuf_iterator<char>());
    return contents;
  }

  void write_file(fs::path file_path, const std::string& contents) {
    std::ofstream out_f(file_path, std::ios::binary);

    if (!out_f) {
      std::cerr << "Failed to open stream for writing!\n";

      return;
    }

    out_f.write(contents.data(), contents.size());
  }
} // namespace ncarray
