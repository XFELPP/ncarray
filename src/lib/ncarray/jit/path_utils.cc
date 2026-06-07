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
#include <sys/mman.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
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

    fs::path grandparent = lib_file.parent_path().parent_path();

    // The install hierarchy is something like this, inside the wheel variant:
    // PREFIX/lib/..../ncarray/
    //                    | ----- lib/
    //                    | ----- include/
    // So, get the parent path from the lib, and then append include to get headers
    // This should work for a standard installation as well (e.g. via conda)
    fs::path std_inc = grandparent / "include";
    if (fs::exists(std_inc, ec)) {
      return std_inc.string();
    }

    // In this case, we are in a split-wheel setup
    // site-packages/ncarray.libs/libncdevarrrayjit.so ...
    // site-packages/ncarray
    //                 | ------- include/
    // So must use grandparent / ncarray/include
    return (grandparent / "ncarray/include").string();
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

  void* read_bin_file_to_exec_mem(fs::path file_path) {
    std::ifstream in_f(file_path, std::ios::binary | std::ios::ate);
    std::streamsize size{in_f.tellg()};
    in_f.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    in_f.read(buffer.data(), size);

    void* exec_mem{nullptr};
#ifdef _WIN32
    exec_mem = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
#else
    exec_mem =
        mmap(NULL, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
#endif

    std::memcpy(exec_mem, buffer.data(), size);

    return exec_mem;
  }

  void write_file(fs::path file_path, const std::string& contents) {
    std::ofstream out_f(file_path, std::ios::binary);

    if (!out_f) {
      std::cerr << "Failed to open stream for writing!\n";

      return;
    }

    out_f.write(contents.data(), contents.size());
  }

  void write_file(fs::path file_path, std::size_t k_size, const std::uint8_t* k_data) {
    std::ofstream out_f(file_path, std::ios::binary);

    if (!out_f) {
      std::cerr << "Failed to open stream for writing!\n";

      return;
    }

    out_f.write(reinterpret_cast<const char*>(k_data), k_size);
  }
} // namespace ncarray
