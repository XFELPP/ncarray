# Copyright (c) 2025-2026 Gabriel Dorlhiac
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import contextlib
import ctypes
import ctypes.util
import os
import platform
import sys


def get_pkg_config() -> str:
    nca_wheel_pkgconfig_dir: str = os.path.join(os.path.dirname(__file__), "pkgconfig")
    if os.path.exists(nca_wheel_pkgconfig_dir):
        return nca_wheel_pkgconfig_dir

    win_prefix_lib_dir: str = os.path.join(sys.prefix, "Library")
    if os.path.exists(win_prefix_lib_dir):
        win_prefix_pkgconfig_dir: str = os.path.join(win_prefix_lib_dir, "pkgconfig")

        if os.path.exists(win_prefix_pkgconfig_dir):
            return win_prefix_pkgconfig_dir

        for subdir in os.listdir(win_prefix_lib_dir):
            subdir_pkgconfig: str = os.path.join(
                win_prefix_lib_dir, subdir, "pkgconfig"
            )
            if os.path.exists(subdir_pkgconfig):
                return subdir_pkgconfig

    unix_prefix_lib_dir: str = os.path.join(sys.prefix, "lib")
    if os.path.exists(unix_prefix_lib_dir):
        unix_prefix_pkgconfig_dir: str = os.path.join(unix_prefix_lib_dir, "pkgconfig")

        if os.path.exists(unix_prefix_pkgconfig_dir):
            return unix_prefix_pkgconfig_dir

        for subdir in os.listdir(unix_prefix_lib_dir):
            subdir_pkgconfig: str = os.path.join(
                unix_prefix_lib_dir, subdir, "pkgconfig"
            )
            if os.path.exists(subdir_pkgconfig):
                return subdir_pkgconfig

    return ""


def nca_pkg_config() -> None:
    import argparse
    import os
    import re
    from typing import Optional

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkg-config-cflags",
        action="store_true",
        help="Print the Cflags from the .pc file.",
    )
    parser.add_argument(
        "--pkg-config-libs",
        action="store_true",
        help="Print the Libs from the .pc file.",
    )
    parser.add_argument(
        "--pkg-config-path",
        action="store_true",
        help="Print the path usable by pkg-config.",
    )

    args: argparse.Namespace = parser.parse_args()

    pc_path: str = get_pkg_config()
    pc_file: str = os.path.join(pc_path, "ncarray.pc")
    if args.pkg_config_cflags:
        if pc_path and os.path.exists(pc_file):
            with open(pc_file, "r") as f:
                pc_content: str = f.read()
                cflags_m: Optional[re.Match] = re.search(
                    r"(?<=Cflags:)(.*)", pc_content
                )
                if cflags_m:
                    print(cflags_m.group().strip())
    elif args.pkg_config_libs:
        if pc_path and os.path.exists(pc_file):
            with open(pc_file, "r") as f:
                pc_content = f.read()
                libs_m: Optional[re.Match] = re.search(r"(?<=Libs:)(.*)", pc_content)
                if libs_m:
                    print(libs_m.group().strip())
    elif args.pkg_config_path:
        print(pc_path)


def get_include() -> str:
    nca_wheel_include_dir: str = os.path.join(os.path.dirname(__file__), "include")
    if os.path.exists(nca_wheel_include_dir):
        return nca_wheel_include_dir

    # When installing with conda or not pure pip, the headers will be in the standard
    # prefix
    win_prefix_include_dir: str = os.path.join(sys.prefix, "Library", "include")
    if os.path.exists(win_prefix_include_dir):
        return win_prefix_include_dir

    unix_prefix_include_dir: str = os.path.join(sys.prefix, "include")
    return unix_prefix_include_dir


def get_lib_dir() -> str:
    py_package_dir: str = os.path.dirname(__file__)

    # Check full-wheel path first: site-packages/ncarray/lib
    nca_wheel_lib_dir: str = os.path.join(py_package_dir, "lib")
    if os.path.exists(nca_wheel_lib_dir):
        return nca_wheel_lib_dir

    # Check split-wheel: site-packages/ncarray/.dylibs
    #                    site-packages/ncarray/.libs
    nca_wheel_dot_dylibs_dir: str = os.path.join(py_package_dir, ".dylibs")
    if os.path.exists(nca_wheel_dot_dylibs_dir):
        return nca_wheel_dot_dylibs_dir

    nca_wheel_dot_libs_dir: str = os.path.join(py_package_dir, ".libs")
    if os.path.exists(nca_wheel_dot_libs_dir):
        return nca_wheel_dot_libs_dir

    # Check split-wheel: site-packages/ncarray/
    #                    site-packages/ncarray.libs/libncarray.so
    parent_dir: str = os.path.dirname(py_package_dir)
    for folder in os.listdir(parent_dir):
        if (
            "ncarray" in folder or "dylib" in folder or "lib" in folder
        ) and folder != os.path.basename(py_package_dir):
            libs_dir: str = os.path.join(parent_dir, folder)
            if not os.path.isdir(libs_dir):
                continue
            for fname in os.listdir(libs_dir):
                if fname.startswith("libncarray") or fname.startswith("ncarray"):
                    if (
                        fname.endswith(".so")
                        or fname.endswith(".dylib")
                        or fname.endswith(".dll")
                        or ".so." in fname
                    ):
                        return libs_dir

    # When installing with conda or not pure pip, the lib will be in the standard
    # prefix
    win_prefix_lib_dir: str = os.path.join(sys.prefix, "Library", "lib")
    if os.path.exists(win_prefix_lib_dir):
        return win_prefix_lib_dir

    unix_prefix_lib_dir: str = os.path.join(sys.prefix, "lib")
    return unix_prefix_lib_dir


if platform.system() == "Windows":
    # Need to explicitly register DLL directories on Windows
    lib_dir: str = get_lib_dir()
    if os.path.exists(lib_dir):
        os.add_dll_directory(lib_dir)

    # Have to add the eager eval library dir as well. This lib allows controlling
    # the global switch between eager evaluation and lazy-expression evaluation
    # libeagereval (eagereval-1.dll on Windows) is in ncarray/core not, ncarray/lib.
    core_dir: str = os.path.join(os.path.dirname(__file__), "core")
    if os.path.exists(core_dir):
        os.add_dll_directory(core_dir)
elif platform.system() == "Linux":
    # Add stub file if needed
    has_driver: bool = ctypes.util.find_library("cuda") is not None
    if not has_driver:
        stub_path: str = os.path.join(
            os.path.dirname(__file__), "core", "stubs", "libcuda.so"
        )
        if os.path.exists(stub_path):
            try:
                ctypes.CDLL(stub_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass

from ncarray.core._pyncarray import *

try:
    from ncarray.core._pyncdevarray import *
except (ModuleNotFoundError, ImportError):
    ...


@contextlib.contextmanager
def lazy_mode():
    """Context manager to delay immediate evaluation of results."""

    old: bool = is_eager()
    set_eager(False)
    try:
        yield
    finally:
        set_eager(old)
