# Copyright (c) 2025-2026 Gabriel Dorlhiac
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import contextlib
import os
import sys

from ncarray.core._pyncarray import *

try:
    from ncarray.core._pyncdevarray import *
except (ModuleNotFoundError, ImportError):
    ...


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
    nca_wheel_lib_dir: str = os.path.join(os.path.dirname(__file__), "lib")
    if os.path.exists(nca_wheel_lib_dir):
        return nca_wheel_lib_dir

    # When installing with conda or not pure pip, the lib will be in the standard
    # prefix
    win_prefix_lib_dir: str = os.path.join(sys.prefix, "Library", "lib")
    if os.path.exists(win_prefix_lib_dir):
        return win_prefix_lib_dir

    unix_prefix_lib_dir: str = os.path.join(sys.prefix, "lib")
    return unix_prefix_lib_dir


@contextlib.contextmanager
def lazy_mode():
    """Context manager to delay immediate evaluation of results."""

    old: bool = is_eager()
    set_eager(False)
    try:
        yield
    finally:
        set_eager(old)
