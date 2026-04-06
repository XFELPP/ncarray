# Copyright (c) 2025-2026 Gabriel Dorlhiac
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

from ncarray.core._pyncarray import *
try:
    from ncarray.core._pyncdevarray import *
except (ModuleNotFoundError, ImportError):
    ...

def get_include() -> str:
    return os.path.join(os.path.dirname(__file__), "include")


def get_lib_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "lib")
