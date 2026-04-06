# Copyright (c) 2025-2026 Gabriel Dorlhiac
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import List  # Needs to run down to Python 3.8

import pytest

import ncarray as nca


def test_vector_struct_access():
    """Test custom float vectors are available in Python."""
    f2: nca.Float2 = nca.Float2(1.0, 2.0)
    assert f2.x == 1.0
    assert f2.y == 2.0

    f2.x = 5.5
    assert f2.x == 5.5

    f4: nca.Float4 = nca.Float4(1, 2, 3, 4)
    assert f4.x == 1.0
    assert f4.z == 3.0
    assert f4.w == 4.0


def test_dtype_enum():
    """Test custom float vector naming makes it through bindings."""
    assert nca.DType.vfloat2.name == "vfloat2"
    assert nca.DType.vdouble4.name == "vdouble4"


def test_array_creation_with_vectors():
    """Test creation of arrays with the custom floats from Python."""
    shape: List[int] = [10]
    ncarr: nca.NCArray = nca.NCArray(shape, nca.DType.vfloat2)
    assert ncarr.dtype == nca.DType.vfloat2
    assert ncarr.itemsize == 8
