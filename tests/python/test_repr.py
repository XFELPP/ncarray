# Copyright (c) 2025-2026 Gabriel Dorlhiac
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
import numpy as np
import numpy.typing as npt

import ncarray


def test_repr_1d_small():
    """Test repr for a small 1D array."""

    ncarr: ncarray.NCArrayRef = ncarray.NCArrayRef(np.array([1, 2, 3], dtype=np.uint16))
    expected: str = "NCArrayRef([[1, 2, 3]], dtype=uint16)"

    assert repr(ncarr) == expected


def test_repr_1d_large():
    """Test repr for a large 1D array with truncation."""

    arr: npt.NDArray[np.int32] = np.arange(10, dtype=np.int32)
    ncarr: ncarray.NCArrayRef = ncarray.NCArrayRef(arr)
    expected: str = "NCArrayRef([[0, 1, 2, ..., 7, 8, 9]], dtype=int32)"

    assert repr(ncarr) == expected


def test_repr_2d_small():
    """Test repr for a small 2D array."""
    arr: npt.NDArray[np.int16] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    ncarr: ncarray.NCArrayRef = ncarray.NCArrayRef(arr)
    expected: str = (
        "NCArrayRef([[[1, 2, 3],\n"
        "             [4, 5, 6]]], dtype=int16)"
    )

    assert repr(ncarr) == expected


def test_repr_3d_small():
    """Test repr for a small 3D array showing multi-line segments."""

    arr: npt.NDArray[np.float32] = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32
    )
    ncarr: ncarray.NCArrayRef = ncarray.NCArrayRef(arr)
    # TODO: Investigate whether extra spaces on empty line should be there (vs NumPy)
    expected = (
        "NCArrayRef([[[[1, 2],\n"
        "              [3, 4]],\n"
        "             \n"
        "             [[5, 6],\n"
        "              [7, 8]]]], dtype=float32)"
    )

    assert repr(ncarr) == expected


def test_repr_large_2d():
    """Test truncation in 2D."""

    arr: npt.NDArray[np.int64] = np.arange(30, dtype=np.int64).reshape((10, 3))
    ncarr: ncarray.NCArrayRef = ncarray.NCArrayRef(arr)
    # TODO: Investigae spurious extra spaces at end.
    expected: str = (
        "NCArrayRef([[[0, 1, 2], \n" # E.g. space before \n here
        "             [3, 4, 5], \n"
        "             [6, 7, 8], \n"
        "             ...\n"
        "             [21, 22, 23],\n"
        "             [24, 25, 26],\n"
        "             [27, 28, 29]]], dtype=int64)"
    )

    assert repr(ncarr) == expected


def test_repr_soarray():
    """Test repr for SOArray and float"""
    # arr: npt.NDArray[np.float64] = np.array([1, 2, 3], dtype=np.float64)
    # TODO: The 1. formatting of NumPy is not reproduced currently. 1.1 works
    arr: npt.NDArray[np.float64] = np.array([1.1, 2.1, 3.1], dtype=np.float64)
    soarr: ncarray.SOArrayRef = ncarray.SOArrayRef(arr)
    expected: str = "SOArrayRef([[1.1, 2.1, 3.1]], dtype=float64)"

    assert repr(soarr) == expected
