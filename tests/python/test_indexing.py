# Copyright (c) 2025-2026 Gabriel Dorlhiac
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
import numpy as np
import numpy.typing as npt

import ncarray


def test_getitem_integer_arr():
    """Test simple indexing.

    NOTE: The reference bindings insert a pointer axis to avoid complications with
    pulling it out from inside NumPy. You can squeeze later to collapse the extra
    axis.
    """
    arr: npt.NDArray[np.int32] = np.arange(10, dtype=np.int32)
    nca: ncarray.NCArrayRef = ncarray.NCArrayRef(arr)
    assert nca[0, 0] == 0
    assert nca[0, 5] == 5
    assert nca[0, -1] == 9


def test_getitem_integer_list_of_arrs():
    """Test simple indexing - but bind a list of arrays.

    NOTE: The reference bindings insert a pointer axis to avoid complications with
    pulling it out from inside NumPy. You can squeeze later to collapse the extra
    axis.
    """
    arr: npt.NDArray[np.int32] = np.arange(10, dtype=np.int32)
    nca: ncarray.NCArrayRef = ncarray.NCArrayRef([arr])
    assert nca[0, 0] == 0
    assert nca[0, 5] == 5
    assert nca[0, -1] == 9


def test_getitem_slice_arr():
    """Test simple slicing.

    NOTE: The reference bindings insert a pointer axis to avoid complications with
    pulling it out from inside NumPy. You can squeeze later to collapse the extra
    axis.
    """
    arr: npt.NDArray[np.int32] = np.arange(10, dtype=np.int32)
    nca: ncarray.NCArrayRef = ncarray.NCArrayRef(arr)
    slc: ncarray.NCArrayView = nca[0, 2:7]
    assert slc.shape == (5,)
    assert np.array_equal(np.array(slc), arr[2:7])


def test_getitem_slice_list_of_arrs():
    """Test simple slicing - but bind a list of arrays.

    NOTE: The reference bindings insert a pointer axis to avoid complications with
    pulling it out from inside NumPy. You can squeeze later to collapse the extra
    axis.
    """
    arr: npt.NDArray[np.int32] = np.arange(10, dtype=np.int32)
    nca: ncarray.NCArrayRef = ncarray.NCArrayRef([arr])
    slc: ncarray.NCArrayView = nca[0, 2:7]
    assert slc.shape == (5,)
    assert np.array_equal(np.array(slc), arr[2:7])


def test_getitem_tuple_2d_arr():
    """Test multi-axis slicing.

    NOTE: The reference bindings insert a pointer axis to avoid complications with
    pulling it out from inside NumPy. You can squeeze later to collapse the extra
    axis.
    """
    arr: npt.NDArray[np.int32] = np.arange(20, dtype=np.int32).reshape((4, 5))
    nca: ncarray.NCArrayRef = ncarray.NCArrayRef(arr)
    assert nca[0, 1, 2] == 7

    slc: ncarray.NCArrayView = nca[0, 1:3, 2:4]
    assert slc.shape == (2, 2)
    assert np.array_equal(np.array(slc), arr[1:3, 2:4])


def test_getitem_tuple_2d_list_of_arrs():
    """Test multi-axis slicing - but bind a list of arrays.

    NOTE: The reference bindings insert a pointer axis to avoid complications with
    pulling it out from inside NumPy. You can squeeze later to collapse the extra
    axis.
    """
    arr: npt.NDArray[np.int32] = np.arange(20, dtype=np.int32).reshape((4, 5))
    nca: ncarray.NCArrayRef = ncarray.NCArrayRef([arr])
    assert nca[0, 1, 2] == 7

    slc: ncarray.NCArrayView = nca[0, 1:3, 2:4]
    assert slc.shape == (2, 2)
    assert np.array_equal(np.array(slc), arr[1:3, 2:4])


def test_squeeze():
    """Test the squeeze function."""
    arr: npt.NDArray[np.float32] = np.ones((1, 5, 1, 3), dtype=np.float32)
    nca: ncarray.NCArrayRef = ncarray.NCArrayRef([arr])
    sq: ncarray.NCArrayView = nca.squeeze()
    assert sq.shape == (5, 3)


def test_setitem_integer():
    """Test simple index setting.

    NOTE: The reference bindings insert a pointer axis to avoid complications with
    pulling it out from inside NumPy. You can squeeze later to collapse the extra
    axis.
    """
    arr: npt.NDArray[np.int32] = np.arange(10, dtype=np.int32)
    nca: ncarray.NCArrayRef = ncarray.NCArrayRef([arr])
    nca[0, 5] = 42
    assert nca[0, 5] == 42
    assert arr[5] == 42  # Ref array modifies underneath numpy array


def test_soarray_indexing_arr():
    """Test simple indexing on SOArray*.

    NOTE: The reference bindings insert a pointer axis to avoid complications with
    pulling it out from inside NumPy. You can squeeze later to collapse the extra
    axis.
    """
    arr: npt.NDArray[np.int32] = np.arange(12, dtype=np.int32).reshape((3, 4))
    soa: ncarray.SOArrayRef = ncarray.SOArrayRef(arr)
    assert soa[0, 1, 2] == 6
    slc: ncarray.SOArrayView = soa[0, 1:3, 0:2]
    assert slc.shape == (2, 2)
    assert np.array_equal(np.array(slc), arr[0, 1:3, 0:2])


def test_soarray_indexing_list_of_arrs():
    """Test simple indexing on SOArray* - but bind a list of arrays.

    NOTE: The reference bindings insert a pointer axis to avoid complications with
    pulling it out from inside NumPy. You can squeeze later to collapse the extra
    axis.
    """
    arr: npt.NDArray[np.int32] = np.arange(12, dtype=np.int32).reshape((3, 4))
    soa: ncarray.SOArrayRef = ncarray.SOArrayRef([arr])
    assert soa[0, 1, 2] == 6
    slc: ncarray.SOArrayView = soa[0, 1:3, 0:2]
    assert slc.shape == (2, 2)
    assert np.array_equal(np.array(slc), arr[1:3, 0:2])
