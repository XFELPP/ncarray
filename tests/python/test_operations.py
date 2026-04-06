# Copyright (c) 2025-2026 Gabriel Dorlhiac
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest
import numpy as np
import numpy.typing as npt

import ncarray as nca


def test_binary_add():
    """Test simple binary sum."""
    arr1: npt.NDArray[np.int32] = np.array([1, 2, 3], dtype=np.int32)
    arr2: npt.NDArray[np.int32] = np.array([4, 5, 6], dtype=np.int32)
    ncarr1: nca.NCArrayRef = nca.NCArrayRef([arr1])

    res: nca.NCArray = ncarr1 + arr2
    # NOTE: Need to squeeze due to extra dim
    assert np.array_equal(np.array(res.squeeze()), arr1 + arr2)


def test_binary_sub():
    """Test simple binary difference."""
    arr1: npt.NDArray[np.float32] = np.array([10, 20, 30], dtype=np.float32)
    arr2: npt.NDArray[np.float32] = np.array([1, 2, 3], dtype=np.float32)
    ncarr1: nca.NCArrayRef = nca.NCArrayRef(arr1)

    res: nca.NCArray = ncarr1 - arr2
    # NOTE: Need to squeeze due to extra dim
    assert np.array_equal(np.array(res.squeeze()), arr1 - arr2)


def test_binary_mul():
    """Test simple binary product."""
    arr1: npt.NDArray[np.int32] = np.array([2, 3, 4], dtype=np.int32)
    arr2: npt.NDArray[np.int32] = np.array([3, 4, 5], dtype=np.int32)
    ncarr1: nca.NCArrayRef = nca.NCArrayRef(arr1)

    res: nca.NCArray = ncarr1 * arr2
    # NOTE: Need to squeeze due to extra dim
    assert np.array_equal(np.array(res.squeeze()), arr1 * arr2)


def test_binary_div():
    """Test simple binary truediv."""
    arr1: npt.NDArray[np.float64] = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    arr2: npt.NDArray[np.float64] = np.array([2.0, 4.0, 5.0], dtype=np.float64)
    ncarr1: nca.NCArrayRef = nca.NCArrayRef(arr1)

    res: nca.NCArray = ncarr1 / arr2
    # NOTE: Need to squeeze due to extra dim
    assert np.array_equal(np.array(res.squeeze()), arr1 / arr2)


def test_reductions():
    """Test all sum/max/min reductions to scalar."""
    arr: npt.NDArray[np.int32] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    ncarr: nca.NCArrayRef = nca.NCArrayRef(arr)

    assert ncarr.sum() == 21
    assert ncarr.max() == 6
    assert ncarr.min() == 1


def test_mean_reduction():
    """Test mean reduction calculation."""
    arr: npt.NDArray[np.float32] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ncarr: nca.NCArrayRef = nca.NCArrayRef(arr)
    assert pytest.approx(ncarr.mean()) == 2.0


def test_scalar_ops():
    """Test scalar broadcast."""
    arr: npt.NDArray[np.float32] = np.array([1, 2, 3], dtype=np.float32)
    ncarr: nca.NCArrayRef = nca.NCArrayRef(arr)

    res_add: nca.NCArray = ncarr + 5.0
    # NOTE: Need to squeeze due to extra dim
    assert np.array_equal(np.array(res_add.squeeze()), arr + 5.0)

    res_mul: nca.NCArray = ncarr * 2.0
    # NOTE: Need to squeeze due to extra dim
    assert np.array_equal(np.array(res_mul.squeeze()), arr * 2.0)


def test_soarray_ops():
    """Test simple binary sum with SOArray (suboffsets)."""
    arr1: npt.NDArray[np.int32] = np.array([2, 4, 6], dtype=np.int32)
    arr2: npt.NDArray[np.int32] = np.array([1, 2, 3], dtype=np.int32)
    soarr: nca.SOArrayRef = nca.SOArrayRef(arr1)

    res: nca.SOArray = soarr + arr2
    # NOTE: Need to squeeze due to extra dim
    assert np.array_equal(np.array(res.squeeze()), arr1 + arr2)
