# Copyright (c) 2025-2026 Gabriel Dorlhiac
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import List  # Needs to run down to Python 3.8

import pytest
import numpy as np
import numpy.typing as npt

import ncarray as nca


def test_gpu_binary_arithmetic():
    """Test GPU array binary arithmetic operators."""
    shape: List = [4, 4, 4]
    ncarr1: nca.NCDevArray = nca.NCDevArray(shape, nca.DType.float32)
    ncarr2: nca.NCDevArray = nca.NCDevArray(shape, nca.DType.float32)

    # Fill with a scalar
    ncarr1.fill(2.0)
    ncarr2.fill(4.0)

    # These are kernels - the results remain on device
    d_sum_res: nca.NCDevArray = ncarr1 + ncarr2
    d_prod_res: nca.NCDevArray = ncarr1 * ncarr2

    # Move back to host for testing
    h_sum_res: nca.NCArray = d_sum_res.to_host()
    h_prod_res: nca.NCArray = d_prod_res.to_host()

    h_nparr_sum: npt.NDArray[np.float32] = np.ones(shape, dtype=np.float32) * 6.0
    h_nparr_prod: npt.NDArray[np.float32] = np.ones(shape, dtype=np.float32) * 8.0

    assert np.allclose(h_nparr_sum, h_sum_res)
    assert np.allclose(h_nparr_prod, h_prod_res)


def test_gpu_reductions():
    """Test simple reductions with GPU arrays."""
    shape: List = [256]
    h_ncarr: nca.NCArray = nca.NCArray(shape, nca.DType.uint32)

    for i in range(256):
        h_ncarr[i] = i

    # Move to device
    d_ncarr: nca.NCDevArray = h_ncarr.to_device()

    # Test reductions on device
    # NOTE: Simple scalars are accessible on host and device (via pinned mem)
    d_sum: float = d_ncarr.sum()
    d_max: float = d_ncarr.max()
    d_min: float = d_ncarr.min()

    assert d_sum == 32640
    assert d_max == 255
    assert d_min == 0

    h_sum: float = h_ncarr.sum()
    h_max: float = h_ncarr.max()
    h_min: float = h_ncarr.min()

    assert d_sum == h_sum
    assert d_max == h_max
    assert d_min == h_min


def test_gpu_copies():
    """Additional tests for device-host round trip copying."""
    data: npt.NDArray[np.float32] = np.random.rand(10, 10).astype(np.float32)
    h_ncarr: nca.NCArrayRef = nca.NCArrayRef(data)

    # Move to device
    d_ncarr: nca.NCDevArray = h_ncarr.to_device()

    # NOTE: All metadata is still accessible here on the host
    assert d_ncarr.dtype == nca.DType.float32

    # Now copy it back
    h_roundtrip_ncarr: nca.NCArray = d_ncarr.to_host()
    assert np.allclose(np.array(h_roundtrip_ncarr), data)

    # Big challenge: Copy directly into NumPy
    np_from_dev: npt.NDArray[np.float32] = np.array(d_ncarr)
    assert np.allclose(np_from_dev, data)


def test_gpu_inplace_arithmetic():
    """Exercise inplace arithmetic operators on device."""
    shape: List[int] = [100]
    d_ncarr: nca.NCDevArray = nca.NCDevArray(shape)
    d_ncarr.fill(10.0)

    d_ncarr += 5.0
    d_ncarr *= 2.0

    # Copy directly into the NumPy array for comparison test
    assert np.allclose(np.array(d_ncarr), 30.0)


def test_gpu_sliced_assignment():
    """Test assignment using indexing and broadcasting"""
    shape: List[int] = [10, 10]
    d_ncarr: nca.NCDevArray = nca.NCDevArray(shape, dtype=nca.DType.float32)
    d_ncarr.fill(0.0)

    # Get slice and broadcast a scalar
    d_ncarr[1:3, 1:3] = 5.0

    # Get slice and broadcast an array
    h_nparr: npt.NDArray[np.float32] = np.ones((2, 2), dtype=np.float32) * 10.0
    d_ncarr[7:9, 7:9] = h_nparr

    # Copy back to host for testing
    h_res: npt.NDArray[np.float32] = np.array(d_ncarr)
    assert h_res[0, 0] == 0.0
    assert np.allclose(h_res[1:3, 1:3], 5.0)
    assert np.allclose(h_res[7:9, 7:9], 10.0)


def test_gpu_comparisons():
    """Test GPU array comparison operators."""
    shape: List[int] = [10]

    d_ncarr: nca.NCDevArray = nca.NCDevArray(shape, nca.DType.float32)
    d_ncarr.fill(10.0)
    d_ncarr[0:5].fill(0.0)
    # Array should now be: [0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    # Create a new boolean array on device
    d_mask_gt0: nca.NCDevArray = d_ncarr > 5.0
    # Expected: [F, F, F, F, F, T, T, T, T, T]
    assert d_mask_gt0.dtype == nca.DType.bool

    h_res_gt0: npt.NDArray[np.bool_] = np.array(d_mask_gt0)
    assert not h_res_gt0[0]
    assert h_res_gt0[9]

    # Reassign
    d_ncarr[0:10] = np.linspace(0, 9, 10, dtype=np.float32)
    # Array should now be: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    d_mask_gt1: nca.NCDevArray = d_ncarr > 5.0
    d_mask_ge0: nca.NCDevArray = d_ncarr >= 5.0
    # Expected: [F, F, F, F, F, F, T, T, T, T]
    # Expected: [F, F, F, F, F, T, T, T, T, T]
    assert d_mask_gt1.dtype == nca.DType.bool_
    assert d_mask_ge0.dtype == nca.DType.bool_

    h_res_gt1: npt.NDArray[np.bool_] = np.array(d_mask_gt1)
    h_res_ge0: npt.NDArray[np.bool_] = np.array(d_mask_ge0)
    assert not h_res_gt1[5]
    assert h_res_ge0[5]
