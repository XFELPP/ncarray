import pytest
import numpy as np

import ncarray


def test_repr_1d_small():
    """Test repr for a small 1D array."""
    # Expected: NCArrayRef([1, 2, 3], dtype=uint16)
    # (Assuming dtype uint16 prints as 'uint16')
    ncarr: ncarray.NCArrayRef = ncarray.NCArrayRef(
        np.array([1, 2, 3], dtype=np.uint16)
    )
    print(ncarr)
    expected: str = "NCArrayRef([[1, 2, 3]], dtype=uint16)"
    assert repr(ncarr) == expected


def test_repr_1d_large():
    """Test repr for a large 1D array with truncation."""
    # shape = (10,)
    # expected = "NCArrayRef([0, 1, 2, ..., 7, 8, 9], dtype=int32)"
    expected = "NCArrayRef([[0, 1, 2, ..., 7, 8, 9]], dtype=int32)"
    # assert repr(arr) == expected


def test_repr_2d_small():
    """Test repr for a small 2D array."""
    # shape (2, 3)
    # indent = 12
    # expected =
    # "NCArrayRef([[1, 2, 3],
    #               [4, 5, 6]], dtype=int16)"
    expected = "NCArrayRef([[[1, 2, 3],\n" "              [4, 5, 6]]], dtype=int16)"
    # assert repr(arr) == expected


def test_repr_3d_small():
    """Test repr for a small 3D array showing multi-line segments."""
    # shape (2, 2, 2)
    # expected =
    # NCArrayRef([[[1, 2],
    #                [3, 4]],
    #
    #               [[5, 6],
    #                [7, 8]]], dtype=float32)
    expected = (
        "NCArrayRef([[[[1, 2],\n"
        "               [3, 4]],\n"
        "\n"
        "              [[5, 6],\n"
        "               [7, 8]]]], dtype=float32)"
    )
    # assert repr(arr) == expected


def test_repr_large_2d():
    """Test truncation in 2D."""
    # shape (10, 3) -> Truncate rows
    # expected =
    # NCArrayRef([[0, 1, 2],
    #               [3, 4, 5],
    #               [6, 7, 8],
    #               ...,
    #               [21, 22, 23],
    #               [24, 25, 26],
    #               [27, 28, 29]], dtype=int64)
    expected = (
        "NCArrayRef([[[0, 1, 2],\n"
        "              [3, 4, 5],\n"
        "              [6, 7, 8],\n"
        "              ...,\n"
        "              [21, 22, 23],\n"
        "              [24, 25, 26],\n"
        "              [27, 28, 29]]], dtype=int64)"
    )
    # assert repr(arr) == expected
