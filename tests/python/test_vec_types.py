from typing import List # Needs to run down to Python 3.8

import pytest

import ncarray

def test_vector_struct_access():
    """Test custom float vectors are available in Python."""
    f2: ncarray.Float2 = ncarray.Float2(1.0, 2.0)
    assert f2.x == 1.0
    assert f2.y == 2.0

    f2.x = 5.5
    assert f2.x == 5.5

    f4: ncarray.Float4 = ncarray.Float4(1, 2, 3, 4)
    assert f4.x == 1.0
    assert f4.z == 3.0
    assert f4.w == 4.0

def test_dtype_enum():
    """Test custom float vector naming makes it through bindings."""
    assert ncarray.DType.vfloat2.name == "vfloat2"
    assert ncarray.DType.vdouble4.name == "vdouble4"

def test_array_creation_with_vectors():
    """Test creation of arrays with the custom floats from Python."""
    shape: List[int] = [10]
    a: ncarray.NCArray = ncarray.NCArray(shape, ncarray.DType.vfloat2)
    assert a.dtype() == ncarray.DType.vfloat2
    assert a.itemsize == 8
