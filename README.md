# ncarray
[![Release build](https://github.com/XFELPP/ncarray/actions/workflows/release.yml/badge.svg)](https://github.com/XFELPP/ncarray/actions/workflows/release.yml)

## Status
This is an early alpha release. The project is still in the very early stages of development.

Documentation is entirely lacking, as are test suites.

## Overview

`ncarray` provides a number of C++ array classes, compatible with NumPy, for working with non-contiguous data that cannot be described exclusively by strides. Specifically, these classes deal with data described by pointer-to-pointer setups/tables (double pointers, i.e. suboffsets in the Python world).

## Who is this for?

NumPy is extraordinarily powerful and flexible; however, it is designed for dealing with strided memory layouts. When dealing with disjoint collections of data spread out over, you would typically be forced to copy each section into a contiguous buffer to consider the entire set as a single array. When dealing with a large number of arrays, or under tight performance constraints, this may be prohibitively costly. `ncarray` is intended to provide a wrapper (known as a span, or view in many languages and libraries) over these disjoint arrays, without needing any initial copy.

## Features

`ncarray` is a C++20 generic library. Python bindings are provided by `pybind11`, although the C++ library can be used standalone, or wrapped using other techniques. The current set of features are:

- Pointer axis support - zero-copy on disjoint sets of arrays
- Type and concept system with various traits allowing for automatic type promotion (small int to wide int) when accumulating, or implementations of comparison operators for types like `std::complex<T>`.
- Multi-dimensional array indexing using integers, slices or placeholder axes (ellipsis in Python).
- NumPy-compatible Python bindings - implements interface methods like `__array__` and `__array_ufunc__` (**Note:** These operations may incur copies! The initial views are copy free, but not all operations thereafter)

NOTE: The type and trait system will automatically promote small integers to larger ones. For subtraction, it will convert unsigned to signed. This behaviour may be different than what is done in NumPy.

## Example Python Usage

```py
from typing import List

import numpy as np
import numpy.typing as npt

import ncarray # Alias it if you'd like :)

# Construct a disjoint set of arrays - subarrays that are really part of 1 larger one
subarray_list: List[npt.NDArray[np.uint32]] = [
    np.random.randint(1, 255, size=(512,1024), dtype=np.uint16) for _ in range(10)
]

# Create a wrapped reference
ncarr: ncarray.NCArrayRef = ncarray.NCArrayRef(subarray_list)

# Index it
first_two_ptrs: ncarray.NCArrayView = ncarr[:2]

# Down to scalar if you'd like
my_int: int = ncarr[2, 1, 2]

# Scalar broadcasts -- This creates a new OWNING array! (NCArray)
# Supported: +, -, *, / (Will provide % and //, i.e. int division in the future)
scalar_bcast_res: ncarray.NCArray = ncarr + 2

# Array arithmetic, supporting +, -, *, / as above
# Require identical shapes, currently!
# Supportable broadcasts may come later (E.g. a row/col into a 2D array)
other_res: ncarray.NCArray = ncarr + ncarr

# Perform operations on it -- also create new OWNING arrays
# Any ufunc *should* work -- this builds NumPy arrays, however.
# Because of this, these operations are rather slow.
result: ncarray.NCArray = np.sin(ncarr)
```

## Terminology

Coming soon... but to provide the most important point.

There are 3 types of classes:
- "Views"
- "Refs"
- Arrays (Owning)

A view is most similar to a span in C++. It owns nothing - it is useful if working in C++, or you will receive views when doing operations like indexing in Python. A ref, as was initially constructed above, holds pointers to each of the individual arrays. Finally, there is an owning array type. This holds the actual buffer/memory.


## Roadmap

- Provide `SOArray*` classes with PEP3118 sub-offset support.
- Fancy indexing on all array types (boolean masking etc).
- `DLPack` suppport
- CUDA/GPU interfaces
