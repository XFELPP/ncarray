# ncarray
[![Release build](https://github.com/XFELPP/ncarray/actions/workflows/release.yml/badge.svg)](https://github.com/XFELPP/ncarray/actions/workflows/release.yml)[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

## Status
This is an early alpha release. The project is still in the very early stages of development.

Documentation is entirely lacking, as are test suites.

## Overview

`ncarray` provides a number of C++ array classes, compatible with NumPy, for working with non-contiguous data that cannot be described exclusively by strides. Specifically, these classes deal with data described by pointer-to-pointer setups/tables (double pointers, i.e. suboffsets in the Python world).

The library provides views for views of data on the CPU and GPU (Linux/Windows only). When working on the device, array objects can be passed directly into kernels and device functions - if they are views. Owner-type arrays must be worked with from the host, but a view can be created using the `.view()` function.

## Who is this for?

NumPy is extraordinarily powerful and flexible; however, it is designed for dealing with strided memory layouts. When dealing with disjoint collections of data spread out over, you would typically be forced to copy each section into a contiguous buffer to consider the entire set as a single array. When dealing with a large number of arrays, or under tight performance constraints, this may be prohibitively costly. `ncarray` is intended to provide a wrapper (known as a span, or view in many languages and libraries) over these disjoint arrays, without needing any initial copy.

## Features

`ncarray` is a C++20 generic library. Python bindings are provided by `pybind11`, although the C++ library can be used standalone, or wrapped using other techniques. The current set of features are:

- Pointer axis support - zero-copy on disjoint sets of arrays
- Type and concept system with various traits allowing for automatic type promotion (small int to wide int) when accumulating, or implementations of comparison operators for types like `std::complex<T>`.
- Multi-dimensional array indexing using integers, slices or placeholder axes (ellipsis in Python).
- NumPy-compatible Python bindings - implements interface methods like `__array__` and `__array_ufunc__` (**Note:** These operations may incur copies! The initial views are copy free, but not all operations thereafter)
- Views of GPU memory using `NCDev*` or `SODev*` prefixed arrays.

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

## Concepts and Terminology

The `ncarray` library provides a series of array types. These are constructed through the composition of a `Layout` and a `Storage` policy specifier in a base `ArrayImpl` class. At the Python level, the base classes are not made available, and instead, bindings are provided for a number of specializations.

### `Layout` policies

There are two types of layouts which define how the memory of the array is distributed, and provide the mechanism to traverse said memory. These layouts (in the C++ source) are:

- `NCOffsetsPolicy`: A layout where there is a single "pointer axis". When traversing this axis, the data is interpreted as a double pointer (alternatively, a pointer table, etc.), and an index is a selection of which pointer to use. Otherwise, the data is traversed using the standard stride mechanism (potential with extra offsets).
- `SOArrayPolicy`: A layout implementing PEP3118 suboffsets.

The latter policy is more flexible and general. Anything specified by the former can be specified by the latter. The former was implemented first, however, and it is generally simpler to reason about. It maps naturally to the case where you have a collection of othewise strided/contiguous arrays, but want to consider the collection as a single array. The "collection axis" then becomes the first axis of the array, and is the pointer axis.

### `Storage` policies

There are 3 kinds of storage specifiers:

- `ViewPolicy`: Holding only a pointer to the underlying data. This is a pure view (e.g. like a span).
- `RefPolicy`: A policy which holds the pointers to the underlying components of the data.
- `OwnerPolicy`: A policy where the array actually owns the memory for the data (and is therefore responsible for allocation and freeing of it).

NOTE: The utility of the second policy becomes apparent when trying to construct a view over a collection of arrays in Python. When passed in a list or array of arrays, something needs to provide a stable pointer to all the various segments. This is what the `RefPolicy` does. As a side effect, however, the construction of a "Ref" type array may have a slightly surprising behaviour on first encounter. Namely, an extra axis is always created, even if the a naked single array is passed in. Once a ref-type array has been created, though, any number of views can be constructed without the extra dimension.

### List of array classes

The combination of the above layouts and storage specifier provides 6 fundamental array constructs (these are all interconvertible to view types):
- `NCArrayView`
- `NCArrayRef`
- `NCArray` (Owner type)
- `SOArrayView`
- `SOArrayRef`
- `SOArray` (Owner type)

### Important Limitations

The main limitation of the array classes as currently implemented is that they only support up to 10 dimensions. This design choice was made to simplify the interface when making it compatible with both CPU and GPU. If it becomes an important limitation it may be refactored in the future.

### GPU support

In addition to the above 6 array classes, there are equivalents for cases where the data lives in GPU memory. These arrays include a `Dev` in the name, after the layout specifier. E.g. `NCDevArrayView` or `SODevArray`. Note that GPU support is only available on Linux and Windows at this time.

Array **views** can be passed directly into kernels and device functions. Additionally, there are overloads to make use of kernels for binary addition, subtraction and multiplication (with more planned for release soon). All arrays are trivially convertible to views. You can either pass an array to a view type constructor (e.g. `NCArrayView(NCArray)`), or use the provided the `view()` method.

**NOTE:** Owner type arrays **cannot** be created in device code, even if they are managing device memory. I.e. `NCDevArray` and `SODevArray` are not constructible inside kernels or device functions. This is due to their use of dynamic memory allocation - the GPU heap for malloc based allocations is quite small, and synchronization of allocations is quite complex if creating an array in a running kernel (where many threads may be working in parallel). As such, these arrays use host APIs only.

## Similarities and Differences Between Python and C++ APIs

For the most part, the Python bindings exactly mirror the C++ bindings. There are, however, a small number of (important) differences:

- Array indexing is bounds checked in Python, while use of `operator[]` in C++ is not.

## Roadmap

- Formalize documentation.
- Test suite.
- Fancy indexing on all array types (boolean masking etc).
- `DLPack` suppport
