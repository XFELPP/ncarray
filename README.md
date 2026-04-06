# ncarray
[![Release build](https://github.com/XFELPP/ncarray/actions/workflows/release.yml/badge.svg)](https://github.com/XFELPP/ncarray/actions/workflows/release.yml)[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

## Status
This project should be considered to be in an early alpha stage. The project is still in the initial stages of development. Internals may change from release to release (which are coming rapidly). However, beginning from version `0.4.0`, the library can be considered usable. Most operations are available for arrays (on both CPU and GPU, for supported platforms). A small set of common reduction routines (sum, max, etc.) are also provided. The high-level APIs and interfaces are likely to be fairly stable (with new releases bringing additional features only). Internal source code may change without notice from release to release though. The general organization of headers may change as well. (Some specific notes for C++ developers are below). If staying within Python (or limiting yourself to solely `ncarrays.hh` etc), you should be okay.

The library is still lacking documentation, and while the tests cover the basics, their coverage is rather sparse.

## Installation
It is recommended to get built versions from `PyPI`. Builds are available for Mac, Linux and Windows. The CUDA compatibility is included in Linux and Windows builds.

```bash
> pip install ncarray
```

The necessary headers and shared libraries are also included in the wheels, if working in C++ or building new extensions. Two utility functions are provided at the top of the `ncarray` package to assist in locating the correct binaries and files:
```py
import ncarray

# Headers:
path_to_headers: str = ncarray.get_include()

# Shared libs:
path_to_libs: str = ncarray.get_lib_dir()
```

The above can be incorporated into your build system as needed.

**NOTE:** In the C++ headers, many operator overloads have a corresponding named alternative. E.g. `operator+()` and `add()` both exist and have the same functionality. The distributed shared libraries do NOT have template specializations for the latter named variants. This helps keep the size of the files down (and helps reduce compile time a bit). The Python bindings only use the `operator+` style calls. This is just to keep it in mind should you prefer using them - they will need to be compiled from scratch.

### Building from source
You can also build directly from source in this repo. The `build.sh` script may or may not work on your platform (likely yes on Linux/Mac, likely no on Windows). The project uses meson, though, and building yourself is essentially:
```bash
> meson setup $MY_BUILD_DIR # --prefix=$MY_INSTALL_DIR #-Dbuildtype=debug # (or release etc).
> meson compile -C $MY_BUILD_DIR
> meson install -C $MY_BUILD_DIR
# For tests:
> meson test -C $MY_BUILD_DIR
```

You can also run `pip install .` directly - this will be very slow though. For wheel creation, the meson calls are done as above, and a final `pip install` is run at the end. See `build.sh` for the mechanism.

**NOTE:** When building GPU support, this can be extremely painful in Python bindings. The strategy adopted by the project is to instantiate specializations for all operations (see `build_macro.hh`). This creates two shared libraries, one for host arrays and one for device arrays. Afterwards, the Python extensions are built. `nvcc` seems to handle the templates within the project fairly well, but the combination with `pybind11` (also a very template heavy project) leads to astronomical build times. If just building C++ libraries, including `array_impl.hh` and `array_operations.hh` should be fine, and no additional tricks should be needed.

#### Requirements
This should build with `gcc 13+` and `clang 15+`. Meson `1.10.1` or greater is recommended, although earlier versions should work fine with the explicit exception of version `1.10.0`.

GPU support is tested with `CUDA 12.8` - the entire `12.x` series will likely be work, and `13.x` is likely okay as well.

### Wheels from GitHub

The wheels and associated assets are also provided in the GitHub releases if not using `pip`.

## Overview

`ncarray` provides a number of C++ array classes, compatible with NumPy, for working with non-contiguous data that cannot be described exclusively by strides. Specifically, these classes deal with data described by pointer-to-pointer setups/tables (double pointers, i.e. suboffsets in the Python world).

The library provides views for views of data on the CPU and GPU (Linux/Windows only). When working on the device, array objects can be passed directly into kernels and device functions - if they are views. Owner-type arrays must be worked with from the host, but a view can be created using the `.view()` function.

## Who is this for?

NumPy is extraordinarily powerful and flexible; however, it is designed for dealing with strided memory layouts. When dealing with disjoint collections of data spread out over, you would typically be forced to copy each section into a contiguous buffer to consider the entire set as a single array. When dealing with a large number of arrays, or under tight performance constraints, this may be prohibitively costly. `ncarray` is intended to provide a wrapper (known as a span, or view in many languages and libraries) over these disjoint arrays, without needing any initial copy.

## Features

`ncarray` is a C++20/23 (GPU/CPU) generic library. Python bindings are provided by `pybind11`, although the C++ library can be used standalone, or wrapped using other techniques. The current set of features are:

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

import ncarray as nca # Alias it if you'd like

# Construct a disjoint set of arrays - subarrays that are really part of 1 larger one
subarray_list: List[npt.NDArray[np.uint32]] = [
    np.random.randint(1, 255, size=(512,1024), dtype=np.uint16) for _ in range(10)
]

# Create a wrapped reference
ncarr: nca.NCArrayRef = nca.NCArrayRef(subarray_list)

# Index it
first_two_ptrs: nca.NCArrayView = ncarr[:2]

# Down to scalar if you'd like
my_int: int = ncarr[2, 1, 2]

# Scalar broadcasts -- This creates a new OWNING array! (NCArray)
# Supported: +, -, *, / (Will provide % and //, i.e. int division in the future)
scalar_bcast_res: nca.NCArray = ncarr + 2

# Array arithmetic, supporting +, -, *, / as above
# Require identical shapes, currently!
# Supportable broadcasts may come later (E.g. a row/col into a 2D array)
other_res: nca.NCArray = ncarr + ncarr

# Perform operations on it -- also create new OWNING arrays
# Any ufunc *should* work -- this builds NumPy arrays, however.
# Because of this, these operations are rather slow.
result: nca.NCArray = np.sin(ncarr)
```

## Concepts and Terminology

The `ncarray` library provides a series of array types. These are constructed through the composition of a `Layout` and a `Storage` policy specifier in a base `ArrayImpl` class. At the Python level, the base classes are not made available, and instead, bindings are provided for a number of specializations listed below.

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

The main limitation of the array classes as currently implemented is that they only support up to 10 dimensions. This design choice was made to simplify the interface when making it compatible with both CPU and GPU. If it becomes an important limitation it may be refactored in the future (or a build option will be provided to allow extending the limit).

### GPU support

In addition to the above 6 array classes, there are equivalents for cases where the data lives in GPU memory. These arrays include a `Dev` in the name, after the layout specifier. E.g. `NCDevArrayView` or `SODevArray`. Note that GPU support is only available on Linux and Windows at this time.

Array **views** can be passed directly into kernels and device functions. Additionally, there are overloads to make use of kernels for binary addition, subtraction and multiplication (with more planned for release soon). All arrays are trivially convertible to views. You can either pass an array to a view type constructor (e.g. `NCArrayView(NCArray)`), or use the provided `view()` method.

**NOTE:** Owner type arrays **cannot** be created in device code, even if they are managing device memory. I.e. `NCDevArray` and `SODevArray` are not constructible inside kernels or device functions. This is due to their use of dynamic memory allocation - the GPU heap for malloc based allocations is quite small, and synchronization of allocations is quite complex if creating an array in a running kernel (where many threads may be working in parallel). As such, these arrays use host APIs only.

### Should I use `NCArray...` or `SOArray...` ?

The answer to this question boils down to the use-case. If you only require a single "pointer axis", for example because you have a collection of otherwise contigiuous array-like objects, then the `NCArray...` style arrays are preferrable. In general, in this scenario your first axis will then be the "pointer axis". Using the NC-style arrays means you can perform operations like iterating over the first axis of the array where each step returns the contiguous sub-array. If, on the other hand, you require multiple "pointer axes", then the decision is already taken for you - you must use `SOArray...` style arrays.

## Similarities and Differences Between Python and C++ APIs

For the most part, the Python bindings exactly mirror the C++ bindings. There are, however, a small number of (important) differences:

- Array indexing is bounds checked in Python, while use of `operator[]` in C++ is not.
- There are a number of indexing strategies in C++, whereas these styles and overloads do not exist in Python (only `__getitem__`, i.e. `arr[...]` is used).
- In Python, 0-D (i.e. scalar) arrays are cast to the corresponding scalar. This is not done automatically in C++.
- In C++, in addition to the operator overloads (like `operator+`), there are named functions if you find them easier (e.g. `NCArrayView::add(other)`). For keeping binary sizes down, though, these are NOT pre-instantiated and compiled into the shared libs.

## For C++ Developers
### Headers
The main array specializations are exposed through `ncarrays.hh`, `soarrays.hh` (and `ncdevarrays.cuh` and `sodevarrays.cuh` for CUDA-supported platforms). These are paired with corresponding `.cc` (`.cu`) files to provide shared libraries for linking (this is a (very) template heavy library - compilation times can be long). If including only these headers, the library should be safe for usage across releases. The internal headers described below may be reorganized at any moment through the period of releases outlined in the timeline below (after which, some stability can be expected).

#### Architecture
The principle components of the library are defined in:

- `layout.hh`: Describes memory layout of arrays
- `storage.hh`: Describes the storage policies of arrays (view-like, or owner, e.g.)
- `dtype.hh`: Contains the the type system for the library
- `array_impl.hh` and `array_operations.hh` contain the `ArrayImpl` specification and the definitions of its operations. These two headers comprise the core of the library. If including them directly, they must be included together and in that order!
- `array_traits.hh` contain concepts for array operations. These also contain operator traits for all the supported datatypes (e.g. to provide greater/less than functionality for `complex` numbers and so on).
- `device/...` and `host/...` contain GPU/CPU specific implementations of the operations for arrays.
- `engines.hh` wrap the specific implementations together for easier dispatch.

**NOTE:** If also building Python extensions (and, specifically, if using `pybind11`), see the note above under the installation section. You may want to use `build_macro.hh` as a guide for instantiating your template specializations in the Python module, otherwise build times will become untenable. For pure C++ projects this should not be an issue (including `array_impl.hh` and `array_operations.hh` should lead to builds of a couple minutes - even for device code compiled with `nvcc`).

### Usage
The following snippets highlight the basic functionality and semantics of using `ncarray` in C++.

```cpp
// All headers are included in appropriate namespaces (as used in the libraries source)
#include "ncarray/ncarrays.hh" // You can get everything at the high-level with just this
//#include "ncarray/soarrays.hh" // If you need suboffset support

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cstdint>
#include <vector>

// Make some dummy non-contiguous data
std::vector<std::int32_t> data(10, 42);
std::vector<ssize_t> shape { 2, 5 };
std::vector<ssize_t> strides { 5 * sizeof(std::int32_t), sizeof(std::int32_t) };
std::vector<void*> ptrs { data.data() };

// Create a view from this data
std::vector<ssize_t> offsets(10);
ncarray::DType dtype = ncarray::DType::int32;
// You can get the dtype from traits too:
// ncarray::dtype_traits<std::int32_t>::value;
auto view = ncarray::NCArrayView(ptrs.data(),
                                 shape.data(),
                                 strides.data(),
                                 offsets.data(),
                                 dtype,
                                 /*pointer_axis=*/0,
                                 /*read_only=*/false);
// NOTE: You can make a reference type, but this is generally not needed in C++ (only Python)
// ncarray::NCArrayRef ref(ptrs, shape, strides, ncarray::DType::int32, 0, false);

// NOTE: You can also use SOArrayView if needed. This implements PEP3118 compliant
// suboffsets. It is more flexible (arbitrary number of pointer axes, while above is limited to 1).
// However, it may be more difficult to reason about. That said, the library deals
// with the complexities for you (after proper creation, that is).

// Get general information about your array
view.ndim(); // 2
view.shape(); // ssize_t* with 2 items. (2, 5)
view.size(); // 10
view.itemsize(); // 4 (bytes)
view.nbytes(); // 40 (size() * itemsize())

// Indexing comes in three variaties - some suggested typedefs shown as well
// 1. Most generic for slicing (This is C++23, and thus host only -- see below for GPU equivalent)
using sl = ncarray::Slice;
auto subview = view[sl(0,1),sl(2,4)]; // New shape = (1, 2)

// 2. Index to an element, with reference semantics
std::int32_t& item13 = view(1,3);
// NOTE: If you need the type, using auto here won't work (auto& item = view(...))
// The above does NOT type check. If you need type checking use:
// auto& proxy = view(1,3); proxy.get<T>();

// Proxy references are assignable
view(1, 2) = 12;
// Can also hold on to the proxy if desired:
// item12 = view(1,2); item12 = 12;

// 3. You can use linearized indexing to a reference as well
ssize_t idx = 2; // IMPORTANT: *Must* be ssize_t
std::int32_t& item02 = view[idx];

// We can also create owner type arrays.
std::vector<ssize_t> shape { 4, 4, 4 };
ncarray::NCArray arr1(shape, ncarray::DType::float32);
ncarray::NCArray arr2(shape, ncarray::DType::float32);

// Can fill rapidly with a value
arr1.fill(2.0f);
arr2.fill(4.0f);

// Can broadcast one array type into another
// arr1.assign(arr2); // Will cast as needed.

// NOTE: Operations below all assume same shape currently.
// Binary operations
auto sum_owner = arr1 + arr2; // These return *owner* type arrays

// Inplace operations
arr1 += arr2;

// Broadcast with a scalar
arr2 += 4.2f;

// Comparison operations
auto res_eq = (arr1 == arr2) // All false here. Has shape of inputs
// Also available: >, <, >=, <=, &&, !, ||
// For logical operators, inplace variants are only available for boolean arrays.

// Reductions to scalar
auto arr_sum = arr1.sum(); // or .max(), .min()
// The above is a variant. You must use: std::get<float>(a_sum) (or other ops in <variant>)
```

### GPU Considerations
All operations defined above are available on GPU as well. The array types are simply prefixed with a `Dev` instead. E.g.:
```
NCArray <-> NCDevArray
NCArrayView <-> NCDevArrayView
... and so on ...
```

**Important:** You must ensure that your build system uses `nvcc` to compile the operations with device arrays. They will compile fine as standard C++, but will have odd behaviours and will not consistently execute operations on GPU. You will likely also encounter runtime exceptions (along the lines of: `Fatal: tried to compile CUDA kernel as C++`).

As described above, array **views** can be passed into kernels, and used anywhere in device code. It is worth reiterating that owners (`NCDevArray` and `SODevArray`) are constructible ONLY on the host. This is due to complexities with dynamic allocations in device code, the small size of the heap and so on. Any owner is trivially convertible to a view, however (using all common constructors). There is also a helper function to make intent clear:

```cpp
auto ncarr = NCDevArray(...);

my_kernel<<<blocks, TPB>>>(ncarr.view());
```

Some language/syntax features available on the host are not available on GPU. This is because the host-side code takes advantage of C++23 features, whereas only up to C++20 is available for CUDA code. The main limitation is the fancy indexing with variadic `operator[]`. As an alternative, a slightly more verbose, but very specific, mechanism is available:
```cpp
ncarray::NCDevArray dev_arr(...)

using Idx = ncarray::IndexItem;
using sl = ncarray::Slice;
Idx region[] = {
  Idx(sl(1, 3)),
  Idx(sl(1, 3))
};
auto view = dev_arr.view_from_indices(region, /*num_indices=*/2);
```
In the future, an overload for `operator()` may be provided to support slices. The main reason it does not currently exist is the distinction between `operator()` and `operator[]`. The former provides access to individual elements by reference, the latter is intended for construction of views (sub-views).

## Roadmap

The current plan for releases, in order is:

1. `v0.4.0` - Arithmetic, logical, and comparison operations (i.e. in place operations `/=` e.g., and `>`, `<` `&&` and so on).

  - This release includes additional improvements in CUDA support including generalized atomics to support these operations.

2. `v0.5.0` - Additional reduction operations (e.g. standard deviation `std`), and shape/axis-aware implementations (e.g. `ncarr.sum(axis=0)`. Other operations include reduction by key.
3. `v0.6.0` - Boolean indexing and conditional selection with something like a `where` function.

  - Additional indexing improvements are planned for this stage as well. Multi-dimensional sliced assignments for instance. Broadcasting support, etc.

4. `v0.7.0` - CuPy integration
5. `v0.8.0` - `DLPack` integration

The above road-map is subject to change. Throughout the process, test coverage and documentation will be improved. Upon initial implementation of the above feature sets, focus will be switched to stabilizing the library.
