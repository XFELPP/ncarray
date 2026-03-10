# ncarray

`ncarray` provides a number of NumPy compatible classes for working with non-contiguous data that cannot be described exclusively by by strides.

NumPy is extraordinarily powerful and flexible, but it does not support PEP3118 **sub-offsets** which are the general Python mechanism for dealing with array-like data spread out in memory. `ncarray` provides support for both sub-offsets, as well as additional specializations where the "pointer axis" is known in advance, and the general flexibility of sub-offsets is not needed.

This is a small extension, the scope of which is restricted to providing a way of representing these non-contiguous ararys in Python, and then facilitating their translation to NumPy. Any array operations which would result in a copy (such as arithmetic) by default return NumPy arrays to bring users back into that eco-system (although this can optionally be turned off).

## Who is this for?

The classes in this extension are quite specialized. The use case is for dealing with memory coming from external sources (C/C++ code, as an example), where you do not want to make an initial copy (e.g. for performance reasons). `ncarray` objects provide a way to avoid an initial copy into a NumPy array, while giving you the flexibility to do so later when it is necessary.
