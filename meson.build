project(
    'ncarray',
    'cpp',
    version: '0.2.0',
    default_options: [
        'cpp_std=c++23',
        'cuda_std=c++20',
        'optimization=3',
    ],
)
add_languages('cpp', native: true, required: true)
nca_has_cuda = add_languages('cuda', required: false)
if nca_has_cuda
    # Source code guards CUDA specific features behind NCA_HAS_CUDA
    add_project_arguments('-DNCA_HAS_CUDA', language: ['cpp', 'cuda'])
endif

add_project_arguments('-frtti', language: 'cpp')

#### Main Python Setup ####
###########################

pymod = import('python')

py3 = pymod.find_installation(pure: false)
py3_dep = py3.dependency()

min_python_ver = '3.8'
py_ver = py3.language_version()
if py_ver.version_compare(f'<@min_python_ver@')
    error(f'Minimum supported Python version is @min_python_ver@, found @py_ver@')
endif

pybind11_dep = dependency('pybind11', required: true)

#### Path Setup ####
####################

py_install_dir = py3.get_install_dir()

ncarray_lib_install_dir = py3.get_install_dir() / 'ncarray/lib'

# Package and library setup will be:
# site-packages/
#    | - ncarray
#           | - lib
#           |    | - libncarray.so
#           |
#           | - core
#           |    | - _pyncarray.so
#           |
#           | - __init__.py (utility functions and expose core lib)

libncarray_install_dir = py_install_dir / 'ncarray/lib'
pyncarray_install_dir = py_install_dir / 'ncarray/core'
py_to_lib_path = '../lib'

# Executables get installed in $prefix/bin while the libs are in $prefix/lib/...
# So the relative path is:
exe_to_libncarray = '..' / libncarray_install_dir

lib_install_rpath = '$ORIGIN'

abs_exe_rpath = py_install_dir / exe_to_libncarray
abs_lib_rpath = py_install_dir / py_to_lib_path

# Setup RPATH for various platforms, for any executables that may need it
if host_machine.system() == 'darwin'
    exe_install_rpath = '@executable_path' + ':' + ('@executable_path' / exe_to_libncarray) + ':' + abs_exe_rpath
    py_to_lib_install_rpath = '@loader_path' + ':' + ('@loader_path' / py_to_lib_path)  + ':' + abs_lib_rpath
else
    exe_install_rpath = '$ORIGIN' + ':' + ('$ORIGIN' / exe_to_libncarray)  + ':' + abs_exe_rpath
    py_to_lib_install_rpath = '$ORIGIN' + ':' + ('$ORIGIN' / py_to_lib_path) + ':' + abs_lib_rpath
endif

#### Dependencies ####
######################
deps_for_ncarray = []
if nca_has_cuda
    cuda_dep = dependency('cuda', modules: ['cudart'])
    deps_for_ncarray += cuda_dep
endif

#### ncarray ####
#################
## Array definitions

ncarray_headers = include_directories('src/lib', 'src/lib/ncarray')

libncarray = shared_library(
    'ncarray',
    [
        'src/lib/ncarray/ncarrays.cc',
        'src/lib/ncarray/soarrays.cc',
    ],
    dependencies: deps_for_ncarray,
    include_directories: ncarray_headers,
    install: true,
    install_dir: libncarray_install_dir,
    install_rpath: lib_install_rpath,
    version: '1',
    soversion: '1',
)

ncarray_dep = declare_dependency(
    link_with: [
        libncarray,
    ],
    include_directories: ncarray_headers,
)

# Install headers into the package for easy bundling
installed_headers = [
    'src/lib/ncarray/array_impl.hh',
    'src/lib/ncarray/array_operations.hh',
    'src/lib/ncarray/array_traits.hh',
    'src/lib/ncarray/dtype.hh',
    'src/lib/ncarray/indexing.hh',
    'src/lib/ncarray/layout.hh',
    'src/lib/ncarray/ncarrays.hh',
    'src/lib/ncarray/soarrays.hh',
    'src/lib/ncarray/storage.hh',
]

py3.install_sources(
    installed_headers,
    pure: false,
    subdir: 'ncarray/include/ncarray',
)


if nca_has_cuda
    # We also only build this lib at all if CUDA is available
    # We make a fat binary for Turing, Ampere, Ada and Hoper
    libncdevarray = shared_library(
        'ncdevarray',
        [
            'src/lib/ncarray/ncdevarrays.cu',
            'src/lib/ncarray/sodevarrays.cu',
        ],
        cuda_args: [
            '--extended-lambda',
            '--expt-relaxed-constexpr',
            # silence warning: declared but not referenced (e.g. pybind, stl...)
            '-Xcudafe', '--diag_suppress=445',
             # silence warning: long double treated as double
            '-Xcudafe', '--diag_suppress=20208',
            '-gencode', 'arch=compute_75,code=sm_75',
            '-gencode', 'arch=compute_80,code=sm_80',
            '-gencode', 'arch=compute_89,code=sm_89',
            '-gencode', 'arch=compute_90,code=sm_90',
        ],
        include_directories: ncarray_headers,
        install: true,
        install_dir: libncarray_install_dir,
        install_rpath: lib_install_rpath,
        version: '1',
        soversion: '1',
    )

    ncdevarray_dep = declare_dependency(
        link_with: [
            libncdevarray,
        ],
        dependencies: cuda_dep,
        include_directories: ncarray_headers,
    )

    # The following headers are for integration with device code on host side
    installed_device_compat_headers = [
        'src/lib/ncarray/engines.hh',
        'src/lib/ncarray/ncdevarrays.cuh',
        'src/lib/ncarray/sodevarrays.cuh',
    ]

    py3.install_sources(
        installed_device_compat_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray',
    )

    # The following headers are device only as well as host-called kernels
    installed_device_headers = [
        'src/lib/ncarray/device/elementwise.cuh',
        'src/lib/ncarray/device/kernels.cuh',
        'src/lib/ncarray/device/reductions.cuh',
    ]

    py3.install_sources(
        installed_device_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray/device',
    )
endif # end nca_has_cuda

# Figure out dependencies
deps_for_py_mod = [ncarray_dep, pybind11_dep]

if nca_has_cuda
    deps_for_py_mod += ncdevarray_dep
endif

py_mod = py3.extension_module(
    '_pyncarray',
    [
        'src/lib/python/ncarray.cc'
    ],
    install: true,
    install_dir: pyncarray_install_dir,
    include_directories: ncarray_headers,
    dependencies : deps_for_py_mod,
    install_rpath: py_to_lib_install_rpath,
    build_rpath: py_to_lib_install_rpath,
)

py_sources = [
    'src/ncarray/__init__.py',
]

py3.install_sources(
    py_sources,
    pure: false,
    subdir: 'ncarray',
)

placeholders = [
    'src/ncarray/core/__init__.py',
]

py3.install_sources(
    placeholders,
    pure: false,
    subdir: 'ncarray/core',
)

#### Testing ####
#################
gtest_dep = dependency('gtest', main: true, static: true, required: false)

if not gtest_dep.found()
    gtest_proj = subproject('gtest', default_options: ['default_library=static'])
    gtest_dep = gtest_proj.get_variable('gtest_main_dep')
endif

deps_for_tests = [gtest_dep, ncarray_dep]

if nca_has_cuda
    deps_for_tests += cuda_dep
endif

test_index = executable(
    'test_array_indexing',
    [
        'tests/cxx/test_indexing.cc',
    ],
    dependencies: deps_for_tests,
    install: false,
    include_directories: ncarray_headers,
)
test('Testing array indexing.', test_index)

test_ops = executable(
    'test_array_operations',
    [
        'tests/cxx/test_operations.cc',
    ],
    dependencies: deps_for_tests,
    install: false,
    include_directories: ncarray_headers,
)
test('Testing array operations.', test_ops)
