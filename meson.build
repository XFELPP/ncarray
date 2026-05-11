project(
    'ncarray',
    'cpp',
    version: '0.5.0',
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

nca_cuda_args = [
    '--extended-lambda',
    '--expt-relaxed-constexpr',
    # silence warning: declared but not referenced (e.g. pybind, stl...)
    '-Xcudafe', '--diag_suppress=445',
    # silence warning: long double treated as double
    '-Xcudafe', '--diag_suppress=20208',
    # We make a fat binary for Turing, Ampere, Ada and Hopper
    #'-gencode', 'arch=compute_75,code=sm_75',
    #'-gencode', 'arch=compute_86,code=sm_86',
    #'-gencode', 'arch=compute_80,code=sm_80',
    #'-gencode', 'arch=compute_89,code=sm_89',
    #'-gencode', 'arch=compute_90,code=sm_90',
    '-Xfatbin', '-compress-all',
    #'-gencode', 'arch=compute_75,code=sm_75',
    '-gencode', 'arch=compute_80,code=[sm_80,compute_80]',
    #'-gencode', 'arch=compute_90,code=[sm_90,compute_90]',
]

openmp_dep = dependency('openmp', required: false)
nca_has_openmp = openmp_dep.found()
if nca_has_openmp
    add_project_arguments('-DNCA_HAS_OPENMP', language: ['cpp', 'cuda'])
endif

deps_for_ncarray = []
if nca_has_cuda
    # CUDA driver and nvrtc are both for JIT
    cuda_dep = dependency('cuda', modules: ['cuda', 'cudart', 'nvrtc'])
    deps_for_ncarray += cuda_dep
endif

if nca_has_openmp
    deps_for_ncarray += openmp_dep
endif

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

#### ncarray ####
#################
## Array definitions

ncarray_headers = include_directories('src/lib', 'src/lib/ncarray')

host_src_dir = meson.current_source_dir() / 'src/lib/ncarray/tus/host'
#run_command(
#    py3, 'utilities/make_tus.py', '--output', host_src_dir,
#    check: true,
#)

host_cc = run_command('find', host_src_dir, '-name', '*.cc', check: true).stdout().strip().split('\n')

libncarray = shared_library(
    'ncarray',
    [
        'src/lib/ncarray/host/casts.cc',
        'src/lib/ncarray/ncarrays.cc',
        'src/lib/ncarray/ncarrayrefs.cc',
        'src/lib/ncarray/ncarrayviews.cc',
        'src/lib/ncarray/soarrays.cc',
        'src/lib/ncarray/soarrayrefs.cc',
        'src/lib/ncarray/soarrayviews.cc',
    ] + host_cc,
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
    'src/lib/ncarray/array_element_proxy.hh',
    'src/lib/ncarray/array_impl.hh',
    'src/lib/ncarray/array_operations.hh',
    'src/lib/ncarray/array_traits.hh',
    'src/lib/ncarray/build_macro.hh',
    'src/lib/ncarray/custom_types.hh',
    'src/lib/ncarray/dtype.hh',
    'src/lib/ncarray/engines.hh',
    'src/lib/ncarray/expression.hh',
    'src/lib/ncarray/indexing.hh',
    'src/lib/ncarray/mvnode.hh',
    'src/lib/ncarray/layout.hh',
    'src/lib/ncarray/ncarrays.hh',
    'src/lib/ncarray/op_code.hh',
    'src/lib/ncarray/op_traits.hh',
    'src/lib/ncarray/reductions.hh',
    'src/lib/ncarray/soarrays.hh',
    'src/lib/ncarray/storage.hh',
]

py3.install_sources(
    installed_headers,
    pure: false,
    subdir: 'ncarray/include/ncarray',
)

installed_host_headers = [
    'src/lib/ncarray/host/allocator.hh',
    'src/lib/ncarray/host/casts.hh',
    'src/lib/ncarray/host/elementwise.hh',
    'src/lib/ncarray/host/reductions.hh',
]

py3.install_sources(
    installed_headers,
    pure: false,
    subdir: 'ncarray/include/ncarray/host',
)

installed_engines_headers = [
    'src/lib/ncarray/engines/gpuengine.hh',
    'src/lib/ncarray/engines/hostengine.hh',
]

py3.install_sources(
    installed_headers,
    pure: false,
    subdir: 'ncarray/include/ncarray/engines',
)

if nca_has_cuda
    jit_headers = [
        'src/lib/ncarray/jit/path_utils.hh',
        'src/lib/ncarray/jit/rtcompiler.hh',
    ]

    libncdevarrayjit = shared_library(
        'ncdevarrayjit',
        [
            'src/lib/ncarray/jit/path_utils.cc',
            'src/lib/ncarray/jit/rtcompiler.cc',
        ],
        cuda_args: nca_cuda_args,
        dependencies: cuda_dep,
        include_directories: ncarray_headers,
        install: true,
        install_dir: libncarray_install_dir,
        install_rpath: lib_install_rpath,
        version: '1',
        soversion: '1',
    )

    ncdevarrayjit_dep = declare_dependency(
        link_with: [
            libncdevarrayjit,
        ],
        dependencies: cuda_dep,
        include_directories: ncarray_headers,
    )

    py3.install_sources(
        jit_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray/jit',
    )

    install_subdir(
        'src/lib/ncarray/jit/include/cuda',
        install_dir: py3.get_install_dir() / 'ncarray/include/jit',
    )
    install_subdir(
        'src/lib/ncarray/jit/include/nv',
        install_dir: py3.get_install_dir() / 'ncarray/include/jit',
    )

    dev_src_dir = meson.current_source_dir() / 'src/lib/ncarray/tus/device'
    #run_command(
    #    py3, 'utilities/make_tus.py', '--output', dev_src_dir, '--device',
    #    check: true,
    #)

    dev_cu = run_command('find', dev_src_dir, '-name', '*.cu', check: true).stdout().strip().split('\n')
    # We also only build this lib at all if CUDA is available
    # We make a fat binary for Turing, Ampere, Ada and Hoper
    libncdevarray = shared_library(
        'ncdevarray',
        [
            'src/lib/ncarray/host/casts.cc',
            'src/lib/ncarray/ncdevarrayviews.cu',
            'src/lib/ncarray/ncdevarrayrefs.cu',
            'src/lib/ncarray/ncdevarrays.cu',
            'src/lib/ncarray/sodevarrayviews.cu',
            'src/lib/ncarray/sodevarrayrefs.cu',
            'src/lib/ncarray/sodevarrays.cu',
        ] + dev_cu,
        cuda_args: nca_cuda_args,
        include_directories: ncarray_headers,
        dependencies: [ncdevarrayjit_dep, ncarray_dep],
        install: true,
        install_dir: libncarray_install_dir,
        install_rpath: lib_install_rpath,
        version: '1',
        soversion: '1',
    )

    ncdevarray_dep = declare_dependency(
        link_with: [
            libncdevarray,
            libncdevarrayjit,
        ],
        dependencies: cuda_dep,
        include_directories: ncarray_headers,
    )

    # The following headers are for integration with device code on host side
    installed_device_compat_headers = [
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
        'src/lib/ncarray/device/atomic.cuh',
        'src/lib/ncarray/device/casts.cuh',
        'src/lib/ncarray/device/elementwise.cuh',
        'src/lib/ncarray/device/kernels.cuh',
        'src/lib/ncarray/device/mem_pool.cuh',
        'src/lib/ncarray/device/reductions.cuh',
        'src/lib/ncarray/device/utilities.cuh',
        'src/lib/ncarray/device/warp.cuh',
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
    deps_for_py_mod += cuda_dep
    deps_for_py_mod += ncdevarray_dep
endif

py_bindings_headers = [
    'src/lib/python/binding_builder.hh',
    'src/lib/python/utilities.hh',
]
py3.install_sources(
    py_bindings_headers,
    pure: false,
    subdir: 'ncarray/include/python',
)

libeagereval = shared_library(
    'eagereval',
    [
        'src/lib/python/eager.cc',
    ],
    install: true,
    install_dir: pyncarray_install_dir,
    include_directories: ncarray_headers,
    dependencies : deps_for_py_mod,
    install_rpath: py_to_lib_install_rpath,
    build_rpath: py_to_lib_install_rpath,
    version: '1',
    soversion: '1',
)

eager_dep = declare_dependency(
    link_with: [
        libeagereval,
    ],
    dependencies: deps_for_py_mod,
    include_directories: ncarray_headers,
)


py_mod = py3.extension_module(
    '_pyncarray',
    [
        'src/lib/python/ncarray.cc',
    ],
    install: true,
    install_dir: pyncarray_install_dir,
    include_directories: ncarray_headers,
    dependencies : deps_for_py_mod + eager_dep,
    install_rpath: py_to_lib_install_rpath,
    build_rpath: py_to_lib_install_rpath,
)

if nca_has_cuda
    # NOTE: For faster/more efficient compilation, each GPU specialization is
    # split off into its own translation unit.
    py_dev_mod = py3.extension_module(
        '_pyncdevarray',
        [
            'src/lib/python/ncdevarray.cu',
            'src/lib/python/ncdevarray_view.cu',
            'src/lib/python/ncdevarray_ref.cu',
            'src/lib/python/ncdevarray_owner.cu',
            'src/lib/python/sodevarray_view.cu',
            'src/lib/python/sodevarray_ref.cu',
            'src/lib/python/sodevarray_owner.cu',
        ],
        install: true,
        install_dir: pyncarray_install_dir,
        include_directories: ncarray_headers,
        dependencies : deps_for_py_mod + eager_dep,
        install_rpath: py_to_lib_install_rpath,
        build_rpath: py_to_lib_install_rpath,
        cuda_args: nca_cuda_args,
    )
endif

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
    deps_for_tests += ncdevarray_dep
    deps_for_tests += ncdevarrayjit_dep
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
test('Testing array indexing.', test_index, verbose: true)

test_binary_ops = executable(
    'test_array_binary_operations',
    [
        'tests/cxx/test_binary_operations.cc',
    ],
    dependencies: deps_for_tests,
    install: false,
    include_directories: ncarray_headers,
)
test('Testing array binary operations.', test_binary_ops, verbose: true)

test_unary_ops = executable(
    'test_array_unary_operations',
    [
        'tests/cxx/test_unary_operations.cc',
    ],
    dependencies: deps_for_tests,
    install: false,
    include_directories: ncarray_headers,
)
test('Testing array unary operations.', test_unary_ops, verbose: true)

test_reduction_ops = executable(
    'test_array_reductions',
    [
        'tests/cxx/test_reductions.cc',
    ],
    dependencies: deps_for_tests,
    install: false,
    include_directories: ncarray_headers,
)
test('Testing array reductions.', test_reduction_ops, verbose: true)

test_float_vecs = executable(
    'test_float_vecs',
    [
        'tests/cxx/test_float_vecs.cc',
    ],
    dependencies: deps_for_tests,
    install: false,
    include_directories: ncarray_headers,
)
test('Testing custom float/double vectors.', test_float_vecs, verbose: true)

if nca_has_cuda
    test_gpu_copies = executable(
        'test_gpu_copies',
        [
            'tests/cxx/test_gpu_copies.cu',
        ],
        dependencies: deps_for_tests,
        install: false,
        include_directories: ncarray_headers,
        cuda_args: nca_cuda_args,
    )
    test(
        'Testing GPU array copies.',
        test_gpu_copies,
        env: [
            'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
            'NCARRAY_JIT_INCLUDE_DIR='+meson.project_source_root() / 'src/lib/ncarray/jit/include',
        ],
        verbose: true
    )

    test_gpu_binary_ops = executable(
        'test_gpu_array_binary_operations',
        [
            'tests/cxx/test_gpu_binary_operations.cu',
        ],
        dependencies: deps_for_tests,
        install: false,
        include_directories: ncarray_headers,
        cuda_args: nca_cuda_args,
    )
    test(
        'Testing GPU array binary operations.',
        test_gpu_binary_ops,
        env: [
            'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
            'NCARRAY_JIT_INCLUDE_DIR='+meson.project_source_root() / 'src/lib/ncarray/jit/include',
        ],
        verbose: true
    )

    test_gpu_unary_ops = executable(
        'test_gpu_array_unary_operations',
        [
            'tests/cxx/test_gpu_unary_operations.cu',
        ],
        dependencies: deps_for_tests,
        install: false,
        include_directories: ncarray_headers,
        cuda_args: nca_cuda_args,
    )
    test(
        'Testing GPU array unary operations.',
        test_gpu_unary_ops,
        env: [
            'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
            'NCARRAY_JIT_INCLUDE_DIR='+meson.project_source_root() / 'src/lib/ncarray/jit/include',
        ],
        verbose: true
    )

    test_gpu_reductions = executable(
        'test_gpu_array_reductions',
        [
            'tests/cxx/test_gpu_reductions.cu',
        ],
        dependencies: deps_for_tests,
        install: false,
        include_directories: ncarray_headers,
        cuda_args: nca_cuda_args,
    )
    test(
        'Testing GPU array reductions.',
        test_gpu_reductions,
        env: [
            'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
            'NCARRAY_JIT_INCLUDE_DIR='+meson.project_source_root() / 'src/lib/ncarray/jit/include',
        ],
        verbose: true
    )
endif

#### Example Programs ####
##########################
if nca_has_cuda
    pixel_binning = executable(
        'benchmark_pixel_binning',
        [
            'examples/benchmark_binning.cu',
        ],
        dependencies: deps_for_tests,
        install: true,
        include_directories: ncarray_headers,
        cuda_args: nca_cuda_args,
    )
endif
