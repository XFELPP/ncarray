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

#### Fetch Build Options ####
#############################
# Core options
build_core_opt = get_option('build_core')
build_python_opt = get_option('build_python')
nca_cuda_archs_opt = get_option('nca_cuda_archs')
nca_as_wheel_opt = get_option('nca_as_wheel')

# Add-ons and development
build_examples_opt = get_option('build_examples')

#### Parse CUDA architectures and set CUDA args ####
####################################################

nca_cuda_args = [
    '--extended-lambda',
    '--expt-relaxed-constexpr',
    # silence warning: declared but not referenced (e.g. pybind, stl...)
    '-Xcudafe', '--diag_suppress=445',
    # silence warning: long double treated as double
    '-Xcudafe', '--diag_suppress=20208',
    '-static-global-template-stub=false',
    '-device-entity-has-hidden-visibility=false',
    '-Xfatbin', '-compress-all',
]

if nca_cuda_archs_opt == 'all'
    # Setup a fat binary for Turing, Ampere, Ada and Hopper
    nca_cuda_args += [
        '-gencode', 'arch=compute_75,code=sm_75',
        '-gencode', 'arch=compute_80,code=sm_80',
        '-gencode', 'arch=compute_86,code=sm_86',
        '-gencode', 'arch=compute_89,code=sm_89',
        '-gencode', 'arch=compute_90,code=[sm_90,compute_90]',
    ]
else
    # Parse comma-separated architectures ('80', or '75,90')
    # Last value will generate [sm_ARCH,compute_ARCH]
    selected_cuda_archs = nca_cuda_archs_opt.split(',')

    num_archs = selected_cuda_archs.length()
    last_arch = selected_cuda_archs[num_archs - 1].strip()
    foreach arch : selected_cuda_archs
        arch = arch.strip()
        if arch != ''
            if arch == last_arch
                # Generate PTX in this case, for forward-compatibility
                nca_cuda_args += [
                    '-gencode', 'arch=compute_' + arch + ',code=[sm_' + arch + ',compute_' + arch + ']'
                ]
            else
                nca_cuda_args += [
                    '-gencode', 'arch=compute_' + arch + ',code=sm_' + arch
                ]
            endif
        endif
    endforeach
endif

#### Core Dependencies ####
###########################

openmp_dep = dependency('openmp', required: false)
nca_has_openmp = openmp_dep.found()
if nca_has_openmp
    add_project_arguments('-DNCA_HAS_OPENMP', language: ['cpp', 'cuda'])
endif

cpp = meson.get_compiler('cpp')

deps_for_ncarray = []
if nca_has_cuda
    # Add libdl as needed for JIT if not merged directly into the cstdlib
    dl_dep = cpp.find_library('dl', required: false)
    if dl_dep.found()
        deps_for_ncarray += dl_dep
    endif

    # CUDA driver and nvrtc are both for JIT
    cuda_dep = dependency('cuda', modules: ['cuda', 'cudart', 'nvrtc'])
    deps_for_ncarray += cuda_dep
endif

if nca_has_openmp
    deps_for_ncarray += openmp_dep
endif

if build_python_opt
    pybind11_dep = dependency('pybind11', required: true)
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

#### Path Setup ####
####################

is_windows = host_machine.system() == 'windows'
if is_windows
    py_install_dir = py3.get_install_dir(pure: true)
else
    py_install_dir = py3.get_install_dir(pure: false)
endif

pyncarray_install_dir = py_install_dir / 'ncarray/core'

if nca_as_wheel_opt
    # Install C++ libraries inside the wheel structure
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
    header_install_dir = py_install_dir / 'ncarray/include'

    # Relative path from 'ncarray/core' to 'ncarray/lib'
    py_to_lib_path = '../lib'

    exe_prefix = py_install_dir
    lib_prefix = pyncarray_install_dir
else
    # Normal install path
    libncarray_install_dir = get_option('libdir')
    header_install_dir = get_option('includedir')

    # Relative path from '$PREFIX/lib/pythonMAJ.MIN/site-packages/ncarray/core' to '$PREFIX/lib'
    py_to_lib_path = '../../../../'

    exe_prefix = get_option('prefix') / 'bin'
    lib_prefix = pyncarray_install_dir
endif

# For both the wheel and non-wheel case, executables relative path, for
# $PREFIX/bin to $PREFIX/lib is
exe_to_libncarray = '..' / libncarray_install_dir

lib_install_rpath = '$ORIGIN'

abs_exe_rpath = exe_prefix / exe_to_libncarray
abs_lib_rpath = lib_prefix / py_to_lib_path

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

if build_core_opt
    host_src_dir = meson.current_source_dir() / 'src/lib/ncarray/tus/host'

    # Create the translation units if they don't already exist.
    # If running manually and need to regenerate them, run the script by hand
    check_host_tus = run_command(
        py3, '-c', 'import glob, sys; sys.exit(0 if glob.glob("' + host_src_dir + '/*.cc") else 1)',
        check: false,
    )
    if check_host_tus.returncode() != 0
        run_command(
            py3, 'utilities/make_tus.py', '--output', host_src_dir,
            check: true,
        )
    endif

    host_cc = run_command(
        py3, '-c', 'import glob; print("\\n".join(glob.glob("' + host_src_dir + '/*.cc")))',
        check: true
    ).stdout().strip().split('\n')

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
        'src/lib/ncarray/indexing.hh',
        'src/lib/ncarray/layout.hh',
        'src/lib/ncarray/ncarrays.hh',
        'src/lib/ncarray/op_code.hh',
        'src/lib/ncarray/op_traits.hh',
        'src/lib/ncarray/reductions.hh',
        'src/lib/ncarray/soarrays.hh',
        'src/lib/ncarray/storage.hh',
    ]

    install_headers(
        installed_headers,
        install_dir: header_install_dir / 'ncarray',
    )

    installed_host_headers = [
        'src/lib/ncarray/host/allocator.hh',
        'src/lib/ncarray/host/casts.hh',
        'src/lib/ncarray/host/elementwise.hh',
        'src/lib/ncarray/host/reductions.hh',
    ]

    install_headers(
        installed_host_headers,
        install_dir: header_install_dir / 'ncarray/host',
    )

    installed_host_engine_headers = [
        'src/lib/ncarray/engines/hostengine.hh',
    ]

    install_headers(
        installed_host_engine_headers,
        install_dir: header_install_dir / 'ncarray/engines',
    )

    installed_expression_headers = [
        'src/lib/ncarray/expression/dynamicmvnode.hh',
        'src/lib/ncarray/expression/interface.hh',
        'src/lib/ncarray/expression/mvnode.hh',
        'src/lib/ncarray/expression/staticmvnode.hh',
    ]

    install_headers(
        installed_expression_headers,
        install_dir: header_install_dir / 'ncarray/expression',
    )

    if nca_has_cuda
        installed_gpu_engine_headers = [
            'src/lib/ncarray/engines/gpuengine.hh',
        ]

        install_headers(
            installed_gpu_engine_headers,
            install_dir: header_install_dir / 'ncarray/engines',
        )

        # Currently, Stencil support is for GPU only
        installed_expression_stencil_headers = [
            'src/lib/ncarray/expression/stencil.hh',
        ]

        install_headers(
            installed_expression_stencil_headers,
            install_dir: header_install_dir / 'ncarray/expression',
        )

        libncdevarrayjit = shared_library(
            'ncdevarrayjit',
            [
                'src/lib/ncarray/jit/path_utils.cc',
                'src/lib/ncarray/jit/jit_utils.cc',
                'src/lib/ncarray/jit/rtcompiler.cc',
            ],
            cuda_args: nca_cuda_args,
            dependencies: [cuda_dep, dl_dep],
            include_directories: ncarray_headers,
            install: true,
            install_dir: libncarray_install_dir,
            install_rpath: lib_install_rpath,
            version: '1',
            soversion: '1',

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
        )

        ncdevarrayjit_dep = declare_dependency(
            link_with: [
                libncdevarrayjit,
            ],
            dependencies: [cuda_dep, dl_dep],
            include_directories: ncarray_headers,
        )

        installed_jit_headers = [
            'src/lib/ncarray/jit/path_utils.hh',
            'src/lib/ncarray/jit/jit_utils.hh',
            'src/lib/ncarray/jit/rtcompiler.hh',
        ]

        install_headers(
            installed_jit_headers,
            install_dir: header_install_dir / 'ncarray/jit',
        )


        #### Install CCCL Headers for JIT ####
        # Will run a little Python script to copy the current CUDA toolkit CCCL headers
        # Afterwards, these will be installed into the wheel for use by the JIT compiler
        jit_headers_build_dir = meson.current_build_dir() / 'jit_tmp_headers'

        run_command(
            py3, 'utilities/copy_cccl_headers.py', jit_headers_build_dir,
            check: true,
        )

        # Install CCCL headers into the wheel
        install_subdir(
            jit_headers_build_dir / 'cuda',
            install_dir: header_install_dir / 'ncarray/jit',
        )
        install_subdir(
            jit_headers_build_dir / 'nv',
            install_dir: header_install_dir / 'ncarray/jit',
        )

        dev_src_dir = meson.current_source_dir() / 'src/lib/ncarray/tus/device'

        # Create the translation units if they don't already exist.
        # If running manually and need to regenerate them, run the script by hand
        check_device_tus = run_command(
            py3, '-c', 'import glob, sys; sys.exit(0 if glob.glob("' + dev_src_dir + '/*.cu") else 1)',
            check: false,
        )
        if check_device_tus.returncode() != 0
            run_command(
                py3, 'utilities/make_tus.py', '--output', dev_src_dir, '--device',
                check: true,
            )
        endif

        dev_cu = run_command(
            py3, '-c', 'import glob; print("\\n".join(glob.glob("' + dev_src_dir + '/*.cu")))',
            check: true
        ).stdout().strip().split('\n')

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

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
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

        install_headers(
            installed_device_compat_headers,
            install_dir: header_install_dir / 'ncarray',
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

        install_headers(
            installed_device_headers,
            install_dir: header_install_dir / 'ncarray/device',
        )

    endif # end nca_has_cuda
else
    libncarray = cpp.find_library('ncarray', required: true)
    ncarray_dep = declare_dependency(
        dependencies: deps_for_ncarray + [
            libncarray,
        ],
        include_directories: ncarray_headers,
    )

    if nca_has_cuda
        libncdevarray = cpp.find_library('ncdevarray', required: true)
        libncdevarrayjit = cpp.find_library('ncdevarrayjit', required: true)
        ncdevarrayjit_dep = declare_dependency(
            dependencies: deps_for_ncarray + [
                libncdevarrayjit,
            ],
            include_directories: ncarray_headers,
        )
        ncdevarray_dep = declare_dependency(
            dependencies: deps_for_ncarray + [
                ncarray_dep,
                ncdevarrayjit_dep,
                libncdevarray,
            ],
            include_directories: ncarray_headers,
        )
    endif
endif # end build_core_opt

if build_python_opt
    # Figure out dependencies
    deps_for_py_mod = [ncarray_dep, pybind11_dep, py3_dep]

    if nca_has_cuda
        deps_for_py_mod += cuda_dep
        deps_for_py_mod += ncdevarray_dep
    endif

    py_bindings_headers = [
        'src/lib/python/binding_builder.hh',
        'src/lib/python/utilities.hh',
    ]

    if nca_as_wheel_opt
        # NOTE: Only installing the bindings specific headers if in a wheel.
        # The include path would be confusing in a global install directory.
        install_headers(
            py_bindings_headers,
            install_dir: header_install_dir / 'python',
        )
    endif

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
        override_options: ['b_lundef=false'],
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

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
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
endif # build_python_opt

if build_core_opt
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

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
        )
        test(
            'Testing GPU array copies.',
            test_gpu_copies,
            env: [
                'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
                'NCARRAY_JIT_INCLUDE_DIR=' + jit_headers_build_dir,
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

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
        )
        test(
            'Testing GPU array binary operations.',
            test_gpu_binary_ops,
            env: [
                'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
                'NCARRAY_JIT_INCLUDE_DIR=' + jit_headers_build_dir,
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

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
        )
        test(
            'Testing GPU array unary operations.',
            test_gpu_unary_ops,
            env: [
                'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
                'NCARRAY_JIT_INCLUDE_DIR=' + jit_headers_build_dir,
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

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
        )
        test(
            'Testing GPU array reductions.',
            test_gpu_reductions,
            env: [
                'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
                'NCARRAY_JIT_INCLUDE_DIR=' + jit_headers_build_dir,
            ],
            verbose: true
        )
    endif

    #### Example Programs ####
    ##########################
    if build_examples_opt
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

                # Force C++ compiler linkage (no RDC so fine)
                link_language: 'cpp',
            )
        endif # Has CUDA
    endif # build_examples_opt
endif # build_core_opt
