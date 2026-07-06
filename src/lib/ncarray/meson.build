# Build the JIT compiler library first so the dependency is available for the core
# This is also guarded behind the build_core_opt in its build file.
subdir('jit')

if build_core_opt
    deps_for_ncarray += [asmjit_dep, ncarrayjit_dep]

    make_tus_script = meson.source_root() + '/utilities/make_tus.py'

    host_src_dir = meson.current_source_dir() / 'tus/host'

    # Create the translation units if they don't already exist.
    # If running manually and need to regenerate them, run the script by hand
    check_host_tus = run_command(
        py3, '-c', 'import glob, sys; sys.exit(0 if glob.glob("' + host_src_dir + '/*.cc") else 1)',
        check: false,
    )
    if check_host_tus.returncode() != 0
        run_command(
            py3, make_tus_script, '--output', host_src_dir,
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
            'host/casts.cc',
            'ncarrays.cc',
            'ncarrayrefs.cc',
            'ncarrayviews.cc',
            'soarrays.cc',
            'soarrayrefs.cc',
            'soarrayviews.cc',
        ] + host_cc,
        cpp_args: ['-DNCA_BUILD_NCARRAY_API'],
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
        dependencies: deps_for_ncarray,
        include_directories: ncarray_headers,
    )

    if nca_has_cuda
        dev_src_dir = meson.current_source_dir() / 'tus/device'

        # Create the translation units if they don't already exist.
        # If running manually and need to regenerate them, run the script by hand
        check_device_tus = run_command(
            py3, '-c', 'import glob, sys; sys.exit(0 if glob.glob("' + dev_src_dir + '/*.cu") else 1)',
            check: false,
        )
        if check_device_tus.returncode() != 0
            run_command(
                py3, make_tus_script, '--output', dev_src_dir, '--device',
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
                'host/casts.cc',
                'ncdevarrayviews.cu',
                'ncdevarrayrefs.cu',
                'ncdevarrays.cu',
                'sodevarrayviews.cu',
                'sodevarrayrefs.cu',
                'sodevarrays.cu',
            ] + dev_cu,
            cpp_args: ['-DNCA_BUILD_NCARRAY_API'],
            cuda_args: nca_cuda_args + ['-DNCA_BUILD_NCARRAY_API'],
            include_directories: ncarray_headers,
            dependencies: [asmjit_dep, ncarrayjit_dep, ncarray_dep],
            install: true,
            install_dir: libncarray_install_dir,
            install_rpath: lib_install_rpath,
            version: '1',
            soversion: '1',

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        )

        ncdevarray_dep = declare_dependency(
            link_with: [
                libncdevarray,
            ],
            dependencies: deps_for_ncarray + [cuda_dep],
            include_directories: ncarray_headers,
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

        ncdevarray_dep = declare_dependency(
            dependencies: deps_for_ncarray + [
                ncarray_dep,
                libncdevarray,
            ],
            include_directories: ncarray_headers,
        )
    endif
endif # end build_core_opt

installed_headers = [
    'array_element_proxy.hh',
    'array_impl.hh',
    'array_operations.hh',
    'array_traits.hh',
    'build_macro.hh',
    'custom_types.hh',
    'dtype.hh',
    'engines.hh',
    'indexing.hh',
    'layout.hh',
    'ncarrays.hh',
    'op_code.hh',
    'op_traits.hh',
    'reductions.hh',
    'soarrays.hh',
    'storage.hh',
]

# The following headers are for integration with device code on host side
installed_device_compat_headers = [
    'ncdevarrays.cuh',
    'sodevarrays.cuh',
]

if not nca_as_wheel_opt and build_core_opt
    install_headers(
        installed_headers,
        install_dir: header_install_dir / 'ncarray',
    )
    install_headers(
        installed_device_compat_headers,
        install_dir: header_install_dir / 'ncarray',
    )
else
    py3.install_sources(
        installed_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray',
    )
    py3.install_sources(
        installed_device_compat_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray',
    )
endif # not nca_as_wheel_opt and build_core_opt


# Collect and install remaining headers
subdir('device')
subdir('engines')
subdir('expression')
subdir('host')
# subdir('jit') was done at the beginning to create the JIT shared lib
