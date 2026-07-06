if build_python_opt
    # Figure out dependencies
    deps_for_py_mod = [asmjit_dep, ncarray_dep, ncarrayjit_dep, pybind11_dep, py3_dep]

    if nca_has_cuda
        deps_for_py_mod += cuda_dep
        deps_for_py_mod += ncdevarray_dep
    endif

    py_bindings_headers = [
        'binding_builder.hh',
        'utilities.hh',
    ]

    if nca_as_wheel_opt
        # NOTE: Only installing the bindings specific headers if in a wheel.
        # The include path would be confusing in a global install directory.
        py3.install_sources(
            py_bindings_headers,
            pure: false,
            subdir: 'ncarray/include/python',
        )
    endif

    # NOTE: The way the CUDA dependencies are linked would cause issues
    # opening host-only extension modules if the CUDA driver is not available. Using
    # DELAYLOAD will avoid the issue on Windows. On Linux, we provide a stub below.
    py_mod_link_args = []
    if nca_has_cuda and cpp.get_id() == 'msvc'
        if cuda_version.version_compare('>=13.0')
            py_mod_link_args += [
                '/DELAYLOAD:ncdevarray-1.dll',
                '/DELAYLOAD:ncarrayjit-1.dll',
                '/DELAYLOAD:cudart64_13.dll',
                '/DELAYLOAD:nvrtc64_130_0.dll',
                'delayimp.lib',
            ]
        else
            py_mod_link_args += [
                '/DELAYLOAD:ncdevarray-1.dll',
                '/DELAYLOAD:ncarrayjit-1.dll',
                '/DELAYLOAD:cudart64_12.dll',
                '/DELAYLOAD:nvrtc64_12.dll',
                'delayimp.lib',
            ]
        endif # cuda 13 casing
    endif

    libeagereval = shared_library(
        'eagereval',
        [
            'eager.cc',
        ],
        install: true,
        install_dir: pyncarray_install_dir,
        include_directories: ncarray_headers,
        cpp_args: ['-DNCA_BUILD_PYNCARRAY_API'],
        dependencies : deps_for_py_mod,
        install_rpath: py_to_lib_install_rpath,
        build_rpath: py_to_lib_install_rpath,
        link_args: py_mod_link_args,
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
            'ncarray.cc',
        ],
        install: true,
        install_dir: pyncarray_install_dir,
        include_directories: ncarray_headers,
        dependencies : deps_for_py_mod + eager_dep,
        install_rpath: py_to_lib_install_rpath,
        build_rpath: py_to_lib_install_rpath,
        link_args: py_mod_link_args,
    )

    if nca_has_cuda
        # On Windows, include paths are getting oddly mangled/quoted with nvcc...
        # Add them back in explicitly here. Wasn't able to get it working more "properly"
        if cpp.get_id() == 'msvc'
            # Python headers
            py3_inc_dir = run_command(
                py3, '-c', 'import sysconfig; print(sysconfig.get_path("include"))',
                check: true,
            ).stdout().strip()

            # pybind11 headers
            pybind11_inc_dir = run_command(
                py3, '-c', 'import pybind11; print(pybind11.get_include())',
                check: true,
            ).stdout().strip()

            py_ext_cuda_args = [
                '-I' + py3_inc_dir,
                '-I' + pybind11_inc_dir,
            ]
        else
            # On Linux this isn't an issue.
            py_ext_cuda_args = []
        endif

        # NOTE: For faster/more efficient compilation, each GPU specialization is
        # split off into its own translation unit.
        py_dev_mod = py3.extension_module(
            '_pyncdevarray',
            [
                'ncdevarray.cu',
                'ncdevarray_view.cu',
                'ncdevarray_ref.cu',
                'ncdevarray_owner.cu',
                'sodevarray_view.cu',
                'sodevarray_ref.cu',
                'sodevarray_owner.cu',
            ],
            install: true,
            install_dir: pyncarray_install_dir,
            include_directories: ncarray_headers,
            dependencies : deps_for_py_mod + eager_dep,
            install_rpath: py_to_lib_install_rpath,
            build_rpath: py_to_lib_install_rpath,
            cuda_args: nca_cuda_args + py_ext_cuda_args,

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        )

        # Provide the stub for Linux
        if host_machine.system() == 'linux'
            fs = import('fs')
            nvcc_bin = find_program('nvcc')
            nvcc_path = nvcc_bin.full_path()

            cuda_root = fs.parent(fs.parent(nvcc_path))

            # Check `lib64/stubs` and `lib/stubs` for driver stub
            cuda_stub_lib = cuda_root / 'lib64/stubs/libcuda.so'
            if not fs.exists(cuda_stub_lib)
                cuda_stub_lib = cuda_root / 'lib/stubs/libcuda.so'
            endif

            # For conda environments it's generally under targets
            if not fs.exists(cuda_stub_lib)
                cuda_stub_lib = cuda_root / 'targets/x86_64-linux/lib/stubs/libcuda.so'
            endif

            if fs.exists(cuda_stub_lib)
                message('**** Installing libcuda stub library: ' + cuda_stub_lib + ' ****')

                install_data(
                    cuda_stub_lib,
                    install_dir: py3.get_install_dir() / 'ncarray/core/stubs'
                )
            else
                # The stub must be provided to prevent issues for CUDA-enabled builds
                # for users running without the CUDA driver (CPU-only mode).
                error('Cannot find CUDA driver stub (libcuda.so) from CUDA root: ' + cuda_root)
            endif
        endif # Linux guard
    endif
endif # build_python_opt
