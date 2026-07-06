if build_core_opt
    deps_for_ncarrayjit = []
    if not nca_has_cuda
        deps_for_ncarrayjit = deps_for_ncarray + [asmjit_dep]
        libncarrayjit = shared_library(
            'ncarrayjit',
            [
                # Host
                'host/a64.cc',
                'host/rtcompiler.cc',
                'host/x86.cc',
                # Common
                'jit_utils.cc',
                'path_utils.cc',
            ],
            cpp_args: ['-DNCA_BUILD_JIT_API'],
            dependencies: deps_for_ncarrayjit,
            include_directories: ncarray_headers,
            install: true,
            install_dir: libncarray_install_dir,
            install_rpath: lib_install_rpath,
        )
    else
        deps_for_ncarrayjit = deps_for_ncarray + [asmjit_dep, cuda_dep, dl_dep] + dbghelp_dep
        libncarrayjit = shared_library(
            'ncarrayjit',
            [
                # Host
                'host/a64.cc',
                'host/rtcompiler.cc',
                'host/x86.cc',
                # Common
                'jit_utils.cc',
                'path_utils.cc',
                # Device
                'device/rtcompiler.cc',
            ],
            cpp_args: ['-DNCA_BUILD_JIT_API'],
            cuda_args: nca_cuda_args,
            # dbghelp_dep is for Windows - otherwise, its an empty list
            dependencies: deps_for_ncarrayjit,
            include_directories: ncarray_headers,
            install: true,
            install_dir: libncarray_install_dir,
            install_rpath: lib_install_rpath,
            version: '1',
            soversion: '1',
        )
    endif # nca_has_cuda

    ncarrayjit_dep = declare_dependency(
        link_with: [
            libncarrayjit,
        ],
        dependencies: deps_for_ncarrayjit,
        include_directories: ncarray_headers,
    )
else
    libncarrayjit = cpp.find_library('ncarrayjit', required: true)
    ncarrayjit_dep = declare_dependency(
        dependencies: deps_for_ncarray + [
            libncarrayjit,
        ],
        include_directories: ncarray_headers,
    )
endif # build_core_opt

installed_common_jit_headers = [
    'jit_utils.hh',
    'path_utils.hh',
]

## NOTE: The only headers that are selective installed are the bundled CCCL headers
##       This is only because the headers cannot be found if the toolkit isn't available
if nca_has_cuda
    #### Install CCCL Headers for JIT ####
    # Will run a little Python script to copy the current CUDA toolkit CCCL headers
    # Afterwards, these will be installed into the wheel for use by the JIT compiler
    jit_headers_build_dir = meson.current_build_dir() / 'jit_tmp_headers'

    copy_script = meson.source_root() + '/utilities/copy_cccl_headers.py'

    run_command(
        py3, copy_script, jit_headers_build_dir,
        check: true,
    )
endif # nca_has_cuda

if not nca_as_wheel_opt and build_core_opt
    install_headers(
        installed_common_jit_headers,
        install_dir: header_install_dir / 'ncarray/jit',
    )

    if nca_has_cuda
        # Install CCCL headers into the prefix headers path
        install_subdir(
            jit_headers_build_dir / 'cuda',
            install_dir: header_install_dir / 'ncarray/jit',
        )
        install_subdir(
            jit_headers_build_dir / 'nv',
            install_dir: header_install_dir / 'ncarray/jit',
        )
    endif # nca_has_cuda
else
    py3.install_sources(
        installed_common_jit_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray/jit',
    )

    if nca_has_cuda
        # Install CCCL headers into the wheel headers path
        install_subdir(
            jit_headers_build_dir / 'cuda',
            install_dir: py3.get_install_dir() / 'ncarray/include/ncarray/jit',
        )
        install_subdir(
            jit_headers_build_dir / 'nv',
            install_dir: py3.get_install_dir() / 'ncarray/include/ncarray/jit',
        )
    endif # nca_has_cuda
endif # not nca_as_wheel_opt

# Install remaining headers from sub-directories
subdir('device')
subdir('host')
