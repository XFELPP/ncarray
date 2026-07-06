installed_host_jit_headers = [
    'a64.hh',
    'rtcompiler.hh',
    'x86.hh',
]

if not nca_as_wheel_opt and build_core_opt
    install_headers(
        installed_host_jit_headers,
        install_dir: header_install_dir / 'ncarray/jit/host',
    )
else
    py3.install_sources(
        installed_host_jit_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray/jit/host',
    )
endif # not nca_as_wheel_opt and build_core_opt
