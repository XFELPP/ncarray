installed_host_headers = [
    'allocator.hh',
    'casts.hh',
    'elementwise.hh',
    'reductions.hh',
]

if not nca_as_wheel_opt and build_core_opt
    install_headers(
        installed_host_headers,
        install_dir: header_install_dir / 'ncarray/host',
    )
else
    py3.install_sources(
        installed_host_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray/host',
    )
endif # not nca_as_wheel_opt and build_core_opt
