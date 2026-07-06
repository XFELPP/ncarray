installed_device_jit_headers = [
    'extensions.hh',
    'rtcompiler.hh',
]

if not nca_as_wheel_opt and build_core_opt
    install_headers(
        installed_device_jit_headers,
        install_dir: header_install_dir / 'ncarray/device',
    )
else
    py3.install_sources(
        installed_device_jit_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray/jit/device',
    )
endif # not nca_as_wheel_opt and build_core_opt
