installed_gpu_engine_headers = [
    'gpuengine.hh',
]

installed_host_engine_headers = [
    'hostengine.hh',
]

if not nca_as_wheel_opt and build_core_opt
    install_headers(
        installed_host_engine_headers,
        install_dir: header_install_dir / 'ncarray/engines',
    )

    # Device implementation headers
    install_headers(
        installed_gpu_engine_headers,
        install_dir: header_install_dir / 'ncarray/engines',
    )

else
    py3.install_sources(
        installed_gpu_engine_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray/engines',
    )
    py3.install_sources(
        installed_host_engine_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray/engines',
    )
endif # not nca_as_wheel_opt and build_core_opt
