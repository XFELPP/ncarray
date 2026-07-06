# The following headers are device only as well as host-called kernels
installed_device_headers = [
    'atomic.cuh',
    'casts.cuh',
    'elementwise.cuh',
    'kernels.cuh',
    'mem_pool.cuh',
    'reductions.cuh',
    'utilities.cuh',
    'warp.cuh',
]

if not nca_as_wheel_opt and build_core_opt
    install_headers(
        installed_device_headers,
        install_dir: header_install_dir / 'ncarray/device',
    )
else
    py3.install_sources(
        installed_device_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray/device',
    )
endif # not nca_as_wheel_opt and build_core_opt
