installed_expression_headers = [
    'dynamicmvnode.hh',
    'interface.hh',
    'mvnode.hh',
    'staticmvnode.hh',
]

# Currently, Stencil support is for GPU only
installed_expression_stencil_headers = [
    'stencil.hh',
]

if not nca_as_wheel_opt and build_core_opt
    install_headers(
        installed_expression_headers,
        install_dir: header_install_dir / 'ncarray/expression',
    )

    install_headers(
        installed_expression_stencil_headers,
        install_dir: header_install_dir / 'ncarray/expression',
    )
else
    py3.install_sources(
        installed_expression_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray/expression',
    )

    py3.install_sources(
        installed_expression_stencil_headers,
        pure: false,
        subdir: 'ncarray/include/ncarray/expression',
    )
endif # not nca_as_wheel_opt and build_core_opt
