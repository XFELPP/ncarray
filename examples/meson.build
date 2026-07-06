#### Example Programs ####
##########################
if build_core_opt and build_examples_opt
    deps_for_examples = [asmjit_dep, ncarray_dep, ncarrayjit_dep]

    pixel_binning_host = executable(
        'benchmark_pixel_binning_host',
        [
            'benchmark_binning.cc',
        ],
        dependencies: deps_for_examples,
        install: true,
        include_directories: ncarray_headers,
    )

    if nca_has_cuda
        deps_for_examples += cuda_dep
        deps_for_examples += ncdevarray_dep

        pixel_binning_device = executable(
            'benchmark_pixel_binning',
            [
                'benchmark_binning.cu',
            ],
            dependencies: deps_for_examples,
            install: true,
            include_directories: ncarray_headers,
            cuda_args: nca_cuda_args,

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        )
    endif # nca_has_cuda
endif # build_core_opt and build_examples_opt
