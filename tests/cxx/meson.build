#### Testing ####
#################
if build_core_opt and build_tests_opt
    gtest_dep = dependency('gtest', main: true, static: true, required: false)

    if not gtest_dep.found()
        gtest_proj = subproject('gtest', default_options: ['default_library=static'])
        gtest_dep = gtest_proj.get_variable('gtest_main_dep')
    endif

    deps_for_tests = [asmjit_dep, gtest_dep, ncarray_dep, ncarrayjit_dep]
    if nca_has_cuda
        deps_for_tests += cuda_dep
        deps_for_tests += ncdevarray_dep
    endif

    test_index = executable(
        'test_array_indexing',
        [
            'test_indexing.cc',
        ],
        dependencies: deps_for_tests,
        install: false,
        include_directories: ncarray_headers,
    )
    test('Testing array indexing.', test_index, verbose: true)

    test_binary_ops = executable(
        'test_array_binary_operations',
        [
            'test_binary_operations.cc',
        ],
        dependencies: deps_for_tests,
        install: false,
        include_directories: ncarray_headers,
    )
    test('Testing array binary operations.', test_binary_ops, verbose: true)

    test_unary_ops = executable(
        'test_array_unary_operations',
        [
            'test_unary_operations.cc',
        ],
        dependencies: deps_for_tests,
        install: false,
        include_directories: ncarray_headers,
    )
    test('Testing array unary operations.', test_unary_ops, verbose: true)

    test_reduction_ops = executable(
        'test_array_reductions',
        [
            'test_reductions.cc',
        ],
        dependencies: deps_for_tests,
        install: false,
        include_directories: ncarray_headers,
    )
    test('Testing array reductions.', test_reduction_ops, verbose: true)

    test_float_vecs = executable(
        'test_float_vecs',
        [
            'test_float_vecs.cc',
        ],
        dependencies: deps_for_tests,
        install: false,
        include_directories: ncarray_headers,
    )
    test('Testing custom float/double vectors.', test_float_vecs, verbose: true)

    if nca_has_cuda
        test_gpu_copies = executable(
            'test_gpu_copies',
            [
                'test_gpu_copies.cu',
            ],
            dependencies: deps_for_tests,
            install: false,
            include_directories: ncarray_headers,
            cuda_args: nca_cuda_args,

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        )
        test(
            'Testing GPU array copies.',
            test_gpu_copies,
            env: [
                'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
                'NCARRAY_JIT_INCLUDE_DIR=' + jit_headers_build_dir,
            ],
            verbose: true
        )

        test_gpu_binary_ops = executable(
            'test_gpu_array_binary_operations',
            [
                'test_gpu_binary_operations.cu',
            ],
            dependencies: deps_for_tests,
            install: false,
            include_directories: ncarray_headers,
            cuda_args: nca_cuda_args,

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        )
        test(
            'Testing GPU array binary operations.',
            test_gpu_binary_ops,
            env: [
                'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
                'NCARRAY_JIT_INCLUDE_DIR=' + jit_headers_build_dir,
            ],
            verbose: true
        )

        test_gpu_unary_ops = executable(
            'test_gpu_array_unary_operations',
            [
                'test_gpu_unary_operations.cu',
            ],
            dependencies: deps_for_tests,
            install: false,
            include_directories: ncarray_headers,
            cuda_args: nca_cuda_args,

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        )
        test(
            'Testing GPU array unary operations.',
            test_gpu_unary_ops,
            env: [
                'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
                'NCARRAY_JIT_INCLUDE_DIR=' + jit_headers_build_dir,
            ],
            verbose: true
        )

        test_gpu_reductions = executable(
            'test_gpu_array_reductions',
            [
                'test_gpu_reductions.cu',
            ],
            dependencies: deps_for_tests,
            install: false,
            include_directories: ncarray_headers,
            cuda_args: nca_cuda_args,

            # Force C++ compiler linkage (no RDC so fine)
            link_language: 'cpp',
            override_options: ['cpp_std=c++20', 'cuda_std=c++20'],
        )
        test(
            'Testing GPU array reductions.',
            test_gpu_reductions,
            env: [
                'NCARRAY_INCLUDE_DIR='+meson.project_source_root() / 'src/lib',
                'NCARRAY_JIT_INCLUDE_DIR=' + jit_headers_build_dir,
            ],
            verbose: true
        )
    endif
endif # build_core_opt and build_tests_opt
