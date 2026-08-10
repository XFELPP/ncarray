project(
    'ncarray',
    'cpp',
    version: '0.7.5',
    meson_version: '>=1.10.0',
    default_options: [
        'cpp_std=c++23,c++latest',
        'cuda_std=c++20',
        'optimization=3',
        'pkgconfig.relocatable=true',
    ],
)
add_languages('cpp', native: true, required: true)
nca_has_cuda = add_languages('cuda', required: false)
if nca_has_cuda
    # Source code guards CUDA specific features behind NCA_HAS_CUDA
    add_project_arguments('-DNCA_HAS_CUDA', language: ['cpp', 'cuda'])

    cuda = meson.get_compiler('cuda')
    cuda_version = cuda.version()
endif

cpp = meson.get_compiler('cpp')

# Check compiler versions (for C++23 suppport especially)
if cpp.get_id() == 'gcc'
    gcc_version = cpp.version()
    if not gcc_version.version_compare('>=13')
        if gcc_version.version_compare('>=12')
            warning('GCC >=13 is recommended, however, GCC 12+ should work.')
        else
            error('Builds with GCC require version >=12')
        endif
    endif

    if nca_has_cuda
        # Too new a compiler for CUDA support also causes problems
        if cuda_version.version_compare('<12.4') and not gcc_version.version_compare('<13')
            error('For CUDA < 12.4 builds, must use GCC < 13')
        elif cuda_version.version_compare('<12.8') and not gcc_version.version_compare('<14')
            error('For CUDA < 12.8 builds, must use GCC < 14')
        elif not gcc_version.version_compare('<15')
            error('For CUDA 12.8+ builds, must use GCC < 15')
        endif
    endif
elif cpp.get_id() == 'clang'
    clang_version = cpp.version()
    if not clang_version.version_compare('>=16')
        error('Builds with clang require version >=16')
    endif

    if nca_has_cuda
        # Too new a compiler for CUDA support also causes problems
        if cuda_version.version_compare('<12.4') and not clang_version.version_compare('<17')
            error('For CUDA < 12.4 builds, must use clang < 17')
        elif cuda_version.version_compare('<12.8') and not clang_version.version_compare('<19')
            error('For CUDA < 12.8 builds, must use clang < 19')
        elif not clang_version.version_compare('<20')
            error('For CUDA 12.8+ builds, must use clang < 20')
        endif
    endif
elif cpp.get_id() == 'msvc'
    msvc_version = cpp.version()
    if not msvc_version.version_compare('>=19.30')
        error('Builds with msvc require version >=19.30')
    endif

    if nca_has_cuda
        # Too new a compiler for CUDA support also causes problems
        if cuda_version.version_compare('<12.4') and not msvc_version.version_compare('<19.38')
            error('For CUDA < 12.4 builds, must use msvc < 19.38 (VS 2022 < 17.8)')
        elif cuda_version.version_compare('<12.8') and not msvc_version.version_compare('<19.41')
            error('For CUDA < 12.8 builds, must use msvc < 19.41 (VS 2022 < 17.11)')
        elif not msvc_version.version_compare('<19.50')
            error('For CUDA 12.8+ builds, must use msvc < 19.50 (VS 2022 < 18.0)')
        endif
    endif
endif

# For Windows adjust for limit of 65,536 COFF sections
if cpp.get_id() == 'msvc'
    add_project_arguments('/bigobj', '/DNOMINMAX', '/Zc:preprocessor', language: 'cpp')
    if nca_has_cuda
        add_project_arguments(
            '-Xcompiler=/bigobj',
            '-Xcompiler=/DNOMINMAX',
            '-Xcompiler=/Zc:preprocessor',
            '-D_SOURCE_LOCATION_',
            language: 'cuda',
        )
    endif
endif

#### Fetch Build Options ####
#############################
# Core options
build_core_opt = get_option('build_core')
build_python_opt = get_option('build_python')
nca_cuda_archs_opt = get_option('nca_cuda_archs')
nca_as_wheel_opt = get_option('nca_as_wheel')

# Add-ons and development
build_examples_opt = get_option('build_examples')

# Test binaries (gtest)
build_tests_opt = get_option('build_tests')

#### Parse CUDA architectures and set CUDA args ####
####################################################

# NOTE: I think there may be a clang-frontend for CUDA... These may not all work if
# using that instead of nvcc specifically!
nca_cuda_args = [
    # On Windows, inspite of overrides and global settings, C++14 gets chosen
    # as the CUDA standard unless this gets added here
    '--std=c++20',
    '--extended-lambda',
    '--expt-relaxed-constexpr',
    # silence warning: declared but not referenced (e.g. pybind, stl...)
    '-Xcudafe', '--diag_suppress=445',
    # silence warning: long double treated as double
    '-Xcudafe', '--diag_suppress=20208',
    # silence warning: float overflow in CUDA cmath header (Windows)
    # TODO: Try to limit this to includes/callsites eventually
    '-Xcudafe', '--diag_suppress=221',
    # silence warning: field of class type without a DLL interface (Windows)
    # TODO: Try to limit this to includes/callsites eventually (tried a bit already)
    '-Xcudafe', '--diag_suppress=1394',
    # silence warning: __host__ called from __host__ __device__ (Linux - use of dispatch in GPUEngine)
    # TODO: Try to limit this to includes/callsites eventually (tried a bit already)
    '-Xcudafe', '--diag_suppress=20011',
    # silence warning: unrecognized GCC pragma (Linux)
    '-Xcudafe', '--diag_suppress=1675',
]
if nca_has_cuda
    if cuda_version.version_compare('>=12.8')
        nca_cuda_args += [
            '-static-global-template-stub=false',
            '-device-entity-has-hidden-visibility=false',
        ]
    endif
endif

nca_cuda_args += ['-Xfatbin', '-compress-all']

if nca_cuda_archs_opt == 'all'
    # Setup a fat binary for Turing, Ampere, Ada and Hopper
    nca_cuda_args += [
        '-gencode', 'arch=compute_75,code=sm_75',
        '-gencode', 'arch=compute_80,code=sm_80',
        '-gencode', 'arch=compute_86,code=sm_86',
        '-gencode', 'arch=compute_89,code=sm_89',
        '-gencode', 'arch=compute_90,code=[sm_90,compute_90]',
    ]
else
    # Parse comma-separated architectures ('80', or '75,90')
    # Last value will generate [sm_ARCH,compute_ARCH]
    selected_cuda_archs = nca_cuda_archs_opt.split(',')

    num_archs = selected_cuda_archs.length()
    last_arch = selected_cuda_archs[num_archs - 1].strip()
    foreach arch : selected_cuda_archs
        arch = arch.strip()
        if arch != ''
            if arch == last_arch
                # Generate PTX in this case, for forward-compatibility
                nca_cuda_args += [
                    '-gencode', 'arch=compute_' + arch + ',code=[sm_' + arch + ',compute_' + arch + ']'
                ]
            else
                nca_cuda_args += [
                    '-gencode', 'arch=compute_' + arch + ',code=sm_' + arch
                ]
            endif
        endif
    endforeach
endif

#### Core Dependencies ####
###########################

openmp_dep = dependency('openmp', required: false)
nca_has_openmp = openmp_dep.found()
if nca_has_openmp
    add_project_arguments('-DNCA_HAS_OPENMP', language: ['cpp', 'cuda'])
endif

deps_for_ncarray = []

# Add libdl as needed for JIT if not merged directly into the cstdlib
dl_dep = cpp.find_library('dl', required: false)
if dl_dep.found()
    deps_for_ncarray += dl_dep
endif

if nca_has_cuda
    # CUDA driver and nvrtc are both for JIT
    cuda_dep = dependency('cuda', modules: ['cuda', 'cudart', 'nvrtc'])

    # On Windows need symbol demangling for the JIT compiler.
    dbghelp_dep = []
    if cpp.get_id() == 'msvc'
        dbghelp_dep = [cpp.find_library('dbghelp', required: true)]
    endif
endif

if nca_has_openmp
    deps_for_ncarray += openmp_dep
endif

if build_python_opt
    pybind11_dep = dependency('pybind11', required: true)
endif

if build_core_opt
    cmake = import('cmake')
    asmjit_opts = cmake.subproject_options()
    asmjit_opts.add_cmake_defines({
        'ASMJIT_STATIC': true,
        'ASMJIT_NO_INSTALL': true,
        'CMAKE_POSITION_INDEPENDENT_CODE': true,
    })
    asmjit_proj = cmake.subproject('asmjit', options: asmjit_opts)
    asmjit_dep = asmjit_proj.dependency('asmjit')
else
    asmjit_dep = declare_dependency()
endif

#### Main Python Setup ####
###########################

pymod = import('python')

py3 = pymod.find_installation(pure: false)
py3_dep = py3.dependency()

min_python_ver = '3.8'
py_ver = py3.language_version()
if py_ver.version_compare(f'<@min_python_ver@')
    error(f'Minimum supported Python version is @min_python_ver@, found @py_ver@')
endif

#### Path Setup ####
####################

py_install_dir = py3.get_install_dir(pure: false)

pyncarray_install_dir = py_install_dir / 'ncarray/core'

if nca_as_wheel_opt
    # Install C++ libraries inside the wheel structure
    # Package and library setup will be:
    # site-packages/
    #    | - ncarray
    #           | - lib
    #           |    | - libncarray.so
    #           |
    #           | - core
    #           |    | - _pyncarray.so
    #           |
    #           | - __init__.py (utility functions and expose core lib)

    libncarray_install_dir = py_install_dir / 'ncarray/lib'
    header_install_dir = py_install_dir / 'ncarray/include'

    # Relative path from 'ncarray/core' to 'ncarray/lib'
    py_to_lib_path = '../lib'

    exe_prefix = py_install_dir
    lib_prefix = pyncarray_install_dir
else
    # Normal install path
    libncarray_install_dir = get_option('libdir')
    header_install_dir = get_option('includedir')

    # Relative path from '$PREFIX/lib/pythonMAJ.MIN/site-packages/ncarray/core' to '$PREFIX/lib'
    py_to_lib_path = '../../../../'

    exe_prefix = get_option('prefix') / 'bin'
    lib_prefix = pyncarray_install_dir
endif

# For both the wheel and non-wheel case, executables relative path, for
# $PREFIX/bin to $PREFIX/lib is
exe_to_libncarray = '..' / libncarray_install_dir

lib_install_rpath = '$ORIGIN'

abs_exe_rpath = exe_prefix / exe_to_libncarray
abs_lib_rpath = lib_prefix / py_to_lib_path

# Setup RPATH for various platforms, for any executables that may need it
if host_machine.system() == 'darwin'
    exe_install_rpath = '@executable_path' + ':' + ('@executable_path' / exe_to_libncarray) + ':' + abs_exe_rpath
    py_to_lib_install_rpath = '@loader_path' + ':' + ('@loader_path' / py_to_lib_path)  + ':' + abs_lib_rpath
else
    exe_install_rpath = '$ORIGIN' + ':' + ('$ORIGIN' / exe_to_libncarray)  + ':' + abs_exe_rpath
    py_to_lib_install_rpath = '$ORIGIN' + ':' + ('$ORIGIN' / py_to_lib_path) + ':' + abs_lib_rpath
endif

#### ncarray ####
#################
## Array definitions

ncarray_headers = include_directories('src/lib', 'src/lib/ncarray')

# Traverse the main source first so the dependency objects are available
# for examples and tests
subdir('src')

# Build example programs
subdir('examples')

# Build the test suite
subdir('tests')

# Setup a pkgconfig file now after the build completed.
pkg = import('pkgconfig')

if build_core_opt and not nca_as_wheel_opt
    pc_install_dir = get_option('libdir') / 'pkgconfig'
else
    pc_install_dir = py_install_dir / 'ncarray/pkgconfig'
endif

pc_libs = [libncarray, libncarrayjit]
if nca_has_cuda
    pc_libs += [libncdevarray]
endif

feature_flags = []
if nca_has_cuda
    feature_flags += ['-DNCA_HAS_CUDA']
endif

pc_headers = ['.', 'ncarray']
pkg.generate(
    name: 'ncarray',
    description: 'Non-contiguous multi-dimensional arrays.',
    libraries: pc_libs,
    extra_cflags: feature_flags,
    subdirs: pc_headers,
    version: meson.project_version(),
    url: 'https://ncarray.xfelpp.org',
    license: 'MPL-2.0',
    install_dir: pc_install_dir,
)
