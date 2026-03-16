project(
    'xtcpp',
    'cpp',
    version: '0.1.0',
    default_options: [
        'cpp_std=c++23',
        'cuda_std=c++20',
        'optimization=3',
        # Statically link subproject libraries by default as there are
        # difficulties getting the wheels to work with shared libraries
        'default_library=static',
    ],
)
add_languages('cuda', required : true)

add_project_arguments('-frtti', language: 'cpp')
add_project_arguments('-fopenmp', language: 'cpp')
add_project_arguments('-march=native', language: 'cpp')

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

pybind11_dep = dependency('pybind11', required: true)

#### Path Setup ####
####################

py_install_dir = py3.get_install_dir()

# Currently there are no sub-directories, so the root install dir is just
# where ncarray is installed
libncarray_install_dir = py_install_dir

# Executables get installed in $prefix/bin while the libs are in $prefix/lib/...
# So the relative path is:
exe_to_libncarray = '..' / libncarray_install_dir

# Setup RPATH for various platforms, for any executables that may need it
if host_machine.system() == 'darwin'
    exe_install_rpath = '@executable_path' + ':' + ('@executable_path' / exe_to_libncarray)
else
    exe_install_rpath = '$ORIGIN' + ':' + ('$ORIGIN' / exe_to_libncarray)
endif


lib_install_rpath = '$ORIGIN'


#### Other Dependencies ####
############################
## spdlog
# spdlog_proj = subproject('spdlog')
# spdlog_dep = spdlog_proj.get_variable('spdlog_dep')

## OpenMP
# openmp_dep = dependency('openmp', required: true)

#### ncarray ####
#################
## Array definitions

ncarray_headers = include_directories('src', 'src/ncarray')

libncarray = shared_library(
    'ncarray',
    [
        'src/ncarray/ncarrayview.cc',
        'src/ncarray/ncarray.cc',
        'src/ncarray/ncarrayref.cc',
    ],
    include_directories: ncarray_headers,
    install: true,
    install_dir: py3.get_install_dir(),
    install_rpath: lib_install_rpath,
    version: '1',
    soversion: '1',
)

ncarray_dep = declare_dependency(
    link_with: [
        libncarray,
    ],
    include_directories: ncarray_headers,
)

py_mod = py3.extension_module(
    'ncarray',
    [
        'src/python/ncarray.cc'
    ],
    install: true,
    install_dir: py3.get_install_dir(),
    include_directories: ncarray_headers,
    dependencies : [
        ncarray_dep,
        pybind11_dep,
    ],
    # No sub-directory required - the entire project compiles to a single extension
    # `ncarray` which will be installed in site-packages
    # subdir: '',
    install_rpath: lib_install_rpath,
)

#### CuFile Python module ####
##############################

#py_mod_gpu = py3.extension_module(
#    '_xtcppgpu',
#    [
#        'src/python/pyxtcppgpu.cc'
#    ],
#    install: true,
#    install_dir: py3.get_install_dir(),
#    dependencies : [
#        pybind11_dep,
#        xtcpp_base_dep,
#        xtcpp_cufile_dep,
#        xtcpp_hdf5_dep,
#        spdlog_dep,
#        cufile_dep,
#        #cudart_lib,
#    ],
#    install_rpath: '$ORIGIN',
#    link_language: 'cpp',
#)
