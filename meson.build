project(
    'ncarray',
    'cpp',
    version: '0.1.0',
    default_options: [
        'cpp_std=c++23',
        'optimization=3',
    ],
)

add_project_arguments('-frtti', language: 'cpp')

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

ncarray_lib_install_dir = py3.get_install_dir() / 'ncarray/lib'

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
pyncarray_install_dir = py_install_dir / 'ncarray/core'
py_to_lib_path = '../lib'

# Executables get installed in $prefix/bin while the libs are in $prefix/lib/...
# So the relative path is:
exe_to_libncarray = '..' / libncarray_install_dir

lib_install_rpath = '$ORIGIN'

abs_exe_rpath = py_install_dir / exe_to_libncarray
abs_lib_rpath = py_install_dir / py_to_lib_path

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

libncarray = shared_library(
    'ncarray',
    [
        'src/lib/ncarray/ncarrayview.cc',
        'src/lib/ncarray/ncarray.cc',
        'src/lib/ncarray/ncarrayref.cc',
    ],
    include_directories: ncarray_headers,
    install: true,
    install_dir: libncarray_install_dir,
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

# Install headers into the package for easy bundling
installed_headers = [
    'src/lib/ncarray/array_operations.hh',
    'src/lib/ncarray/array_traits.hh',
    'src/lib/ncarray/dtype.hh',
    'src/lib/ncarray/indexing.hh',
    'src/lib/ncarray/ncarray.hh',
    'src/lib/ncarray/ncarrayref.hh',
    'src/lib/ncarray/ncarrayview.hh',
]
py3.install_sources(
    installed_headers,
    pure: false,
    subdir: 'ncarray/include',
)

py_mod = py3.extension_module(
    '_pyncarray',
    [
        'src/lib/python/ncarray.cc'
    ],
    install: true,
    install_dir: pyncarray_install_dir,
    include_directories: ncarray_headers,
    dependencies : [
        ncarray_dep,
        pybind11_dep,
    ],
    install_rpath: py_to_lib_install_rpath,
    build_rpath: py_to_lib_install_rpath,
)

py_sources = [
    'src/ncarray/__init__.py',
]

py3.install_sources(
    py_sources,
    pure: false,
    subdir: 'ncarray',
)

placeholders = [
    'src/ncarray/core/__init__.py',
]

py3.install_sources(
    placeholders,
    pure: false,
    subdir: 'ncarray/core',
)
