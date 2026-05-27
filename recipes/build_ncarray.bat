@echo off

set "CUDAFLAGS=-ccbin %CXX%"

meson setup builddir         ^
      -Dbuild_core=false     ^
      -Dbuild_examples=false ^
      -Dbuild_python=true    ^
      -Dnca_as_wheel=false   ^
      --prefix=%PREFIX%      ^
      --libdir=lib
if errorlevel 1 exit 1

meson compile -C builddir
if errorlevel 1 exit 1

meson install -C builddir
if errorlevel 1 exit 1

%PYTHON% -m pip install .                     ^
         --prefix="%PREFIX%"                  ^
         --no-build-isolation                 ^
         --no-deps                            ^
         --config-settings=build-dir=builddir
