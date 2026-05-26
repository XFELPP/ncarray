@echo off

set "CUDAFLAGS=-ccbin %CXX%"

meson setup builddir         ^
      -Dbuild_core=true      ^
      -Dbuild_examples=false ^
      -Dbuild_python=false   ^
      -Dnca_as_wheel=false   ^
      --prefix=%PREFIX%      ^
      --libdir=lib           ^
      -Ddebug=false
if errorlevel 1 exit 1

meson compile -j 6 -C builddir -l 10
if errorlevel 1 exit 1

meson install -C builddir
if errorlevel 1 exit 1
