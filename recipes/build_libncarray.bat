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

set "JOBS=%CPU_COUNT%"
if not "%cuda_compiler_version%" == "None" (
    set "JOBS=2"
) else (
    if %JOBS% GTR 4 (
        set "JOBS=4"
    )
)

meson compile -j %JOBS% -C builddir
if errorlevel 1 exit 1

meson install -C builddir
if errorlevel 1 exit 1
