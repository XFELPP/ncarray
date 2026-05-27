#!/bin/bash

export CXXFLAGS="${CXXFLAGS//-fvisibility-inlines-hidden/}"
export CXXFLAGS="${CXXFLAGS//-fvisibility=hidden/}"

export CUDAFLAGS="-ccbin ${CXX}"

meson setup builddir         \
      -Dbuild_core=false     \
      -Dbuild_examples=false \
      -Dbuild_python=true    \
      -Dnca_as_wheel=false   \
      -Ddebug=false          \
      --prefix=$PREFIX       \
      --libdir=lib
meson compile -C builddir
meson install -C builddir

${PYTHON} -m pip install .                     \
          --prefix="${PREFIX}"                 \
          --no-build-isolation                 \
          --no-deps                            \
          --config-settings=build-dir=builddir
