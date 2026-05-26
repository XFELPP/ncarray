#!/bin/bash

export CXXFLAGS="${CXXFLAGS//-fvisibility-inlines-hidden/}"
export CXXFLAGS="${CXXFLAGS//-fvisibility=hidden/}"

export CUDAFLAGS="-ccbin ${CXX}"

meson setup builddir         \
      -Dbuild_core=true      \
      -Dbuild_examples=false \
      -Dbuild_python=false   \
      -Dnca_as_wheel=false   \
      --prefix=$PREFIX       \
      --libdir=lib           \
      -Ddebug=false
meson compile -j 4 -C builddir -l 4
meson install -C builddir
