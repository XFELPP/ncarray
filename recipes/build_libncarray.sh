#!/bin/bash

export CXXFLAGS="${CXXFLAGS//-fvisibility-inlines-hidden/}"
export CXXFLAGS="${CXXFLAGS//-fvisibility=hidden/}"

export CUDAFLAGS="-ccbin ${CXX}"

meson setup builddir -Dbuild_core=true -Dbuild_python=false -Dnca_as_wheel=false --prefix=$PREFIX --libdir=lib -Ddebug=false
meson compile -j 6 -C builddir -l 10
meson install -C builddir
