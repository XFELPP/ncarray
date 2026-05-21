#!/bin/bash

export CXXFLAGS="${CXXFLAGS//-fvisibility-inlines-hidden/}"
export CXXFLAGS="${CXXFLAGS//-fvisibility=hidden/}"

export CUDAFLAGS="-ccbin ${CXX}"

meson setup builddir -Dbuild_core=false -Dbuild_python=true -Dnca_as_wheel=false --prefix=$PREFIX --libdir=lib
meson compile -C builddir
meson install -C builddir
