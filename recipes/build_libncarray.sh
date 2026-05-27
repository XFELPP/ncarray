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

MAX_JOBS=4
if [ "${cuda_compiler_version}" != "None" ]; then
    MAX_JOBS=2
fi

JOBS=${CPU_COUNT:-2}
if [ $JOBS -gt $MAX_JOBS ]; then
    JOBS=$MAX_JOBS
fi

meson compile -j $JOBS -C builddir
meson install -C builddir
