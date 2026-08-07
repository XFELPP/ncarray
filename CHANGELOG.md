# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.4] - 2026-08-07

### Fixed
- Correct lib directory search strategy to find macOS delocate wheel-repaired libraries

## [0.7.3] - 2026-08-02

### Added
- Update CI to allow builds for Python up to 3.15t. Builds are now done for versions 3.8-3.15t.

## [0.7.2] - 2026-08-01

### Changed
- Where possible switch from many `using` directives for bringing `std` and `cuda::std` objects into scope. Instead, alias to `hd_std`.

### Fixed
- Have the `make_tus.py` and `copy_cccl_headers.py` scripts run from `project_source_root` instead of `source_root` so that it works when `ncarray` is used as a subproject. (`source_root` is deprecated anyway)

## [0.7.1] - 2026-07-06

### Added
- A `build_tests` option for the build.

### Changed
- Refactored the top-level `meson.build` into files per-sub-directory.
- Renamed the `python` directory for the Python bindings to `pyncarray` and updated includes accordingly. This allows for their installation even when not in a wheel format. Previously this was avoided as they would be under a "python" directory in a global prefix which could cause confusion.

### Fixed
- All headers are always installed to avoid issues. There are some transitive includes which mean that even if not used, GPU/device headers are needed when including and linking `libncarray`. Host-only wheels did not provide them.

## [0.7.0] - 2026-07-01

### Added
- Added `StencilJITExtensions` for inserting pro/epilogue code into JIT-compiled Stencil kernels on device.
- Setup system for building multiple variants per release.
- Update documentation.

### Changed
- Reorganize to avoid users needing to have `asmjit` as an explicit dependency - headers no longer needed.
- Move GitHub actions to pinning by commit hash instead of tag/branch.

### Fixed
- Update the shared library directory returned by the Python binding's `get_lib_dir` for the split-wheel approach.

## [0.6.0] - 2026-06-13

### Added
- Similar runtime JIT compilation strategy for host-side expressions. Provide a rudimentary compilation phase and using `asmjit` output functions for linearizable expression tapes. Includes x86 and AArch64 (ARM) backends. Other ISAs not currently supported. Fallback to slow VM path remains.

### Changed
- Switch to using JIT compiler for the dynamic VM in device code.
- Additional index checks for out-of-bounds indexing in Python bindings.
- Map long double to double for device code. Map complex<long double> to complex<double> for device code.

### Fixed
- Fixed improper behaviour when slicing (indexing) a temporary view in Python bindings.
- Broken long double in device code.
- Fixed modulo for quad precision issues and complex numbers.

## [0.5.1] - 2026-06-03

### Changed
- Switch to split build for CI wheel jobs.

### Fixed
- Bug finding include path for runtime compiler depending on install type and how libncdevarrayjit is loaded.

## [0.5.0] - 2026-05-26

### Added
- Axis-aware reduction functions
- Stencils for CUDA kernels of single arrays
- JIT compilation of CUDA kernels with small numbers of operands, or using Stencils
- Conda recipes

### Changed
- Switch to using an expression engine (inspired by numexpr) for lazily evaluating array operations.
- Use initializer-list based indexing for indexing to references (via the proxy objects)
- Use StaticCorrds objects for alternative to indexing to reference (used internally)

### Fixed
- Issues with suboffsets and sliced views.
- Issues with view construction without data.
- Clang compatibility problems.

## [0.4.0] - 2026-04-03

### Added
- Use `ArrayElementProxy` for ergonomics when indexing. Linearized `operator[]` and `operator()` will return a proxy which casts to appropriate data types. This allows syntax such as: `float& item = ncarr(1,2,3)`. This is NOT type checked. There are `get` and `set` type checked versions: `auto proxy = ncarr(1,2,3); auto& item = proxy.get<float>()`.
- Generalized atomic operations for all supported data types for use in GPU code.
- Memory pools for scalar object creation device side.
- Fill out missing operators: inplace arithmetic, logical operators. Bring host and device code to parity in this regard. Python bindings for all new operators.
- `bool` casts for the vector data types (if any object is non-zero, `true`). Also add missing operators (`>=` and `<=`)
- Provide `ge` and `le` in `op_traits` for those comparisons when using complex data types.

### Fixed
- Various bugs in GPU kernels and device functions.
- Owner types not being read/write by default.
- Reference/value passing issues causing issues when trying to bind non-const temporaries.

### Changed
- Reorganize `array_operations.hh`: All implementations are namespaced into `host/..` and `device/...` headers. `engines.hh` defines a dispatch struct which should be used to access the implementations. `array_operations.hh` just uses the engines to define all array operations.
- Reorganize main include headers (`ncarrays.hh`, `soarrays.hh`, etc.) - create explicit specializations of all operators as well in the built shared libs. Do not expose `array_operations.hh` directly again after these build.

## [0.3.0] - 2026-03-31

### Added
- Linearized `operator[]` which takes one `ssize_t` - for faster indexing in hot loops over the view-based (and slice supporting) variadic version.
- `Float2/3/4` and `Double2/3/4` vector classes. Provide host and device access to a vector dtype equivalent to CUDA's `float2` etc.
- Additional test files.

### Fixed
- Kernel launch errors in the `GPUEngine`

## [0.2.0] - 2026-03-30

### Added
- Preliminary device array support
- Addition of `SOArrayPolicy` which provides `PEP3118`-like layouts for arrays. This adds flexibility (and complexity) over the `NCOffsetsPolicy` which supports only 1 "pointer axis".
- Initial testing suite.

### Fixed
- Bugs with copy/move constructors
- Clang compatibility
- Python-side not reflecting changes memory in C++.

### Changed
- Class hierarchy overhauled: all arrays are specializations of `ArrayImpl` which is composed (through inheritance) of a `Layout` and `Storage` class.

## [0.1.1] - 2026-03-16

### Changed
- Included `README.md` in the `pyproject.toml` so that it populates on `pypi` for the package description.

## [0.1.0] - 2026-02-02
- Initial release with basic support for `NCArray` types on CPU only.
