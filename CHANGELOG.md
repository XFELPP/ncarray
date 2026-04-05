# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
