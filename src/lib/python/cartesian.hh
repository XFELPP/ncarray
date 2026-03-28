/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef PYNCARRAY_CARTESIAN_HH
#define PYNCARRAY_CARTESIAN_HH

#include "ncarray/indexing.hh"

#include <pybind11/pybind11.h>

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <tuple>
#include <type_traits>

namespace py = pybind11;

/**
 * The mess of TypeList and related structs below construct a cartesian product
 * of supported indexing objects (integers, slices and ellipses). This template
 * metaprogramming allows the use of the variadic operator[] from Python with
 * automatic type conversion of the Python equivalents to C++ representations.
 * The compiler generates functions for all combinations - this is slow and
 * memory intensive. Up to 4 axes are currently supported.
 *
 * TODO: This can be improved by handling the special behaviour of ellipsis a little
 * more thoroughly - that is it removes combinations if present. The simplest
 * possiblity would be to just allow only a single ellipsis.
 */

namespace pyncarray {
  template <typename... Ts>
  struct TypeList {};

  template <typename L1, typename L2>
  struct Concat;
  template <typename... A, typename... B>
  struct Concat<TypeList<A...>, TypeList<B...>> {
    using type = TypeList<A..., B...>;
  };

  template <typename T, typename Tuple>
  struct PrependTuple;
  template <typename T, typename... Ts>
  struct PrependTuple<T, std::tuple<Ts...>> {
    using type = std::tuple<T, Ts...>;
  };

  template <typename T, typename List>
  struct PrependAll;
  template <typename T, typename... Tuples>
  struct PrependAll<T, TypeList<Tuples...>> {
    using type = TypeList<typename PrependTuple<T, Tuples>::type...>;
  };

  template <typename... Lists>
  struct ConcatMany;
  template <typename L1, typename L2, typename... Rest>
  struct ConcatMany<L1, L2, Rest...> {
    using type = typename ConcatMany<typename Concat<L1, L2>::type, Rest...>::type;
  };
  template <typename L>
  struct ConcatMany<L> {
    using type = L;
  };
  template <>
  struct ConcatMany<> {
    using type = TypeList<>;
  };

  template <typename Types, std::size_t N>
  struct Combos;
  template <typename... Ts, std::size_t N>
  struct Combos<TypeList<Ts...>, N> {
    using type =
      typename ConcatMany<
        typename PrependAll<
          Ts,
          typename Combos<TypeList<Ts...>, N-1>::type
        >::type...>::type;
  };
  template <typename... Ts>
  struct Combos<TypeList<Ts...>, 0> {
    using type = TypeList<std::tuple<>>;
  };

  template <typename Types, std::size_t MaxDim>
  struct AllCombos;
  template <typename Types, std::size_t N>
  struct AllCombos {
    using type = typename Concat<typename AllCombos<Types, N-1>::type, typename Combos<Types, N>::type>::type;
  };
  template <typename Types>
  struct AllCombos<Types, 0> {
    using type = TypeList<std::tuple<>>;
  };

  ncarray::Slice pyslice_to_slice(ssize_t axis_shape, py::slice slice) {
    ssize_t start, stop, step, length;
    if (!slice.compute(axis_shape, &start, &stop, &step, &length)) {
      throw py::error_already_set();
    }
    return ncarray::Slice(start, stop, step);
  }

  template <class TargetT, class ArrayT>
  auto convert_to_ncarray_type(const ArrayT& self,
                               std::size_t axis,
                               const py::handle& obj) {
    if constexpr (std::is_same_v<TargetT, ncarray::Slice>) {
      if (!py::isinstance<py::slice>(obj)) {
        // Cast errors are handled below, so trap non-slices and throw as cast_error
        // These are otherwise type_error exceptions. Instead of handling all these
        // exceptions, we convert everything to 1 kind.
        throw py::cast_error();
      }
      if (axis >= static_cast<size_t>(self.ndim())) {
        throw py::index_error("Too many indices for array!");
      }
      return pyslice_to_slice(self.shape(axis), obj.cast<py::slice>());
    } else if constexpr (std::is_same_v<TargetT, ncarray::Ellipsis>) {
      if (!py::isinstance<py::ellipsis>(obj)) {
        throw py::cast_error();
      }
      return ncarray::Ellipsis{};
    } else if constexpr (std::is_same_v<TargetT, ssize_t>) {
      ssize_t i = obj.cast<ssize_t>();
      // Add protection against out of bounds axes as well is indices
      if (axis >= static_cast<size_t>(self.ndim()) ||
          i < -self.shape(axis) ||
          i >= self.shape(axis)) {
        throw py::index_error("Index out of bounds!");
      }
      return i;
    } else {
      throw py::index_error("Invalid indexing argument!");
    }
  }

  template <class Tuple, class ArrayT, class F, std::size_t... Is>
  bool try_tuple_invoke_impl(const py::tuple& t,
                             const ArrayT& self,
                             F&& f,
                             std::index_sequence<Is...>) {
    try {
      f(convert_to_ncarray_type<std::tuple_element_t<Is, Tuple>>(self, Is, t[Is])...);
      return true;
    } catch (const py::cast_error&) {
      return false;
    }
  }

  template <class Tuple, class ArrayT, class F> bool
  try_tuple_invoke(const py::tuple& t, const ArrayT& self, F&& f) {
    if (t.size() != std::tuple_size_v<Tuple>) {
      return false;
    }
    return try_tuple_invoke_impl<Tuple>(t,
                                        self,
                                        std::forward<F>(f),
                                        std::make_index_sequence<std::tuple_size_v<Tuple>>{});
  }
  using IndexTypes = TypeList<ssize_t, ncarray::Slice, ncarray::Ellipsis>;
  using All = typename AllCombos<IndexTypes, 4>::type;

  template <class ArrayT, typename... Tuples>
  typename ArrayT::ViewType dispatch_cartesian(const py::tuple& t,
                                               const ArrayT& self,
                                               TypeList<Tuples...>) {
    typename ArrayT::ViewType result;
    bool matched = false;
    (
        [&] {
          if (!matched) {
            matched = try_tuple_invoke<Tuples>(t, self, [&](auto&&... args) {
              result = self[std::forward<decltype(args)>(args)...];
            });
          }
        }(),
        ...);
    if (!matched) {
      throw py::index_error("No matching indexing signature found!");
    }
    return result;
  }
} // pyncarray

#endif // PYNCARRAY_CARTESIAN_HH
