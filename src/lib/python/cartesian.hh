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

  template <typename Tuple, typename F, std::size_t... Is>
  bool try_tuple_invoke_impl(const py::tuple& t, F&& f, std::index_sequence<Is...>) {
    try {
      f(t[Is].cast<std::tuple_element_t<Is, Tuple>>()...);
      return true;
    } catch (const py::cast_error&) {
      return false;
    }
  }

  template <typename Tuple, typename F> bool
  try_tuple_invoke(const py::tuple& t, F&& f) {
    if (t.size() != std::tuple_size_v<Tuple>)
      return false;
    return try_tuple_invoke_impl<Tuple>(t,
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
            matched = try_tuple_invoke<Tuples>(t, [&](auto&&... args) {
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
