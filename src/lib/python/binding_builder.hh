/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCA_PYTHON_BINDING_BUILDER_HH
#define NCA_PYTHON_BINDING_BUILDER_HH

#include "python/utilities.hh"

#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"
#include "ncarray/ncarrays.hh"
#include "ncarray/op_code.hh"
#include "ncarray/soarrays.hh"

#ifdef NCA_HAS_CUDA
// This code doesn't actually have anything GPU-specific (__device__ or kernels)
// They are the specializations for GPU arrays
#include "ncarray/ncdevarrays.cuh"
#include "ncarray/sodevarrays.cuh"
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;

// On Windows need to export symbols for DLLs
#ifdef NCA_BUILD_PYNCARRAY_API
#define NCA_PYNCARRAY_API __declspec(dllexport)
#else
#define NCA_PYNCARRAY_API __declspec(dllimport)
#endif
#else
#include <sys/types.h>

#define NCA_PYNCARRAY_API
#endif

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace py = pybind11;

namespace pyncarray {
  /**
   * Global toggle for whether Python will eagerly convert Expr objects to arrays.
   */
  extern NCA_PYNCARRAY_API bool g_eager_eval;

  NCA_PYNCARRAY_API void set_eager(bool eager);

  NCA_PYNCARRAY_API bool is_eager();

  template <class LPolicy, class MemType>
  inline bool is_nc_array(const py::handle& val) {
    using ncarray::ArrayImpl;
    using ncarray::NCOffsetsPolicy;

    using ViewType = typename ncarray::StoragePolicyTraits<MemType>::View;
    using RefType = typename ncarray::StoragePolicyTraits<MemType>::Ref;
    using OwnerType = typename ncarray::StoragePolicyTraits<MemType>::Owner;

    using ArrView = ArrayImpl<LPolicy, ViewType>;
    using ArrRef = ArrayImpl<LPolicy, RefType>;
    using ArrOwner = ArrayImpl<LPolicy, OwnerType>;
    if (py::isinstance<ArrView>(val) ||
        py::isinstance<ArrRef>(val) ||
        py::isinstance<ArrOwner>(val)) {
      return true;
    }
    return false;
  };

  inline bool needs_device_vm(const py::handle& node) {
#ifdef NCA_HAS_CUDA
    if (is_nc_array<ncarray::NCOffsetsPolicy, ncarray::DevTag>(node) ||
        is_nc_array<ncarray::SOArrayPolicy, ncarray::DevTag>(node)) {
      return true;
    }
    if (py::isinstance<ncarray::ExprMVNode<ncarray::DevTag>>(node)) {
      return true;
    }
#endif
    return false;
  }

  template <typename MemType>
  void _build_node_helper(ncarray::ExprMVNode<MemType>& self, py::object other) {
    using VPolicy = typename ncarray::StoragePolicyTraits<MemType>::View;
    if (py::isinstance<ncarray::ExprMVNode<MemType>>(other)) {
      self.build_node(other.cast<ncarray::ExprMVNode<MemType>&>());
    } else if (is_nc_array<ncarray::NCOffsetsPolicy, MemType>(other)) {
      self.build_node(other.cast<ncarray::ArrayImpl<ncarray::NCOffsetsPolicy, VPolicy>&>());
    } else if (is_nc_array<ncarray::SOArrayPolicy, MemType>(other)) {
      self.build_node(other.cast<ncarray::ArrayImpl<ncarray::SOArrayPolicy, VPolicy>&>());
    } else {
      auto cast_op = [&](auto&& val) {
        self.build_node(val);
      };
      std::visit(cast_op, other.cast<ncarray::Scalar>());
    }
  }

  template <typename MemType>
  ncarray::NCOwnerFor<MemType> eval_python_expr(const py::object& pyexpr) {
    if (py::isinstance<ncarray::ExprMVNode<MemType>>(pyexpr)) {
      return ncarray::NCOwnerFor<MemType>(pyexpr.cast<ncarray::ExprMVNode<MemType>>());
    } else {
      ncarray::ExprMVNode<MemType> expr;
      _build_node_helper<MemType>(expr, pyexpr);

      return ncarray::NCOwnerFor<MemType>(expr);
    }
  }

#define REGISTER_EXPR_OPERATION(PYMETHOD, CODE)                                       \
  .def("__" PYMETHOD "__", [](ncarray::ExprMVNode<MemType>& self, py::object other) { \
    _build_node_helper(self, other);                                                  \
    self.instrs.push_back(ncarray::pack_instruction(ncarray::OpCode::CODE, 0));       \
    return py::cast(self, py::return_value_policy::reference);                        \
  })

  template <typename MemType>
  void register_expr_class(py::module_& m, const std::string& name) {
    py::classh<ncarray::ExprMVNode<MemType>>(m, name.c_str())
      REGISTER_EXPR_OPERATION("add", ADD)
      REGISTER_EXPR_OPERATION("sub", SUB)
      REGISTER_EXPR_OPERATION("mul", MUL)
      REGISTER_EXPR_OPERATION("eq", EQ)
      REGISTER_EXPR_OPERATION("ne", NE)
      REGISTER_EXPR_OPERATION("lt", LT)
      REGISTER_EXPR_OPERATION("le", LE)
      REGISTER_EXPR_OPERATION("gt", GT)
      REGISTER_EXPR_OPERATION("GE", GE)
      REGISTER_EXPR_OPERATION("and", LAND)
      REGISTER_EXPR_OPERATION("or", LOR);
  }

#undef REGISTER_EXPR_OPERATION

  /**
   * @def REGISTER_OPERATION(PYMETHOD, OP)
   * @brief A helper to attach a dunder to a class binding for operator overloads.
   * @example REGISTER_OPERATION("add", +) binds operator+(...) to __add__
   * @todo We currently need to convert arrays to view, because the C++ lib cannot
   *       take the array directly.
   */
#define REGISTER_OPERATION(PYMETHOD, OP)                                                         \
    .def("__" PYMETHOD "__",                                                                     \
         [](const ArrayT& self,                                                                  \
            const ncarray::ArrayImpl<typename ArrayT::LayoutPolicy,                              \
                                     typename ncarray::StoragePolicyTraits<                      \
                                     typename ArrayT::MemType>::View>& other) {                  \
      auto expr = self.view() OP other;                                                          \
      if (is_eager()) {                                                                          \
        return py::cast(ncarray::NCOwnerFor<typename ArrayT::MemType>(expr));                    \
      }                                                                                          \
      return py::cast(expr);                                                                     \
    },                                                                                           \
      py::is_operator())                                                                         \
    .def("__" PYMETHOD "__", [](const ArrayT& self, const py::array& other) {                    \
      auto expr = self.view() OP pyarray_to_view<typename ArrayT::ViewType>(other, self.ndim()); \
      if (is_eager()) {                                                                          \
        return py::cast(ncarray::NCOwnerFor<typename ArrayT::MemType>(expr));                    \
      }                                                                                          \
      return py::cast(expr);                                                                     \
    },                                                                                           \
      py::is_operator())                                                                         \
    .def("__r" PYMETHOD "__", [](const ArrayT& self, const py::array& other) {                   \
      auto view = pyarray_to_view<typename ArrayT::ViewType>(other, self.ndim());                \
      auto expr = view OP self.view();                                                           \
      if (is_eager()) {                                                                          \
        return py::cast(ncarray::NCOwnerFor<typename ArrayT::MemType>(expr));                    \
      }                                                                                          \
      return py::cast(expr);                                                                     \
    },                                                                                           \
      py::is_operator())                                                                         \
    .def("__" PYMETHOD "__", [](const ArrayT& self, const ncarray::Scalar& other) {              \
      auto expr = self.view() OP other;                                                          \
      if (is_eager()) {                                                                          \
        return py::cast(ncarray::NCOwnerFor<typename ArrayT::MemType>(expr));                    \
      }                                                                                          \
      return py::cast(expr);                                                                     \
    },                                                                                           \
      py::is_operator())

  /**
   * @def REGISTER_INPLACE_OPERATION(PYMETHOD, OP)
   * @brief A helper to attach a dunder to a class binding for inplace operator overloads.
   * @example REGISTER_INPLACE_OPERATION("iadd", +=) binds operator+=(...) to __iadd__
   * @todo We currently need to convert arrays to view, because the C++ lib cannot
   *       take the array directly.
   */
#define REGISTER_INPLACE_OPERATION(PYMETHOD, OP)                                    \
    .def("__" PYMETHOD "__", [](ArrayT& self, const ArrayT& other) {                \
      auto view = other.view();                                                     \
      self OP view;                                                                 \
      return self;                                                                  \
    },                                                                              \
      py::is_operator())                                                            \
    .def("__" PYMETHOD "__", [](ArrayT& self, const py::array& other) {             \
      auto view = pyarray_to_view<typename ArrayT::ViewType>(other, self.ndim());   \
      self OP view;                                                                 \
      return self;                                                                  \
    },                                                                              \
      py::is_operator())                                                            \
    .def("__" PYMETHOD "__", [](ArrayT& self, const ncarray::Scalar& other) {       \
      self OP other;                                                                \
      return self;                                                                  \
    },                                                                              \
      py::is_operator())

  /**
   * Helper function to attach common methods to a Python binding for an array
   * specialization.
   *
   * All array specializations generally have the same functions in C++ and in
   * their Python bindings (plus/minus some speciality features). This function
   * just attaches those all.
   *
   * @tparam ArrayT The kind of array being used. This is a full array specifier,
   *         including storage+layout specifier.
   * @param[in] arr_cl The class of the Python binding.
   */
  template <typename ArrayT>
  void register_common_array_methods(py::classh<ArrayT>& arr_cl) {
    using LayoutPolicy = typename ArrayT::LayoutPolicy;
    using HostViewType = ncarray::ArrayImpl<LayoutPolicy, ncarray::ViewPolicy>;
    using ViewType = typename ArrayT::ViewType;

    using ViewOrScalar = std::variant<ncarray::Scalar,ViewType>;

    arr_cl.def("__repr__", &ArrayT::repr, py::is_operator())
    .def_property_readonly("shape", [](const ArrayT& self) -> py::tuple {
      auto* shape = self.shape();
      py::list l;
      for (ssize_t i = 0; i < self.ndim(); ++i) {
        l.append(shape[i]);
      }
      return l;
    })
    .def_property_readonly("strides", [](const ArrayT& self) -> py::tuple {
      auto* strides = self.strides();
      py::list l;
      for (ssize_t i = 0; i < self.ndim(); ++i) {
        l.append(strides[i]);
      }
      return l;
    })
    // --- Standard Container Methods --- //
    .def("__len__", [](const ArrayT& self) {
      if (self.ndim() > 0) {
        return self.shape(0);
      }
      return ssize_t(0);
    })
    //.def("__iter__", [](const ArrayT& self) {
    //    return py::make_iterator(self.begin(), self.end());
    //})
    // --- Array-Like Methods (indexing, size, shapedtype, etc) --- //
    .def_property_readonly("size",
                           &ArrayT::size,
                           "The number of items in the array type.")
    .def_property_readonly("ndim",
                           &ArrayT::ndim,
                           "The number of dimensions in the array type.")
    .def_property_readonly("itemsize",
                           &ArrayT::itemsize,
                           "The size in bytes of a single item in the array type.")
    .def_property_readonly("nbytes",
                           &ArrayT::nbytes,
                           "The total size in bytes of all items in the array type.")
    .def_property_readonly("dtype",
                           &ArrayT::dtype,
                           "The data type of the underlying elements.")
    .def_property_readonly("is_contiguous",
                           &ArrayT::is_contiguous,
                           "Check if the underlying data is contiguous")
    .def("squeeze",
         &ArrayT::squeeze,
         "Collapse and remove all axes of length 1.")
    .def("astype",
         [](const ArrayT& self, ncarray::DType& dtype_out) {
           return self.astype(dtype_out);
         },
         py::arg("dtype"),
         "Convert an NCArray* to the specified data type.")
    .def("view",
         &ArrayT::view,
         "Convert the array to a *View type for use in view-only APIs (like kernels).")
    .def("__getitem__",
         [](const ArrayT& self, py::object idx) -> ViewOrScalar {
           // NOTE: The Python bindings diverge from the C++ library on scalars.
           //       For simplicity, in Python, scalars are returned as scalars.
           //       In C++, they remain as an object tied to the array class.
           ssize_t num_indices { 0 };
           std::vector<ncarray::IndexItem> indices;
           if (py::isinstance<py::int_>(idx)) {
             auto idx_val = idx.cast<ssize_t>();
             if (idx_val < -self.shape(0) || idx_val >= self.shape(0)) {
               throw py::index_error("Index out of bounds!");
             }
             indices.push_back(ncarray::IndexItem { idx_val });
             num_indices++;
           } else if (py::isinstance<py::slice>(idx)) {
             auto slice = pyslice_to_slice(self.shape(0), idx.cast<py::slice>());
             indices.push_back(ncarray::IndexItem { slice });
             num_indices++;
           } else if (py::isinstance<py::ellipsis>(idx)) {
             indices.push_back(ncarray::IndexItem { ncarray::Ellipsis {} });
             num_indices++;
           } else if (py::isinstance<py::tuple>(idx)) {
             py::tuple tup { idx.cast<py::tuple>() };
             std::vector<ncarray::IndexItem> tup_indices = pytuple_to_indices(self,
                                                                              tup);

             indices.insert(indices.end(), tup_indices.begin(), tup_indices.end());
             num_indices += tup.size();
           } else {
             throw py::type_error("Invalid indexing argument!");
           }

           ViewType view = self.view_from_indices(indices.data(), num_indices);
           // For convenience convert scalars to... scalars (as opposed to 0-D array)
           // But -- check if using GPU memory though
           using MemType = typename std::decay_t<ArrayT>::MemType;
           if constexpr (!std::is_same_v<MemType, ncarray::DevTag>) {
             if (view.ndim() == 0) {
               return view.get_scalar(view.data());
             }
           }
           return view;
         },
         py::is_operator(),
         py::return_value_policy::reference)
    .def("__setitem__",
         [](const ArrayT& self, py::object idx, py::object val) {
           ssize_t num_indices { 0 };
           std::vector<ncarray::IndexItem> indices;
           if (py::isinstance<py::int_>(idx)) {
             auto idx_val = idx.cast<ssize_t>();
             if (idx_val < -self.shape(0) || idx_val >= self.shape(0)) {
               throw py::index_error("Index out of bounds!");
             }
             indices.push_back(ncarray::IndexItem { idx_val });
             num_indices++;
           } else if (py::isinstance<py::slice>(idx)) {
             auto slice = pyslice_to_slice(self.shape(0), idx.cast<py::slice>());
             indices.push_back(ncarray::IndexItem { slice });
             num_indices++;
           } else if (py::isinstance<py::ellipsis>(idx)) {
             indices.push_back(ncarray::IndexItem { ncarray::Ellipsis {} });
             num_indices++;
           } else if (py::isinstance<py::tuple>(idx)) {
             py::tuple tup { idx.cast<py::tuple>() };
             std::vector<ncarray::IndexItem> tup_indices = pytuple_to_indices(self, tup);

             indices.insert(indices.end(), tup_indices.begin(), tup_indices.end());
             num_indices += tup.size();
           } else {
             throw py::type_error("Invalid indexing argument!");
           }

           // We use this helper function instead of variadic operator[] -- tried
           // doing it before with a cartesian product to create all the function
           // overloads but it was prohibitively costly for compilation times
           ViewType view = self.view_from_indices(indices.data(), num_indices);
           if (py::isinstance<py::array>(val)) {
             // Make sure that we always use a HOST view even with GPU support
             auto rhs_view = pyarray_to_view<HostViewType>(val.cast<py::array>());
             view.assign(rhs_view);
           } else if (py::isinstance<ArrayT>(val)) {
             view.assign(val.cast<ArrayT&>());
           } else {
             // See if its another nc/so array type
             auto assign_op = [&] <typename LayoutP, typename STag> () {
               using VP =
                 typename ncarray::StoragePolicyTraits<STag>::View;
               using RP =
                 typename ncarray::StoragePolicyTraits<STag>::Ref;
               using OP =
                 typename ncarray::StoragePolicyTraits<STag>::Owner;
               using ArrView = ncarray::ArrayImpl<LayoutP, VP>;
               using ArrRef = ncarray::ArrayImpl<LayoutP, RP>;
               using ArrOwner = ncarray::ArrayImpl<LayoutP, OP>;
               if (py::isinstance<ArrView>(val) ||
                   py::isinstance<ArrRef>(val)  ||
                   py::isinstance<ArrOwner>(val)) {
                 view.assign(val.cast<ArrView>());
                 return true;
               }
               return false;
             };
#ifdef NCA_HAS_CUDA
             // TODO: When cross-layouts are added to the shared libs add full matrix
             // I.e. test NC to SOArray, SOArray to NC etc.
             // For now, just use the single LayoutPolicy
             if (assign_op.template operator()<LayoutPolicy, ncarray::DevTag>()) {
               return;
             }
             /*
             if (assign_op.template operator()<ncarray::NCOffsetsPolicy, ncarray::DevTag>()) {
               return;
             }
             if (assign_op.template operator()<ncarray::SOArrayPolicy, ncarray::DevTag>()) {
               return;
             }
             */
#endif
             if (assign_op.template operator()<LayoutPolicy, ncarray::HostTag>()) {
               return;
             }
             /*
             if (assign_op.template operator()<ncarray::NCOffsetsPolicy, ncarray::HostTag>()) {
               return;
             }
             if (assign_op.template operator()<ncarray::SOArrayPolicy, ncarray::HostTag>()) {
               return;
             }
             */
             // Convertible to scalar
             // Use the algorithm directly to avoid the variant gets
             try {
               view.fill(val.cast<ncarray::Scalar>());
             } catch (...) {
               throw py::type_error("Unrecognized type for assignment!");
             }
           }
         },
         py::is_operator())
    // --- Array copy/assignment/fill methods --- //
    .def("fill", &ArrayT::fill, "Broadcast a scalar value into an array.")
    //.def("assign", &ArrayT::assign, "Assign (copy) another array into this one.")
    .def("to_host", [](const ArrayT& self,
                       std::optional<ncarray::DType> dtype) {
#ifdef NCA_HAS_CUDA
      ncarray::DType out_dtype = dtype.value_or(self.dtype());

      using MemType = typename std::decay_t<ArrayT>::MemType;
      if (std::is_same_v<MemType, ncarray::DevTag>) {
        using LayoutType = typename ArrayT::LayoutPolicy;
        ncarray::ArrayImpl<LayoutType, ncarray::OwnerPolicy> h_arr(self.ndim(),
                                                                   self.shape(),
                                                                   out_dtype);
        auto copy_into = [&] <typename DestT> () {
          self.template copy_into_astype<DestT>(reinterpret_cast<DestT*>(h_arr.data()));
        };

        host_dispatch(out_dtype, copy_into);

        return h_arr;
      }
#endif
      throw std::runtime_error("Array is already on host!");
    },
      py::arg("dtype") = py::none(),
      "Transfer a device array to host. If already on the host, return the same array.")
    .def("to_device", [](const ArrayT& self,
                         std::optional<ncarray::DType> dtype) {
#ifdef NCA_HAS_CUDA
      ncarray::DType out_dtype = dtype.value_or(self.dtype());

      using MemType = typename std::decay_t<ArrayT>::MemType;
      if (std::is_same_v<MemType, ncarray::HostTag>) {
        using LayoutType = typename ArrayT::LayoutPolicy;
        ncarray::ArrayImpl<LayoutType, ncarray::DevOwnerPolicy> d_arr(self.ndim(),
                                                                      self.shape(),
                                                                      out_dtype);

        // NOTE: Host type arrays don't have device copy semantics -- we do the opposite
        // to above. The destination (on device) pulls in the data from host.
        // The device version of assigning is more general and can handle the transfer
        // NOTE: For simplicity and reducing build size, we also limit everything to views.
        // So create a view of myself before continuing
        //auto my_view = self.view();
        //d_arr.assign(my_view);
        d_arr.assign(self);

        return d_arr;
      } else {
        throw std::runtime_error("Array is already on device!");
      }
#else
      throw std::runtime_error("Device transfer not available on this platform!");
#endif
    },
      py::arg("dtype") = py::none(),
      "Transfer a host array to device. If already on device, return the same array.")
    // --- Array Reduction Methods (Reduce to scalar) --- //

      // sum
    .def("sum", [](const ArrayT& self) {
      return self.sum();
    })
    .def("sum", [](const ArrayT& self, std::vector<ssize_t>& axes) {
      return self.sum(axes);
    },
      py::arg("axis"))

      // max, argmax
    .def("max", [](const ArrayT& self) {
      return self.max();
    })
    .def("max", [](const ArrayT& self, std::vector<ssize_t>& axes) {
      return self.max(axes);
    },
      py::arg("axis"))
    .def("argmax", [](const ArrayT& self) {
      return self.argmax();
    })
    .def("argmax", [](const ArrayT& self, std::vector<ssize_t>& axes) {
      return self.argmax(axes);
    },
      py::arg("axis"))

      // min, argmin
    .def("min", [](const ArrayT& self) {
      return self.min();
    })
    .def("min", [](const ArrayT& self, std::vector<ssize_t>& axes) {
      return self.min(axes);
    },
      py::arg("axis"))
    .def("argmin", [](const ArrayT& self) {
      return self.argmin();
    })
    .def("argmin", [](const ArrayT& self, std::vector<ssize_t>& axes) {
      return self.argmin(axes);
    },
      py::arg("axis"))

      // mean, std, var
    .def("mean", [](const ArrayT& self) {
      return self.mean();
    })
    .def("mean", [](const ArrayT& self, std::vector<ssize_t>& axes) {
      return self.mean(axes);
    },
      py::arg("axis"))
    .def("var", [](const ArrayT& self, ssize_t ddof) {
      return self.var(ddof);
    },
      py::arg("ddof") = 0.0)
    .def("var", [](const ArrayT& self, std::vector<ssize_t>& axes, ssize_t ddof) {
      return self.var(axes, ddof);
    },
      py::arg("axis"),
      py::arg("ddof") = 0.0)
    .def("std", [](const ArrayT& self, ssize_t ddof) {
      return self.std(ddof);
    },
      py::arg("ddof") = 0.0)
    .def("std", [](const ArrayT& self, std::vector<ssize_t>& axes, ssize_t ddof) {
      return self.std(axes, ddof);
    },
      py::arg("axis"),
      py::arg("ddof") = 0.0)
      // all, any
    .def("all", [](const ArrayT& self) {
      return self.all();
    })
    .def("all", [](const ArrayT& self, std::vector<ssize_t>& axes) {
      return self.all(axes);
    },
      py::arg("axis"))
    .def("any", [](const ArrayT& self) {
      return self.any();
    })
    .def("any", [](const ArrayT& self, std::vector<ssize_t>& axes) {
      return self.any(axes);
    },
      py::arg("axis"))

    // --- Binary Arithmetic Methods --- //
    REGISTER_OPERATION("add", +)
    REGISTER_OPERATION("sub", -)
    REGISTER_OPERATION("mul", *)
    REGISTER_OPERATION("truediv", /)
    // --- Inplace Binary Arithmetic Methods --- //
    REGISTER_INPLACE_OPERATION("iadd", +=)
    REGISTER_INPLACE_OPERATION("isub", -=)
    REGISTER_INPLACE_OPERATION("imul", *=)
    REGISTER_INPLACE_OPERATION("itruediv", /=)
    // --- Binary Comparisons --- //
    REGISTER_OPERATION("eq", ==)
    REGISTER_OPERATION("ne", !=)
    REGISTER_OPERATION("lt", <)
    REGISTER_OPERATION("le", <=)
    REGISTER_OPERATION("gt", >)
    REGISTER_OPERATION("ge", >=)
    // --- Logical Operations --- //
    REGISTER_OPERATION("and", &&)
    REGISTER_OPERATION("or", ||)
    // --- Inplace Logical Operations (Boolean Arrays Only) --- //
    REGISTER_INPLACE_OPERATION("iand", &=)
    REGISTER_INPLACE_OPERATION("ior", |=)
    // --- Scattering Operations --- //
      /*
    .def("scatter_add", [](ArrayT& self,
                           const py::object& indices,
                           const py::object& src) {
#ifdef NCA_HAS_CUDA
      // To keep the combinatorial explosion down while dev/host
      // scatter add is supported in arbitrary combos, we only include
      // dev views in the shared lib. So must cast and viewify beforehand
      using MemType = typename std::decay_t<ArrayT>::MemType;
      if constexpr (std::is_same_v<MemType, ncarray::DevTag>) {
        auto is_arr_type = [&] <typename LayoutP, typename STag> (const py::object& val) {
          using VP = typename ncarray::StoragePolicyTraits<STag>::View;
          using RP = typename ncarray::StoragePolicyTraits<STag>::Ref;
          using OP = typename ncarray::StoragePolicyTraits<STag>::Owner;
          using ArrView = ncarray::ArrayImpl<LayoutP, VP>;
          using ArrRef = ncarray::ArrayImpl<LayoutP, RP>;
          using ArrOwner = ncarray::ArrayImpl<LayoutP, OP>;
          if (py::isinstance<ArrView>(val) ||
              py::isinstance<ArrRef>(val)  ||
              py::isinstance<ArrOwner>(val)) {
            return true;
          }
          return false;
        };
        if (is_arr_type.template operator()<LayoutPolicy, ncarray::HostTag>(indices)) {
          auto h_idx = indices.cast<ncarray::ArrayImpl<LayoutPolicy, ncarray::ViewPolicy>>();
          ncarray::ArrayImpl<LayoutPolicy, ncarray::DevOwnerPolicy> d_idx(h_idx.ndim(),
                                                                        h_idx.shape(),
                                                                        h_idx.dtype());

          d_idx.assign(h_idx);
          if (is_arr_type.template operator()<LayoutPolicy, ncarray::HostTag>(src)) {
            auto h_src = src.cast<ncarray::ArrayImpl<LayoutPolicy, ncarray::ViewPolicy>>();
            ncarray::ArrayImpl<LayoutPolicy, ncarray::DevOwnerPolicy> d_src(h_src.ndim(),
                                                                          h_src.shape(),
                                                                          h_src.dtype());
            d_src.assign(h_src);
            auto self_view = self.view();
            self_view.scatter_add(d_idx.view(), d_src.view());
          } else {
            auto d_src = src.cast<ncarray::ArrayImpl<LayoutPolicy, ncarray::DevViewPolicy>>();
            auto self_view = self.view();
            self_view.scatter_add(d_idx.view(), d_src.view());
          }
        } else {
          auto d_idx = indices.cast<ncarray::ArrayImpl<LayoutPolicy, ncarray::DevViewPolicy>>();
          if (is_arr_type.template operator()<LayoutPolicy, ncarray::HostTag>(src)) {
            auto h_src = src.cast<ncarray::ArrayImpl<LayoutPolicy, ncarray::ViewPolicy>>();
            ncarray::ArrayImpl<LayoutPolicy, ncarray::DevOwnerPolicy> d_src(h_src.ndim(),
                                                                          h_src.shape(),
                                                                          h_src.dtype());
            d_src.assign(h_src);
            auto self_view = self.view();
            self_view.scatter_add(d_idx.view(), d_src.view());
            } else {
            auto d_src = src.cast<ncarray::ArrayImpl<LayoutPolicy, ncarray::DevViewPolicy>>();
            auto self_view = self.view();
            self_view.scatter_add(d_idx, d_src);
          }
        }
        return;
      }
#endif
      auto self_view = self.view();
      auto idx_view = indices.cast<ncarray::ArrayImpl<LayoutPolicy, ncarray::ViewPolicy>>();
      auto src_view = src.cast<ncarray::ArrayImpl<LayoutPolicy, ncarray::ViewPolicy>>();

      self_view.scatter_add(idx_view, src_view);
    })
      */
    // NumPy protocol compatibility
    // __array__(self, dtype=None, copy=None)
    .def("__array__", [](const ArrayT& self,
                         const py::object& dtype,
                         const py::object& copy) {
      // If this is on GPU, the internal copy structures will figure it out
      return ncarr_to_numpy(self);
    },
         py::arg("dtype") = py::none(),
         py::arg("copy") = py::none())
    // __array_priority__ attribute - set high so NCArray* funcs used, and is returned
    .def_property_readonly_static("__array_priority__", [](const py::object&) {
      return 100.0;
    })
    .def("__array_ufunc__", [](const ArrayT& self,
                               py::handle ufunc,
                               py::str method,
                               py::args args,
                               py::kwargs kwargs) {
      if (method.cast<std::string>() != "__call__") {
        return py::none().cast<py::object>();
      }

      // For now, just convert to NumPy
      // TODO: Optimize this with NCArray* directly
      py::list new_args;
      for (const auto& arg : args) {
        if (py::isinstance<ArrayT>(arg)) {
          new_args.append(ncarr_to_numpy(arg.cast<ArrayT>()));
        } else {
          new_args.append(arg);
        }
      }
      return ufunc(*new_args, **kwargs);
    });
#undef REGISTER_OPERATION
#undef REGISTER_OPERATION_NOSCALAR
#undef REGISTER_INPLACE_OPERATION
  }
}  // namespace pyncarray

#endif // NCA_PYTHON_BINDING_BUILDER_HH
