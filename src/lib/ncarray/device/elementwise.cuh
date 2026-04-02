/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_DEVICE_ELEMENTWISE_HH
#define NCARRAY_DEVICE_ELEMENTWISE_HH

#include "ncarray/array_traits.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#ifndef NCARRAY_MAX_NDIM
#define NCARRAY_MAX_NDIM 10
#endif

namespace ncarray {
  namespace device {
    namespace impl {
      template <
        typename T,
        typename ResultT,
        class LeftT,
        class RightT,
        class OutT,
        class Op
      >
      __device__ inline void block_binary_transform(const LeftT& left,
                                                    const RightT& right,
                                                    OutT& result,
                                                    Op op) {
        unsigned tid { threadIdx.x };
        unsigned stride { blockDim.x };

        for (ssize_t i = static_cast<ssize_t>(tid); i < left.size(); i += stride) {
          ssize_t coords[NCARRAY_MAX_NDIM];
          ssize_t tmp_idx { i };

          for (ssize_t d = left.ndim() - 1; d >= 0; --d) {
            coords[d] = tmp_idx % left.shape(d);
            tmp_idx /= left.shape(d);
          }

          const void* lhs_ptr = const_cast<void*>(left.data());
          for (int d = 0; d < left.ndim(); ++d) {
            lhs_ptr = left.advance(lhs_ptr, d, coords[d]);
          }


          void* rhs_ptr = const_cast<void*>(right.data());

          ssize_t diff = left.ndim() - right.ndim();
          for (int d = 0; d < right.ndim(); ++d) {
            ssize_t r_coord = (right.shape(d) == 1) ? 0 : coords[d + diff];
            rhs_ptr = right.advance(rhs_ptr, d, r_coord);
          }
          void* res_ptr = result.data();
          for (int d = 0; d < result.ndim(); ++d) {
            res_ptr = result.advance(res_ptr, d, coords[d]);
          }

          *static_cast<ResultT*>(res_ptr) =
            op(*static_cast<const T*>(lhs_ptr), *static_cast<const T*>(rhs_ptr));
        }
      }

      template <typename T, class LeftT, class RightT, class Op>
      __device__ inline void block_inplace_binary_transform(LeftT& left,
                                                            const RightT& right,
                                                            Op op) {
        unsigned tid { threadIdx.x };
        unsigned stride { blockDim.x };

        for (ssize_t i = static_cast<ssize_t>(tid); i < left.size(); i += stride) {
          ssize_t coords[NCARRAY_MAX_NDIM];
          ssize_t tmp_idx { i };

          for (ssize_t d = left.ndim() - 1; d >= 0; --d) {
            coords[d] = tmp_idx % left.shape(d);
            tmp_idx /= left.shape(d);
          }

          void* lhs_ptr = const_cast<void*>(left.data());
          for (int d = 0; d < left.ndim(); ++d) {
            lhs_ptr = left.advance(lhs_ptr, d, coords[d]);
          }

          void* rhs_ptr = const_cast<void*>(right.data());

          ssize_t diff = left.ndim() - right.ndim();
          for (int d = 0; d < right.ndim(); ++d) {
            ssize_t r_coord = (right.shape(d) == 1) ? 0 : coords[d + diff];
            rhs_ptr = right.advance(rhs_ptr, d, r_coord);
          }
          *static_cast<T*>(lhs_ptr) =
            op(*static_cast<T*>(lhs_ptr), *static_cast<const T*>(rhs_ptr));
        }
      }

      /**
       * Scalar broadcast version of block_binary_transform.
       */
      template <typename T, class LeftT, class OutT, class Op>
      __device__ inline void block_binary_scalar_transform(const LeftT& left,
                                                           const T& scalar_val,
                                                           OutT& result,
                                                           Op op) {
        unsigned tid { threadIdx.x };
        unsigned stride { blockDim.x };

        for (ssize_t i = static_cast<ssize_t>(tid); i < left.size(); i += stride) {
          ssize_t coords[NCARRAY_MAX_NDIM];
          ssize_t tmp_idx { i };
          for (ssize_t d = left.ndim() - 1; d >= 0; --d) {
            coords[d] = tmp_idx % left.shape(d);
            tmp_idx /= left.shape(d);
          }

          const void* lhs_ptr = left.data();
          for (int d = 0; d < left.ndim(); ++d) {
            lhs_ptr = left.advance(lhs_ptr, d, coords[d]);
          }

          void* res_ptr = result.data();
          for (int d = 0; d < result.ndim(); ++d) {
            res_ptr = result.advance(res_ptr, d, coords[d]);
          }
          *static_cast<T*>(res_ptr) =
            op(*static_cast<const T*>(lhs_ptr), scalar_val);
        }
      }

      template <typename T, class LeftT, class Op>
      __device__ inline void block_inplace_binary_scalar_transform(LeftT& left,
                                                                   const T& scalar_val,
                                                                   Op op) {
        unsigned tid { threadIdx.x };
        unsigned stride { blockDim.x };

        for (ssize_t i = static_cast<ssize_t>(tid); i < left.size(); i += stride) {
          ssize_t coords[NCARRAY_MAX_NDIM];
          ssize_t tmp_idx { i };

          for (ssize_t d = left.ndim() - 1; d >= 0; --d) {
            coords[d] = tmp_idx % left.shape(d);
            tmp_idx /= left.shape(d);
          }

          void* lhs_ptr = const_cast<void*>(left.data());
          for (int d = 0; d < left.ndim(); ++d) {
            lhs_ptr = left.advance(lhs_ptr, d, coords[d]);
          }

          *static_cast<T*>(lhs_ptr) = op(*static_cast<T*>(lhs_ptr), scalar_val);
        }
      }
    } // namespace impl

    // --- Binary non-broadcast operations (same shape) --- //

    template <typename T, class LeftT, class RightT, class OutT>
    __device__ inline void block_add(const LeftT& left,
                                     const RightT& right,
                                     OutT& out) {

      impl::block_binary_transform<T, T, LeftT, RightT, OutT>(left,
                                                              right,
                                                              out,
                                                              [] __device__ (auto a, auto b) {
                                                                return a + b;
                                                              });
      //__syncthreads();
    }

    template <typename T, class LeftT, class RightT, class OutT>
    __device__ inline void block_sub(const LeftT& left,
                                     const RightT& right,
                                     OutT& out) {
      impl::block_binary_transform<T, T, LeftT, RightT, OutT>(left,
                                                              right,
                                                              out,
                                                              [] __device__(auto a, auto b) {
                                                                return a - b;
                                                              });
      //__syncthreads();
    }

    template <typename T, class LeftT, class RightT, class OutT>
    __device__ inline void block_mul(const LeftT& left,
                                     const RightT& right,
                                     OutT& out) {
      impl::block_binary_transform<T, T, LeftT, RightT, OutT>(left,
                                                              right,
                                                              out,
                                                              [] __device__(auto a, auto b) {
                                                                return a * b;
                                                              });
      //__syncthreads();
    }

    template <typename T, class LeftT, class RightT, class OutT>
    __device__ inline void block_truediv(const LeftT& left,
                                         const RightT& right,
                                         OutT& out) {
      impl::block_binary_transform<T, T, LeftT, RightT, OutT>(left,
                                                              right,
                                                              out,
                                                              [] __device__(auto a, auto b) {
                                                                return a / b;
                                                              });
      //__syncthreads();
    }

    // --- Inplace binary operations --- //

    template <typename T, class LeftT, class RightT>
    __device__ inline void inplace_block_add(LeftT& left, const
                                             RightT& right) {
      impl::block_inplace_binary_transform<T>(left,
                                              right,
                                              [] __device__(auto& a, auto b) {
                                                return a + b;
                                              });
    }

    template <typename T, class LeftT, class RightT>
    __device__ inline void inplace_block_sub(LeftT& left,
                                             const RightT& right) {
      impl::block_inplace_binary_transform<T>(left,
                                              right,
                                              [] __device__(auto& a, auto b) {
                                                return a - b;
                                              });
    }

    template <typename T, class LeftT, class RightT>
    __device__ inline void inplace_block_mul(LeftT& left,
                                             const RightT& right) {
      impl::block_inplace_binary_transform<T>(left,
                                              right,
                                              [] __device__(auto& a, auto b) {
                                                return a * b;
                                              });
    }

    template <typename T, class LeftT, class RightT>
    __device__ inline void inplace_block_truediv(LeftT& left,
                                             const RightT& right) {
      using ResultT = typename op_traits<T>::truediv_type;
      impl::block_inplace_binary_transform<T>(left,
                                              right,
                                              [] __device__(auto& a, auto b) {
                                                return static_cast<T>(static_cast<ResultT>(a) /
                                                                      static_cast<ResultT>(b));
                                              });
    }

    // --- Binary operations with a scalar broadcast --- //

    template <typename T, class LeftT, class OutT>
    __device__ inline void block_scalar_add(const LeftT& left,
                                            const T& scalar_val,
                                            OutT& out) {
      using AccumT = typename op_traits<T>::sum_type;
      impl::block_binary_scalar_transform<T, LeftT, OutT>(left,
                                                          scalar_val,
                                                          out,
                                                          [] __device__ (auto a, auto b) {
                                                            return static_cast<AccumT>(a) + b;
                                                          });
      //__syncthreads();
    }

    template <typename T, class LeftT, class OutT>
    __device__ inline void block_scalar_sub(const LeftT& left,
                                            const T& scalar_val,
                                            OutT& out) {
      using DiffT = typename op_traits<T>::diff_type;
      impl::block_binary_scalar_transform<T, LeftT, OutT>(left,
                                                          scalar_val,
                                                          out,
                                                          [] __device__(auto a, auto b) {
                                                            return static_cast<DiffT>(a) - b;
                                                          });
      //__syncthreads();
    }

    template <typename T, class LeftT, class OutT>
    __device__ inline void block_scalar_mul(const LeftT& left,
                                            const T& scalar_val,
                                            OutT& out) {
      impl::block_binary_scalar_transform<T, LeftT, OutT>(left,
                                                          scalar_val,
                                                          out,
                                                          [] __device__(auto a, auto b) {
                                                            return a * b;
                                                          });
      //__syncthreads();
    }

    template <typename T, class LeftT, class OutT>
    __device__ inline void block_scalar_truediv(const LeftT& left,
                                                const T& scalar_val,
                                                OutT& out) {
      using ResultT = typename op_traits<T>::truediv_type;

      auto div_op = [] __device__(auto a, auto b) {
        using std::isfinite;
        bool is_finite { false };
        if (b == T(0)) {
          if constexpr (requires { a.real(); }) {
            is_finite = isfinite(a.real()) && isfinite(a.imag());
          } else {
            is_finite = isfinite(a);
          }
          return is_finite ? std::nan("") : static_cast<ResultT>(a);
        } else {
          return static_cast<ResultT>(a) / static_cast<ResultT>(b);
        }
      };
      impl::block_binary_scalar_transform<T, LeftT, OutT>(left,
                                                          scalar_val,
                                                          out,
                                                          div_op);
      //__syncthreads();
    }

    // --- Inplace binary operations with a scalar broadcast --- //

    template <typename T, class LeftT>
    __device__ inline void inplace_block_scalar_add(LeftT& left,
                                                    const T& scalar_val) {
      impl::block_inplace_binary_scalar_transform<T>(left,
                                                     scalar_val,
                                                     [] __device__(auto& a, auto b) {
                                                       return a + b;
                                                     });
    }

    template <typename T, class LeftT>
    __device__ inline void inplace_block_scalar_sub(LeftT& left,
                                                    const T& scalar_val) {
      using DiffT = typename op_traits<T>::diff_type;
      impl::block_inplace_binary_scalar_transform<T>(left,
                                                     scalar_val,
                                                     [] __device__(auto& a, auto b) {
                                                       return static_cast<T>(static_cast<DiffT>(a) - b);
                                                     });
    }

    template <typename T, class LeftT>
    __device__ inline void inplace_block_scalar_mul(LeftT& left,
                                                    const T& scalar_val) {
      impl::block_inplace_binary_scalar_transform<T>(left,
                                                     scalar_val,
                                                     [] __device__(auto& a, auto b) {
                                                       return a * b;
                                                     });
    }

    template <typename T, class LeftT>
    __device__ inline void inplace_block_scalar_truediv(LeftT& left,
                                                        const T& scalar_val) {
      using ResultT = typename op_traits<T>::truediv_type;

      auto div_op = [] __device__(auto a, auto b) {
        return static_cast<T>(static_cast<ResultT>(a) / static_cast<ResultT>(b));
      };

      impl::block_inplace_binary_scalar_transform<T>(left, scalar_val, div_op);
    }

    // --- Logical and boolean operators --- //

    template <typename T, class LeftT, class RightT, class OutT>
    __device__ inline void block_equal(const LeftT& left,
                                       const RightT& right,
                                       OutT& out) {

      impl::block_binary_transform<T, bool, LeftT, RightT, OutT>(left,
                                                                 right,
                                                                 out,
                                                                 [] __device__ (auto a, auto b) {
                                                                   return a == b;
                                                                 });
      //__syncthreads();
    }

    template <typename T, class LeftT, class RightT, class OutT>
    __device__ inline void block_not_equal(const LeftT& left,
                                           const RightT& right,
                                           OutT& out) {

      impl::block_binary_transform<T, bool, LeftT, RightT, OutT>(left,
                                                                 right,
                                                                 out,
                                                                 [] __device__ (auto a, auto b) {
                                                                   return a != b;
                                                                 });
      //__syncthreads();
    }

    template <typename T, class LeftT, class RightT, class OutT>
    __device__ inline void block_less_than(const LeftT& left,
                                           const RightT& right,
                                           OutT& out) {
      impl::block_binary_transform<T, bool, LeftT, RightT, OutT>(left,
                                                                 right,
                                                                 out,
                                                                 [] __device__(auto a, auto b) {
                                                                   return a < b;
                                                                 });
      //__syncthreads();
    }

    template <typename T, class LeftT, class RightT, class OutT>
    __device__ inline void block_greater_than(const LeftT& left,
                                              const RightT& right,
                                              OutT& out) {
      impl::block_binary_transform<T, bool, LeftT, RightT, OutT>(left,
                                                                 right,
                                                                 out,
                                                                 [] __device__(auto a, auto b) {
                                                                   return a > b;
                                                                 });
      //__syncthreads();
    }
  } // namespace device
} // namespace ncarray

#endif // NCARRAY_DEVICE_ELEMENTWISE_HH
