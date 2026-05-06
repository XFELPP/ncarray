#ifndef NCARRAY_BUILD_MACRO_HH
#define NCARRAY_BUILD_MACRO_HH

#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"
#include "ncarray/engines.hh"
#include "ncarray/expression.hh"
#include "ncarray/layout.hh"
#include "ncarray/vmexpression.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cstdint>
#include <complex>
#include <vector>

#define BASE_GENERAL_LIST(ACTION, L, S)                                                            \
                                                                                                   \
                               /* Constructors */                                                  \
                                                                                                   \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>::ArrayImpl(const std::vector<ssize_t>&,        \
                                                               ncarray::DType);                    \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>::ArrayImpl(const ncarray::Metadata&,           \
                                                               ncarray::DType);                    \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>::ArrayImpl(ssize_t, const ssize_t*,            \
                                                               ncarray::DType);                    \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>::ArrayImpl(                                    \
      const std::vector<void*>&, const std::vector<ssize_t>&, const std::vector<ssize_t>&,         \
      ncarray::DType, ncarray::Metadata::value_type, bool);                                        \
                                                                                                   \
                        /* Assignment/Materialization */                                           \
                                                                                                   \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator=<ncarray::ExprMVNode<                       \
      typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>>(                              \
      const ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>&);   \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator=<ncarray::DynamicExprMVNode<                \
      typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>>(                              \
      const ncarray::DynamicExprMVNode<                                                            \
          typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>&);                         \
                                                                                                   \
                        /* Copy, assignment, reshaping */                                          \
                                                                                                   \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::assign(                                  \
     const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);                                           \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::fill(ncarray::Scalar);                   \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::astype(ncarray::DType&) const;                       \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into(void*) const;                  \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                                     \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_as_shape(const ssize_t*, ssize_t) const;        \
                                                                                                   \
                        /* Repr, small utilities */                                                \
                                                                                                   \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::get_scalar(void*) const;      \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<char>(         \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<std::uint8_t>( \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<std::uint16_t>(            \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<std::uint32_t>(            \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<std::uint64_t>(            \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<std::int8_t>(  \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<std::int16_t>( \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<std::int32_t>( \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<std::int64_t>( \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<float>(        \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<double>(       \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<long double>(  \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<bool>(         \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<std::complex<float>>(      \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<std::complex<double>>(     \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<               \
      std::complex<long double>>(std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;     \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<ncarray::Float2>(          \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<ncarray::Float3>(          \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<ncarray::Float4>(          \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<ncarray::Double2>(         \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<ncarray::Double3>(         \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;                                \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::repr_recursive_dispatched<ncarray::Double4>(         \
      std::ostringstream&, void*, ssize_t, ssize_t, ssize_t) const;



#define BASE_ARITHMETIC_LIST(ACTION, L, S)                                                         \
                                                                                                   \
                             /* Binary Arithmetic */                                               \
                                                                                                   \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>         \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator+(const ncarray::ArrayImpl<ncarray::L,   \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator+(const ncarray::Scalar&) const;         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>         \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator-(const ncarray::ArrayImpl<ncarray::L,   \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator-(const ncarray::Scalar&) const;         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>         \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator*(const ncarray::ArrayImpl<ncarray::L,   \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator*(const ncarray::Scalar&) const;         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>         \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator/(const ncarray::ArrayImpl<ncarray::L,   \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator/(const ncarray::Scalar&) const;         \
                                                                                                   \
                        /* Binary Inplace Arithmetic */                                            \
                                                                                                   \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator+=(const ncarray::ExprMVNode<            \
      typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>&);                             \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator+=(const ncarray::Scalar&);              \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator-=(const ncarray::ExprMVNode<            \
      typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>&);                             \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator-=(const ncarray::Scalar&);              \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator*=(const ncarray::ExprMVNode<            \
      typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>&);                             \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator*=(const ncarray::Scalar&);              \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator/=(const ncarray::ExprMVNode<            \
      typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>&);                             \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator/=(const ncarray::Scalar&);              \

#define BASE_COMPARISON_LIST(ACTION, L, S)                                                         \
                                                                                                   \
                              /* Comparisons */                                                    \
                                                                                                   \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator==(const ncarray::ArrayImpl<ncarray::L,  \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator==(const ncarray::Scalar&) const;        \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator!=(const ncarray::ArrayImpl<ncarray::L,  \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator!=(const ncarray::Scalar&) const;        \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator<(const ncarray::ArrayImpl<ncarray::L,   \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator<(const ncarray::Scalar&) const;         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator<=(const ncarray::ArrayImpl<ncarray::L,  \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator<=(const ncarray::Scalar&) const;        \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator>(const ncarray::ArrayImpl<ncarray::L,   \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator>(const ncarray::Scalar&) const;         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator>=(const ncarray::ArrayImpl<ncarray::L,  \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator>=(const ncarray::Scalar&) const;        \

#define BASE_LOGICAL_LIST(ACTION, L, S)                                                            \
                                                                                                   \
                        /* Logical Operations */                                                   \
                                                                                                   \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator&&(const ncarray::ArrayImpl<ncarray::L,  \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator&&(const ncarray::Scalar&) const;        \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator||(const ncarray::ArrayImpl<ncarray::L,  \
      ncarray::S>&) const;                                                                         \
  ACTION ncarray::ExprMVNode<typename ncarray::ArrayImpl<ncarray::L,ncarray::S>::MemType>          \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator||(const ncarray::Scalar&) const;        \
                                                                                                   \
                        /* Logical Inplace Operations */                                           \
                                                                                                   \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator&=(const ncarray::ExprMVNode<            \
      typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>&);                             \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator&=(const ncarray::Scalar&);              \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator|=(const ncarray::ExprMVNode<            \
      typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::MemType>&);                             \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator|=(const ncarray::Scalar&);              \


#define BASE_REDUCTIONS_LIST(ACTION, L, S)                                                         \
                                                                                                   \
                        /* Full Reductions To Scalar */                                            \
                                                                                                   \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::sum() const;                  \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::max() const;                  \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::argmax() const;               \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::min() const;                  \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::argmin() const;               \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::mean() const;                 \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::var(ssize_t) const;           \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::std(ssize_t) const;           \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::all() const;                  \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::any() const;                  \
                                                                                                   \
                        /* Axis Aware Reductions */                                                \
                                                                                                   \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::sum(const std::vector<ssize_t>&) const;          \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::max(const std::vector<ssize_t>&) const;          \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::argmax(const std::vector<ssize_t>&) const;       \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::min(const std::vector<ssize_t>&) const;          \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::argmin(const std::vector<ssize_t>&) const;       \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::mean(const std::vector<ssize_t>&) const;         \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::var(const std::vector<ssize_t>&, ssize_t) const; \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::std(const std::vector<ssize_t>&, ssize_t) const; \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::all(const std::vector<ssize_t>&) const;          \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
      ncarray::ArrayImpl<ncarray::L, ncarray::S>::any(const std::vector<ssize_t>&) const;


#define INSTANTIATE_NC_BASE_GENERAL(L, S) BASE_GENERAL_LIST(template, L, S)
#define EXTERN_NC_BASE_GENERAL(L, S) BASE_GENERAL_LIST(extern template, L, S)

#define INSTANTIATE_NC_BASE_ARITHMETIC(L, S) BASE_ARITHMETIC_LIST(template, L, S)
#define EXTERN_NC_BASE_ARITHMETIC(L, S) BASE_ARITHMETIC_LIST(extern template, L, S)

#define INSTANTIATE_NC_BASE_COMPARISON(L, S) BASE_COMPARISON_LIST(template, L, S)
#define EXTERN_NC_BASE_COMPARISON(L, S) BASE_COMPARISON_LIST(extern template, L, S)

#define INSTANTIATE_NC_BASE_LOGICAL(L, S) BASE_LOGICAL_LIST(template, L, S)
#define EXTERN_NC_BASE_LOGICAL(L, S) BASE_LOGICAL_LIST(extern template, L, S)

#define INSTANTIATE_NC_BASE_REDUCTIONS(L, S) BASE_REDUCTIONS_LIST(template, L, S)
#define EXTERN_NC_BASE_REDUCTIONS(L, S) BASE_REDUCTIONS_LIST(extern template, L, S)


#define EXTERN_NC_BASE_OPS(L, S)  \
  EXTERN_NC_BASE_GENERAL(L, S)    \
  EXTERN_NC_BASE_ARITHMETIC(L, S) \
  EXTERN_NC_BASE_COMPARISON(L, S) \
  EXTERN_NC_BASE_LOGICAL(L, S)    \
  EXTERN_NC_BASE_REDUCTIONS(L, S)


#define INSTANTIATE_NC_BASE_OPS(L, S)  \
  INSTANTIATE_NC_BASE_GENERAL(L, S)    \
  INSTANTIATE_NC_BASE_ARITHMETIC(L, S) \
  INSTANTIATE_NC_BASE_COMPARISON(L, S) \
  INSTANTIATE_NC_BASE_LOGICAL(L, S)    \
  INSTANTIATE_NC_BASE_REDUCTIONS(L, S)


#define INSTANTIATE_HOST_VM_REDUCE(T, L, Trait)                                                    \
  template void ncarray::HostEngine::execute_reduce_axes<T,                                        \
      ncarray::Trait,                                                                              \
      ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>,                                         \
      ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>>(                                        \
          const ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>&,                              \
          const ncarray::ReductionParams&,                                                         \
          ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>&);

#define EXTERN_HOST_VM_REDUCE(T, L, Trait)                                                         \
  extern template void ncarray::HostEngine::execute_reduce_axes<T,                                 \
      ncarray::Trait,                                                                              \
      ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>,                                         \
      ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>>(                                        \
          const ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>&,                              \
          const ncarray::ReductionParams&,                                                         \
          ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>&);

/* NOT CURRENTLY DEFINED YET!
#define INSTANTIATE_HOST_VM_FULL_REDUCE(T, L, Trait)                                               \
  template ncarray::Scalar ncarray::HostEngine::execute_full_reduce<T,                             \
      ncarray::Trait,                                                                              \
      ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>>(                                        \
          const ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>&);

#define EXTERN_HOST_VM_FULL_REDUCE(T, L, Trait)                                                    \
  extern template ncarray::Scalar ncarray::HostEngine::execute_full_reduce<T,                      \
      ncarray::Trait,                                                                              \
      ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>>(                                        \
          const ncarray::ArrayImpl<ncarray::L, ncarray::ViewPolicy>&);
*/

#define INSTANTIATE_HOST_VM_OPS(T, L)                                                              \
  template void ncarray::HostEngine::execute_binary_expression<                                    \
      T, ncarray::ExprMVNode<ncarray::HostTag>,                                                    \
      ncarray::ArrayImpl<ncarray::L, ncarray::OwnerPolicy>>(                                       \
      const ncarray::ExprMVNode<ncarray::HostTag>&,                                                \
      ncarray::ArrayImpl<ncarray::L, ncarray::OwnerPolicy>&);


#define EXTERN_HOST_VM_OPS(T, L)                                                                   \
  extern template void ncarray::HostEngine::execute_binary_expression<                             \
      T, ncarray::ExprMVNode<ncarray::HostTag>,                                                    \
      ncarray::ArrayImpl<ncarray::L, ncarray::OwnerPolicy>>(                                       \
      const ncarray::ExprMVNode<ncarray::HostTag>&,                                                \
      ncarray::ArrayImpl<ncarray::L, ncarray::OwnerPolicy>&);                                      \
  EXTERN_HOST_VM_REDUCE(T, L, SumTraits)                                                           \
  EXTERN_HOST_VM_REDUCE(T, L, MaxTraits)                                                           \
  EXTERN_HOST_VM_REDUCE(T, L, ArgmaxTraits)                                                        \
  EXTERN_HOST_VM_REDUCE(T, L, MinTraits)                                                           \
  EXTERN_HOST_VM_REDUCE(T, L, ArgminTraits)                                                        \
  EXTERN_HOST_VM_REDUCE(T, L, MeanTraits)                                                          \
  EXTERN_HOST_VM_REDUCE(T, L, VarTraits)                                                           \
  EXTERN_HOST_VM_REDUCE(T, L, StdTraits)                                                           \
  EXTERN_HOST_VM_REDUCE(T, L, AllTraits)                                                           \
  EXTERN_HOST_VM_REDUCE(T, L, AnyTraits)

/* NOT CURRENTLY DEFINED YET!
  EXTERN_HOST_VM_FULL_REDUCE(T, L, SumTraits)                                                      \
  EXTERN_HOST_VM_FULL_REDUCE(T, L, MaxTraits)                                                      \
  EXTERN_HOST_VM_FULL_REDUCE(T, L, ArgmaxTraits)                                                   \
  EXTERN_HOST_VM_FULL_REDUCE(T, L, MinTraits)                                                      \
  EXTERN_HOST_VM_FULL_REDUCE(T, L, ArgminTraits)                                                   \
  EXTERN_HOST_VM_FULL_REDUCE(T, L, MeanTraits)                                                     \
  EXTERN_HOST_VM_FULL_REDUCE(T, L, VarTraits)                                                      \
  EXTERN_HOST_VM_FULL_REDUCE(T, L, StdTraits)                                                      \
  EXTERN_HOST_VM_FULL_REDUCE(T, L, AllTraits)                                                      \
  EXTERN_HOST_VM_FULL_REDUCE(T, L, AnyTraits)
*/

#define INSTANTIATE_HOST_REDUCTIONS(T, L)                                                          \
  INSTANTIATE_HOST_VM_REDUCE(T, L, SumTraits)                                                      \
  INSTANTIATE_HOST_VM_REDUCE(T, L, MaxTraits)                                                      \
  INSTANTIATE_HOST_VM_REDUCE(T, L, ArgmaxTraits)                                                   \
  INSTANTIATE_HOST_VM_REDUCE(T, L, MinTraits)                                                      \
  INSTANTIATE_HOST_VM_REDUCE(T, L, ArgminTraits)                                                   \
  INSTANTIATE_HOST_VM_REDUCE(T, L, MeanTraits)                                                     \
  INSTANTIATE_HOST_VM_REDUCE(T, L, VarTraits)                                                      \
  INSTANTIATE_HOST_VM_REDUCE(T, L, StdTraits)                                                      \
  INSTANTIATE_HOST_VM_REDUCE(T, L, AllTraits)                                                      \
  INSTANTIATE_HOST_VM_REDUCE(T, L, AnyTraits)

/* NOT CURRENTLY DEFINED YET!
  INSTANTIATE_HOST_VM_FULL_REDUCE(T, L, SumTraits)                                                 \
  INSTANTIATE_HOST_VM_FULL_REDUCE(T, L, MaxTraits)                                                 \
  INSTANTIATE_HOST_VM_FULL_REDUCE(T, L, ArgmaxTraits)                                              \
  INSTANTIATE_HOST_VM_FULL_REDUCE(T, L, MinTraits)                                                 \
  INSTANTIATE_HOST_VM_FULL_REDUCE(T, L, ArgminTraits)                                              \
  INSTANTIATE_HOST_VM_FULL_REDUCE(T, L, MeanTraits)                                                \
  INSTANTIATE_HOST_VM_FULL_REDUCE(T, L, VarTraits)                                                 \
  INSTANTIATE_HOST_VM_FULL_REDUCE(T, L, StdTraits)                                                 \
  INSTANTIATE_HOST_VM_FULL_REDUCE(T, L, AllTraits)                                                 \
  INSTANTIATE_HOST_VM_FULL_REDUCE(T, L, AnyTraits)
*/

#ifdef __CUDACC__
// --- DEVICE MACROS (CUDA-Guarded) ---

#define INSTANTIATE_DEV_VM_REDUCE(T, L, Trait)                                                     \
  template __global__ void axes_reduce_kernel<true, T,                                             \
      typename ncarray::Reducer<T, Trait>::AccumT,                                                 \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::Reducer<T, Trait>>(                                                                 \
          ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                  \
          typename ncarray::Reducer<T, Trait>::AccumT*,                                            \
          ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                  \
          ncarray::ReductionParams,                                                                \
          ncarray::Reducer<T, Trait>);                                                             \
  template __global__ void axes_reduce_kernel<false, T,                                            \
      typename ncarray::Reducer<T, Trait>::AccumT,                                                 \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::Reducer<T, Trait>>(                                                                 \
          ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                  \
          typename ncarray::Reducer<T, Trait>::AccumT*,                                            \
          ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                  \
          ncarray::ReductionParams,                                                                \
          ncarray::Reducer<T, Trait>);                                                             \
  template void ncarray::GPUEngine::execute_reduce_axes<T,                                         \
      ncarray::Trait,                                                                              \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>>(                                     \
          const ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>&,                           \
          const ncarray::ReductionParams&,                                                         \
          ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>&);

#define EXTERN_DEV_VM_REDUCE(T, L, Trait)                                                          \
  extern template __global__ void axes_reduce_kernel<true, T,                                      \
      typename ncarray::Reducer<T, Trait>::AccumT,                                                 \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::Reducer<T, Trait>>(                                                                 \
          ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                  \
          typename ncarray::Reducer<T, Trait>::AccumT*,                                            \
          ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                  \
          ncarray::ReductionParams,                                                                \
          ncarray::Reducer<T, Trait>);                                                             \
  extern template __global__ void axes_reduce_kernel<false, T,                                     \
      typename ncarray::Reducer<T, Trait>::AccumT,                                                 \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::Reducer<T, Trait>>(                                                                 \
          ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                  \
          typename ncarray::Reducer<T, Trait>::AccumT*,                                            \
          ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                  \
          ncarray::ReductionParams,                                                                \
          ncarray::Reducer<T, Trait>);                                                             \
  extern template void ncarray::GPUEngine::execute_reduce_axes<T,                                  \
      ncarray::Trait,                                                                              \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>>(                                     \
          const ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>&,                           \
          const ncarray::ReductionParams&,                                                         \
          ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>&);

#define INSTANTIATE_DEV_VM_FULL_REDUCE(T, L, Trait)                                                \
  template __global__ void reduce_kernel<256, T,                                                   \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::Reducer<T, Trait>>(                                                                 \
          const ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                            \
          ncarray::Reducer<T, Trait>,                                                              \
          typename ncarray::Reducer<T, Trait>::AccumT*);                                           \
  template ncarray::Scalar ncarray::GPUEngine::execute_full_reduce<T,                              \
      ncarray::Trait,                                                                              \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>>(                                     \
          const ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>&);

#define EXTERN_DEV_VM_FULL_REDUCE(T, L, Trait)                                                     \
  extern template __global__ void reduce_kernel<256, T,                                            \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                                      \
      ncarray::Reducer<T, Trait>>(                                                                 \
          const ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>,                            \
          ncarray::Reducer<T, Trait>,                                                              \
          typename ncarray::Reducer<T, Trait>::AccumT*);                                           \
  extern template ncarray::Scalar ncarray::GPUEngine::execute_full_reduce<T,                       \
      ncarray::Trait,                                                                              \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>>(                                     \
          const ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>&);

#define INSTANTIATE_DEV_VM_OPS(T, L)                                                               \
  template __global__ void execute_expression_kernel<T,                                            \
    ncarray::DynamicExprMVNode<ncarray::DevTag>,                                                   \
    ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>>(                                       \
    ncarray::DynamicExprMVNode<ncarray::DevTag>,                                                   \
    ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>);                                       \
  template void ncarray::GPUEngine::execute_binary_expression<                                     \
      T, ncarray::DynamicExprMVNode<ncarray::DevTag>,                                              \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>>(                                     \
      const ncarray::DynamicExprMVNode<ncarray::DevTag>&,                                          \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>&);

#define INSTANTIATE_DEV_REDUCTIONS(T, L)                                                           \
  INSTANTIATE_DEV_VM_REDUCE(T, L, SumTraits)                                                       \
  INSTANTIATE_DEV_VM_REDUCE(T, L, MaxTraits)                                                       \
  INSTANTIATE_DEV_VM_REDUCE(T, L, ArgmaxTraits)                                                    \
  INSTANTIATE_DEV_VM_REDUCE(T, L, MinTraits)                                                       \
  INSTANTIATE_DEV_VM_REDUCE(T, L, ArgminTraits)                                                    \
  INSTANTIATE_DEV_VM_REDUCE(T, L, MeanTraits)                                                      \
  INSTANTIATE_DEV_VM_REDUCE(T, L, VarTraits)                                                       \
  INSTANTIATE_DEV_VM_REDUCE(T, L, StdTraits)                                                       \
  INSTANTIATE_DEV_VM_REDUCE(T, L, AllTraits)                                                       \
  INSTANTIATE_DEV_VM_REDUCE(T, L, AnyTraits)                                                       \
  INSTANTIATE_DEV_VM_FULL_REDUCE(T, L, SumTraits)                                                  \
  INSTANTIATE_DEV_VM_FULL_REDUCE(T, L, MaxTraits)                                                  \
  INSTANTIATE_DEV_VM_FULL_REDUCE(T, L, ArgmaxTraits)                                               \
  INSTANTIATE_DEV_VM_FULL_REDUCE(T, L, MinTraits)                                                  \
  INSTANTIATE_DEV_VM_FULL_REDUCE(T, L, ArgminTraits)                                               \
  INSTANTIATE_DEV_VM_FULL_REDUCE(T, L, MeanTraits)                                                 \
  INSTANTIATE_DEV_VM_FULL_REDUCE(T, L, VarTraits)                                                  \
  INSTANTIATE_DEV_VM_FULL_REDUCE(T, L, StdTraits)                                                  \
  INSTANTIATE_DEV_VM_FULL_REDUCE(T, L, AllTraits)                                                  \
  INSTANTIATE_DEV_VM_FULL_REDUCE(T, L, AnyTraits)

#define EXTERN_DEV_VM_OPS(T, L)                                                                    \
  extern template __global__ void execute_expression_kernel<T,                                     \
      ncarray::DynamicExprMVNode<ncarray::DevTag>,                                                 \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>>(                                     \
      ncarray::DynamicExprMVNode<ncarray::DevTag>,                                                 \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>);                                     \
  extern template void ncarray::GPUEngine::execute_binary_expression<T,                            \
      ncarray::DynamicExprMVNode<ncarray::DevTag>,                                                 \
      ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>>(                                     \
          const ncarray::DynamicExprMVNode<ncarray::DevTag>&,                                      \
          ncarray::ArrayImpl<ncarray::L, ncarray::DevViewPolicy>&);                                \
  EXTERN_DEV_VM_REDUCE(T, L, SumTraits)                                                            \
  EXTERN_DEV_VM_REDUCE(T, L, MaxTraits)                                                            \
  EXTERN_DEV_VM_REDUCE(T, L, ArgmaxTraits)                                                         \
  EXTERN_DEV_VM_REDUCE(T, L, MinTraits)                                                            \
  EXTERN_DEV_VM_REDUCE(T, L, ArgminTraits)                                                         \
  EXTERN_DEV_VM_REDUCE(T, L, MeanTraits)                                                           \
  EXTERN_DEV_VM_REDUCE(T, L, VarTraits)                                                            \
  EXTERN_DEV_VM_REDUCE(T, L, StdTraits)                                                            \
  EXTERN_DEV_VM_REDUCE(T, L, AllTraits)                                                            \
  EXTERN_DEV_VM_REDUCE(T, L, AnyTraits)                                                            \
  EXTERN_DEV_VM_FULL_REDUCE(T, L, SumTraits)                                                       \
  EXTERN_DEV_VM_FULL_REDUCE(T, L, MaxTraits)                                                       \
  EXTERN_DEV_VM_FULL_REDUCE(T, L, ArgmaxTraits)                                                    \
  EXTERN_DEV_VM_FULL_REDUCE(T, L, MinTraits)                                                       \
  EXTERN_DEV_VM_FULL_REDUCE(T, L, ArgminTraits)                                                    \
  EXTERN_DEV_VM_FULL_REDUCE(T, L, MeanTraits)                                                      \
  EXTERN_DEV_VM_FULL_REDUCE(T, L, VarTraits)                                                       \
  EXTERN_DEV_VM_FULL_REDUCE(T, L, StdTraits)                                                       \
  EXTERN_DEV_VM_FULL_REDUCE(T, L, AllTraits)                                                       \
  EXTERN_DEV_VM_FULL_REDUCE(T, L, AnyTraits)


using CanonicalCoords = ncarray::StaticCoords<NCARRAY_MAX_NDIM, ssize_t>;

#else
  #define INSTANTIATE_DEV_VM_OPS(T, L)
  #define EXTERN_DEV_VM_OPS(T, L)
#endif


// --- CROSS-OPERATIONS LIST ---
#define CROSS_OPS_LIST(ACTION, L1, S1, L2, S2)                                                     \
  ACTION void ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::assign(                                \
      const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&);

#define EXTERN_NC_CROSS_OPS(L1, S1, L2, S2) CROSS_OPS_LIST(extern template, L1, S1, L2, S2)
#define INSTANTIATE_NC_CROSS_OPS(L1, S1, L2, S2) CROSS_OPS_LIST(template, L1, S1, L2, S2)
#endif
