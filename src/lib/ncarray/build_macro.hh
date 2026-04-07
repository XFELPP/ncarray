#ifndef NCARRAY_BUILD_MACRO_HH
#define NCARRAY_BUILD_MACRO_HH

#include "ncarray/array_traits.hh"
#include "ncarray/custom_types.hh"
#include "ncarray/dtype.hh"
#include "ncarray/layout.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cstdint>
#include <complex>
#include <vector>

namespace ncarray {
  template <ArrayLike A> Scalar sum(const A& arr);
  template <ArrayLike A> Scalar max(const A& arr);
  template <ArrayLike A> Scalar argmax(const A& arr);
  template <ArrayLike A> Scalar min(const A& arr);
  template <ArrayLike A> Scalar argmin(const A& arr);
  template <ArrayLike A> Scalar mean(const A& arr);
  template <ArrayLike A> Scalar var(const A& arr, ssize_t ddof);
  template <ArrayLike A> Scalar std(const A& arr, ssize_t ddof);
  template <ArrayLike A> Scalar all(const A& arr);
  template <ArrayLike A> Scalar any(const A& arr);
} // namespace ncarray

#define BASE_OPS_LIST(ACTION, L, S)                                                                \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>::ArrayImpl(const std::vector<ssize_t>&,        \
                                                               ncarray::DType);                    \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>::ArrayImpl(const ncarray::Metadata&,           \
                                                               ncarray::DType);                    \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>::ArrayImpl(ssize_t, const ssize_t*,            \
                                                               ncarray::DType);                    \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>::ArrayImpl(                                    \
      const std::vector<void*>&, const std::vector<ssize_t>&, const std::vector<ssize_t>&,         \
      ncarray::DType, ncarray::Metadata::value_type, bool);                                        \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator+(                                           \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator-(                                           \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator*(                                           \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator/(                                           \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator+(const ncarray::Scalar&) const;             \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator-(const ncarray::Scalar&) const;             \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator*(const ncarray::Scalar&) const;             \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator/(const ncarray::Scalar&) const;             \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator+=(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);                                          \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator-=(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);                                          \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator*=(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);                                          \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator/=(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);                                          \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator+=(const ncarray::Scalar&);                  \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator-=(const ncarray::Scalar&);                  \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator*=(const ncarray::Scalar&);                  \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator/=(const ncarray::Scalar&);                  \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator==(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator!=(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator<(                                           \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator<=(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator>(                                           \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator>=(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator==(const ncarray::Scalar&) const;            \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator!=(const ncarray::Scalar&) const;            \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator<(const ncarray::Scalar&) const;             \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator<=(const ncarray::Scalar&) const;            \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator>(const ncarray::Scalar&) const;             \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator>=(const ncarray::Scalar&) const;            \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator!() const;                                   \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator&&(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator||(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&) const;                                    \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator&=(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);                                          \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator|=(                                          \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);                                          \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator&&(const ncarray::Scalar&) const;            \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator||(const ncarray::Scalar&) const;            \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator&=(const ncarray::Scalar&);                  \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>&                                               \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::operator|=(const ncarray::Scalar&);                  \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::sum() const;                  \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::max() const;                  \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::argmax() const;               \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::min() const;                  \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::argmin() const;               \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::mean() const;                 \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::var(ssize_t ddof) const;      \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::std(ssize_t ddof) const;      \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::all() const;                  \
  ACTION ncarray::Scalar ncarray::ArrayImpl<ncarray::L, ncarray::S>::any() const;                  \
  ACTION ncarray::Scalar ncarray::sum(const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);          \
  ACTION ncarray::Scalar ncarray::max(const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);          \
  ACTION ncarray::Scalar ncarray::argmax(const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);       \
  ACTION ncarray::Scalar ncarray::min(const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);          \
  ACTION ncarray::Scalar ncarray::argmin(const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);       \
  ACTION ncarray::Scalar ncarray::mean(const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);         \
  ACTION ncarray::Scalar ncarray::var(const ncarray::ArrayImpl<ncarray::L, ncarray::S>&, ssize_t); \
  ACTION ncarray::Scalar ncarray::std(const ncarray::ArrayImpl<ncarray::L, ncarray::S>&, ssize_t); \
  ACTION ncarray::Scalar ncarray::all(const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);          \
  ACTION ncarray::Scalar ncarray::any(const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);          \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::assign(                                  \
      const ncarray::ArrayImpl<ncarray::L, ncarray::S>&);                                          \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::fill(ncarray::Scalar);                   \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::astype(ncarray::DType&) const;                       \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::OwnerType                            \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::to_contiguous() const;                               \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into(void*) const;                  \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::Iterator                             \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::begin();                                             \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::Iterator                             \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::end();                                               \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::ConstIterator                        \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::begin() const;                                       \
  ACTION typename ncarray::ArrayImpl<ncarray::L, ncarray::S>::ConstIterator                        \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::end() const;                                         \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>::ViewType                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::view() const;                                        \
  ACTION ncarray::ArrayImpl<ncarray::L, ncarray::S>::ViewType                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::squeeze() const;                                     \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<char>(                  \
      char*) const;                                                                                \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<std::uint8_t>(          \
      std::uint8_t*) const;                                                                        \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<std::uint16_t>(         \
      std::uint16_t*) const;                                                                       \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<std::uint32_t>(         \
      std::uint32_t*) const;                                                                       \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<std::uint64_t>(         \
      std::uint64_t*) const;                                                                       \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<std::int8_t>(           \
      std::int8_t*) const;                                                                         \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<std::int16_t>(          \
      std::int16_t*) const;                                                                        \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<std::int32_t>(          \
      std::int32_t*) const;                                                                        \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<std::int64_t>(          \
      std::int64_t*) const;                                                                        \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<float>(float*) const;   \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<double>(double*) const; \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<long double>(           \
      long double*) const;                                                                         \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<bool>(bool*) const;     \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<std::complex<float>>(   \
      std::complex<float>*) const;                                                                 \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<std::complex<double>>(  \
      std::complex<double>*) const;                                                                \
  ACTION void                                                                                      \
  ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<std::complex<long double>>(         \
      std::complex<long double>*) const;                                                           \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<ncarray::Float2>(       \
      ncarray::Float2*) const;                                                                     \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<ncarray::Float3>(       \
      ncarray::Float3*) const;                                                                     \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<ncarray::Float4>(       \
      ncarray::Float4*) const;                                                                     \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<ncarray::Double2>(      \
      ncarray::Double2*) const;                                                                    \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<ncarray::Double3>(      \
      ncarray::Double3*) const;                                                                    \
  ACTION void ncarray::ArrayImpl<ncarray::L, ncarray::S>::copy_into_astype<ncarray::Double4>(      \
      ncarray::Double4*) const;                                                                    \
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

// --- CROSS-OPERATIONS LIST ---
#define CROSS_OPS_LIST(ACTION, L1, S1, L2, S2)                                                     \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator+(                                     \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator-(                                     \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator*(                                     \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator/(                                     \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION ncarray::ArrayImpl<ncarray::L1, ncarray::S1>&                                             \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator+=(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&);                                    \
  ACTION ncarray::ArrayImpl<ncarray::L1, ncarray::S1>&                                             \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator-=(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&);                                    \
  ACTION ncarray::ArrayImpl<ncarray::L1, ncarray::S1>&                                             \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator*=(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&);                                    \
  ACTION ncarray::ArrayImpl<ncarray::L1, ncarray::S1>&                                             \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator/=(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&);                                    \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator==(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator!=(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator<(                                     \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator<=(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator>(                                     \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator>=(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator&&(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION typename ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::OwnerType                          \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator||(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&) const;                              \
  ACTION ncarray::ArrayImpl<ncarray::L1, ncarray::S1>&                                             \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator&=(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&);                                    \
  ACTION ncarray::ArrayImpl<ncarray::L1, ncarray::S1>&                                             \
      ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::operator|=(                                    \
          const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&);                                    \
  ACTION void ncarray::ArrayImpl<ncarray::L1, ncarray::S1>::assign(                                \
      const ncarray::ArrayImpl<ncarray::L2, ncarray::S2>&);

#define INSTANTIATE_NC_BASE_OPS(L, S) BASE_OPS_LIST(template, L, S)
#define INSTANTIATE_NC_CROSS_OPS(L1, S1, L2, S2) CROSS_OPS_LIST(template, L1, S1, L2, S2)

#define EXTERN_NC_BASE_OPS(L, S) BASE_OPS_LIST(extern template, L, S)
#define EXTERN_NC_CROSS_OPS(L1, S1, L2, S2) CROSS_OPS_LIST(extern template, L1, S1, L2, S2)

#endif
