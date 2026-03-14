#ifndef NCARRAY_ARRAY_TRAITS_HH
#define NCARRAY_ARRAY_TRAITS_HH

#include "dtype.hh"

// How - Requirements and concepts

#include <concepts>
#include <cstdint>
#include <limits>
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

namespace ncarray {
  /**
   * The following concepts enforce the interface for the generic algorithms.
   */
  template <typename T>
  concept Shaped = requires(const T arr) {
    { arr.ndim() } -> std::convertible_to<ssize_t>;
    { arr.shape() } -> std::convertible_to<const ssize_t*>;
  };

  template <typename T>
  concept Strided = requires(const T arr) {
    { arr.strides() } -> std::convertible_to<const ssize_t*>;
  };

  template <typename T>
  concept HasData = requires(const T arr) {
    { arr.data() } -> std::convertible_to<const void*>;
  };

  template <typename T>
  concept HasDType = requires(const T arr) {
    { arr.dtype() } -> std::same_as<ncarray::DType>;
  };

  template <typename T>
  concept ArrayLike = Shaped<T> && Strided<T> && HasData<T> && HasDType<T>;

  // NCArray* arrays have pointer axes (suboffsets may too) - so additional
  // optional contract for that
  template <typename T>
  constexpr bool get_is_pointer_axis(const T& arr, ssize_t axis) {
    if constexpr (requires { arr.is_pointer_axis(axis); }) {
      return arr.is_pointer_axis(axis);
    } else {
      // Non-NCArray* or suboffsets arrays never have pointer axis
      return false;
    }
  }

  // Likewise check if there are offsets
  template <typename T>
  constexpr ssize_t* if_has_get_offsets(const T& arr) {
    if constexpr (requires { arr.ofsets() -> std::convertible_to<const ssize_t*>; }) {
      return arr.offsets();
    } else {
      return nullptr;
    }
  }

  /**
   * Numeric traits for specifying types during arithmetic as well as providing
   * operations for types that don't have them, or overriding their default
   * behaviour.
   *
   * Examples include type promotion for small integers (e.g. std::uint8_t) so
   * accumulation operations don't overflow, or providing comparisons for complex
   * number types.
   */
  template <typename T> struct BaseOpTraits {
    using sum_type = T;
    using truediv_type = double;

    // Comparisons and identities -- needed especially for specializations below
    // on things like complex
    static bool greater(const T& a, const T& b) { return a > b; }
    static bool less(const T& a, const T& b) { return a < b; }
    static T lowest() { return std::numeric_limits<T>::lowest(); }
    static T max() { return std::numeric_limits<T>::max(); }
  };

  template <typename T> struct op_traits : BaseOpTraits<T> {};

  // Complex need comparison operations
  template <typename T>
  struct op_traits<std::complex<T>> : BaseOpTraits<T> {
    using sum_type = std::complex<T>;
    using truediv_type = std::complex<double>;

    static bool greater(const std::complex<T>& a, const std::complex<T>& b) {
      if (a.real() != b.real()) {
        return a.real() > b.real();
      }
      return a.imag() > b.imag();
    }

    static bool less(const std::complex<T>& a, const std::complex<T>& b) {
      if (a.real() != b.real()) {
        return a.real() < b.real();
      }
      return a.imag() < b.imag();
    }

    static std::complex<T> lowest() {
      return {std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest()};
    }

    static std::complex<T> max() {
      return {std::numeric_limits<T>::max(), std::numeric_limits<T>::max()};
    }
  };

  template <>
  struct op_traits<std::int8_t> : BaseOpTraits<std::int8_t> {
    using sum_type = int64_t;
  };

  template <>
  struct op_traits<std::uint8_t> : BaseOpTraits<std::uint8_t> {
    using sum_type = uint64_t;
  };

  template <>
  struct op_traits<std::int16_t> : BaseOpTraits<std::int16_t> {
    using sum_type = int64_t;
  };

  template <>
  struct op_traits<std::uint16_t> : BaseOpTraits<std::uint16_t> {
    using sum_type = uint64_t;
  };

} // namespace ncarray

#endif // NCARRAY_ARRAY_TRAITS_HH
