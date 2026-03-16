#ifndef NCARRAY_ARRAY_TRAITS_HH
#define NCARRAY_ARRAY_TRAITS_HH

#include "dtype.hh"

#include <complex>
#include <concepts>
#include <cstdint>
#include <limits>
#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif
#include <vector>

namespace ncarray {
  namespace impl {
    template <typename T = void> struct default_owner;
  } // namespace impl
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

  // Mutable/writable arrays can get data that is not just const void*
  template <typename T>
  concept MutableArrayLike = ArrayLike<T> && requires(T arr) {
    { arr.data() } -> std::same_as<void*>;
  };

  // ArrayLikes that own the data should be constructable from just the shape and type
  // This indicates they can control data buffer
  template <typename T>
  concept OwningArrayLike = ArrayLike<T> && requires(std::vector<ssize_t> shape, DType dtype) {
    T(shape, dtype);
  };

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
  constexpr const ssize_t* if_has_get_offsets(const T& arr) {
    if constexpr (requires { arr.offsets(); } &&
                  std::is_convertible_v<decltype(arr.offsets()), const ssize_t*>) {
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
    using diff_type = T;
    using truediv_type = double;

    // Comparisons and identities -- needed especially for specializations below
    // on things like complex
    static bool greater(const T& a, const T& b) { return a > b; }
    static bool less(const T& a, const T& b) { return a < b; }
    static T lowest() { return std::numeric_limits<T>::lowest(); }
    static T max() { return std::numeric_limits<T>::max(); }

    template <typename To>
    static To cast(const T& val) {
      if constexpr (std::is_same_v<To, T>) {
        return val;
      } else if constexpr (requires { To().real(); }) {
        // Scalar cast to complex
        return To(static_cast<typename To::value_type>(val), 0);
      } else {
        // Scalar cast to scalar
        return static_cast<To>(val);
      }
    }
  };

  template <typename T> struct op_traits : BaseOpTraits<T> {};

  // Complex need comparison operations
  template <typename T>
  struct op_traits<std::complex<T>> : BaseOpTraits<T> {
    using sum_type = std::complex<T>;
    using diff_type = std::complex<T>;
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

    template <typename To>
    static To cast(const std::complex<T>& val) {
      if constexpr (std::is_same_v<To, std::complex<T>>) {
        return val;
      } else if constexpr (requires { To().real(); }) {
        // Complex to complex cast
        return To(static_cast<typename To::value_type>(val.real()),
                  static_cast<typename To::value_type>(val.imag()));
      } else {
        // Complex to scalar cast (narrowed to real)
        return static_cast<To>(val.real());
      }
    }
  };

  template <>
  struct op_traits<std::int8_t> : BaseOpTraits<std::int8_t> {
    using sum_type = int64_t;
    using diff_type = int64_t;
  };

  template <>
  struct op_traits<std::uint8_t> : BaseOpTraits<std::uint8_t> {
    using sum_type = uint64_t;
    // NOTE: Unsigned types promot to SIGNED for subtraction!
    using diff_type = int64_t;
  };

  template <>
  struct op_traits<std::int16_t> : BaseOpTraits<std::int16_t> {
    using sum_type = int64_t;
    using diff_type = int64_t;
  };

  template <>
  struct op_traits<std::uint16_t> : BaseOpTraits<std::uint16_t> {
    using sum_type = uint64_t;
    // NOTE: Unsigned types promot to SIGNED for subtraction!
    using diff_type = int64_t;
  };

} // namespace ncarray

#endif // NCARRAY_ARRAY_TRAITS_HH
