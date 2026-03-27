#ifndef NCARRAY_STORAGE_HH
#define NCARRAY_STORAGE_HH

#include "ncarray/dtype.hh"

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <cstdint>
#include <memory>

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

#ifndef NCARRAY_MAX_NDIM
#define NCARRAY_MAX_NDIM 10
#endif

namespace ncarray {
  /**
   * The StoragePolicy specifies the storage class for the array in question.
   * E.g., a view type array holds no memory, while an owning type does.
   *
   * The StoragePolicy gives access to a data type, the underlying data,
   * and whether or not said data is writeable, or is just read only.
   */
  template <class Derived>
  struct StoragePolicy {
  public:
    NCA_HD StoragePolicy() = default;

    // Allow universal interconversions when doing shallow copies
    template <class OtherDerived>
    NCA_HD StoragePolicy(const StoragePolicy<OtherDerived>& other)
      : m_dtype(other.dtype())
      , m_data(other.data())
      , m_read_only(other.read_only())
    {}

    NCA_HD inline void* data() const { return this->m_data; }

    NCA_HD inline ssize_t itemsize() const {
      return static_cast<ssize_t>(ncarray::itemsize(m_dtype));
    }

    NCA_HD inline DType dtype() const { return m_dtype; }

    NCA_HD inline bool read_only() const { return m_read_only; }

  protected:
    DType m_dtype;
    void* m_data { nullptr };
    bool m_read_only { true };
  };

  /**
   * The ViewPolicy dictates arrays that have only a view of the data.
   */
  struct ViewPolicy : public StoragePolicy<ViewPolicy> {};

  /**
   * The RefPolicy dictates arrays which hold pointers to the individual
   * components of the array.
   */
  struct RefPolicy : public StoragePolicy<RefPolicy> {
  protected:
    void* m_ref_ptrs[NCARRAY_MAX_NDIM];
  };

  /**
   * The OwnerPolicy dictates an array that manages its own buffer for the
   * memory that backs the array.
   */
  struct OwnerPolicy : public StoragePolicy<OwnerPolicy> {
    NCA_HD inline void allocate(ssize_t nbytes) {
      m_storage = std::make_unique<std::uint8_t[]>(nbytes);
    }
  protected:
    std::unique_ptr<std::uint8_t[]> m_storage;
  };
} // namespace ncarray

#endif // NCARRAY_STORAGE_HH
