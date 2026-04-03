#ifndef NCARRAY_STORAGE_HH
#define NCARRAY_STORAGE_HH

#include "ncarray/dtype.hh"

#ifdef NCA_HAS_CUDA
#include "ncarray/device/utilities.cuh" // Macro error checks

#include "cuda_runtime_api.h"
#endif

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>

#ifndef NCA_HD
#ifdef __CUDACC__
#define NCA_HD __host__ __device__
#else
#define NCA_HD
#endif
#endif

#ifndef NCA_H
#ifdef __CUDACC__
#define NCA_H __host__
#else
#define NCA_H
#endif
#endif

#ifndef NCARRAY_MAX_NDIM
#define NCARRAY_MAX_NDIM 10
#endif

namespace ncarray {
#ifdef NCA_HAS_CUDA
  inline std::atomic<bool> NCA_STREAMS_INIT { false };
  inline cudaStream_t alloc_stream;
#endif
  struct MemTag {};
  struct HostTag : MemTag {};
  struct DevTag : MemTag {};


  struct ViewTag {};
  struct RefTag {};
  struct OwnerTag {};


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
    using MemType = HostTag;

    StoragePolicy() = default;

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

    /**
     * The repr functions return a string to identify the storage policy when
     * writing out string representations of the array.
     */
    NCA_HD inline const char* storage_repr() const {
      return static_cast<Derived*>(this)->storage_repr();
    }

  protected:
    DType m_dtype;
    void* m_data { nullptr };
    bool m_read_only { true };
  };

  /**
   * The ViewPolicy dictates arrays that have only a view of the data.
   */
  struct ViewPolicy : public StoragePolicy<ViewPolicy>, public ViewTag {
  public:
    using MemType = HostTag;

    NCA_HD inline const char* storage_repr() const { return "View"; }
  };

  struct DevViewPolicy : public StoragePolicy<DevViewPolicy>, public ViewTag {
  public:
    using MemType = DevTag;

    NCA_HD inline const char* storage_repr() const { return "View"; }
  };

  /**
   * The RefPolicy dictates arrays which hold pointers to the individual
   * components of the array.
   */
  struct RefPolicy : public StoragePolicy<RefPolicy>, public RefTag {
  public:
    using MemType = HostTag;

    NCA_HD inline const char* storage_repr() const { return "Ref"; }

  protected:
    void* m_ref_ptrs[NCARRAY_MAX_NDIM];
  };

  struct DevRefPolicy : public StoragePolicy<DevRefPolicy>, public RefTag {
  public:
    using MemType = DevTag;

    NCA_HD inline const char* storage_repr() const { return "Ref"; }

  protected:
    void* m_ref_ptrs[NCARRAY_MAX_NDIM];
  };

  struct DevDeleter {
    void operator()(std::uint8_t* ptr) {
#ifdef NCA_HAS_CUDA
      cudaFree(ptr);
#endif
    }
  };

  struct HostDeleter {
    void operator()(std::uint8_t* ptr) {
      delete[] ptr;
    }
  };

  /**
   * The OwnerPolicy dictates an array that manages its own buffer for the
   * memory that backs the array.
   */
  struct OwnerPolicy : public StoragePolicy<OwnerPolicy>, public OwnerTag {
  public:
    using MemType = HostTag;

    // By default, owners are NOT read only.
    NCA_H OwnerPolicy() { this->m_read_only = false; }

    NCA_HD inline const char* storage_repr() const { return "Owner"; }

    NCA_H inline void allocate(ssize_t nbytes) {
      m_storage =
          std::unique_ptr<std::uint8_t[], HostDeleter>(new std::uint8_t[nbytes], HostDeleter());
    }

    NCA_H inline void copy(void* src, ssize_t nbytes) {
      std::copy(reinterpret_cast<std::uint8_t*>(src),
                reinterpret_cast<std::uint8_t*>(src) + nbytes,
                this->m_storage.get());
    }

  protected:
    std::unique_ptr<std::uint8_t[], HostDeleter> m_storage;
  };

  struct DevOwnerPolicy : public StoragePolicy<DevOwnerPolicy>, public OwnerTag {
  public:
    using MemType = DevTag;

    // By default, owners are NOT read only.
    NCA_H DevOwnerPolicy() { this->m_read_only = false; }

    NCA_HD inline const char* storage_repr() const { return "Owner"; }

    NCA_H inline void allocate(ssize_t nbytes) {
#ifdef NCA_HAS_CUDA
      if (!NCA_STREAMS_INIT.exchange(true)) {
        CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&alloc_stream,
                                                   cudaStreamNonBlocking));
      }
      std::uint8_t* devPtr { nullptr };
      CHECK_CUDA_ERROR(cudaMallocAsync(reinterpret_cast<void**>(&devPtr),
                                       nbytes,
                                       alloc_stream));
      cudaDeviceSynchronize();
      m_storage = std::unique_ptr<std::uint8_t[], DevDeleter>(devPtr, DevDeleter());
#endif
    }

    NCA_H inline void copy(void* src, ssize_t nbytes) {
#ifdef NCA_HAS_CUDA
      CHECK_CUDA_ERROR(cudaMemcpy(this->m_storage.get(),
                                  src,
                                  nbytes,
                                  cudaMemcpyDefault));
#endif
    }

  protected:
    std::unique_ptr<std::uint8_t[], DevDeleter> m_storage;
  private:
#ifdef NCA_HAS_CUDA
    cudaStream_t m_stream;
#endif
  };

  template <class MemTag>
  struct StoragePolicyTraits;

  template <>
  struct StoragePolicyTraits<HostTag> {
    using View = ViewPolicy;
    using Ref = RefPolicy;
    using Owner = OwnerPolicy;
  };

  template <>
  struct StoragePolicyTraits<DevTag> {
    using View = DevViewPolicy;
    using Ref = DevRefPolicy;
    using Owner = DevOwnerPolicy;
  };

} // namespace ncarray

#endif // NCARRAY_STORAGE_HH
