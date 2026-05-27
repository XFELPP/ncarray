/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NCARRAY_HOST_ALLOCATOR_HH
#define NCARRAY_HOST_ALLOCATOR_HH

#include <cstddef>
#include <new>
#include <type_traits>

namespace ncarray {
  /**
   * A simple allocator for STL containers which upholds alignment specifications.
   *
   * A number of functions are requirements for STL compliance - e.g. allocators
   * compare equal.
   */
  namespace host {
    namespace impl {
      template <typename T, std::size_t Alignment = 16>
      struct AlignedAllocator {
        static_assert(Alignment >= alignof(T), "Alignment must be >= alignof(T)");
        static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of 2");

        using value_type = T;

        AlignedAllocator() noexcept = default;

        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

        T* allocate(std::size_t n) {
          if (n == 0) {
            return nullptr;
          }

          return
            static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(Alignment)));
        }

        void deallocate(T* ptr, std::size_t) noexcept {
          ::operator delete(ptr, std::align_val_t(Alignment));
        }

        // Required for compliance
        template <typename U>
        struct rebind {
          using other = AlignedAllocator<U, Alignment>;
        };
      };

      template <typename T1, std::size_t A1, typename T2, std::size_t A2>
      bool operator==(const AlignedAllocator<T1, A1>&,
                      const AlignedAllocator<T2, A2>&) noexcept {
        return A1 == A2;
      }

      template <typename T1, std::size_t A1, typename T2, std::size_t A2>
      bool operator!=(const AlignedAllocator<T1, A1>&,
                      const AlignedAllocator<T2, A2>&) noexcept {
        return !(A1 == A2);
      }
    } // namespace impl
  } // namespace host
} // namespace ncarray

#endif // NCARRAY_HOST_ALLOCATOR_HH
