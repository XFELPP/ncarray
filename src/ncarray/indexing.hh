#ifndef NCARRAY_INDEXING_HH
#define NCARRAY_INDEXING_HH

#ifdef _WIN32
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <sys/types.h>
#endif
#include <variant>
#include <vector>

namespace ncarray {
  struct Ellipsis {};

  struct Slice {
    Slice(ssize_t start_, ssize_t stop_)
      : start(start_)
      , stop(stop_)
      , step(1)
      , length(stop - start)
    {}

    Slice(ssize_t start_, ssize_t stop_, ssize_t step_)
      : start(start_)
      , stop(stop_)
      , step(step_)
      , length(step % 2 == 0 ? (stop - start) / 2 : (stop - start + 1) / 2)
    {}

    ssize_t start;
    ssize_t stop;
    ssize_t step;
    ssize_t length;
  };

  using IndexVariant = std::variant<ssize_t, Slice, Ellipsis>;
  using ArrayIndices = std::vector<IndexVariant>;
} // namespace ncarray
#endif // NCARRAY_INDEXING_HH
