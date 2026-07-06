#include "pyncarray/binding_builder.hh"

namespace pyncarray {
  bool g_eager_eval = true;

  void set_eager(bool eager) {
    g_eager_eval = eager;
  }

  bool is_eager() { return g_eager_eval; }
} // namespace pyncarray
