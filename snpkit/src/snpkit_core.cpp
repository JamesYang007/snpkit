#include "decl.hpp"
#include <snpkit_core/core.hpp>
#include <snpkit_core/util/types.hpp>

namespace sk = snpkit_core;

PYBIND11_MODULE(snpkit_core, m) {
    auto m_io = m.def_submodule("io", "IO submodule.");
    register_io(m_io);

      
}