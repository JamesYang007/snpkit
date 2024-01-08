#include "decl.hpp"
#include <snpkit_core/core.hpp>
#include <snpkit_core/util/types.hpp>

namespace sk = snpkit_core;

PYBIND11_MODULE(snpkit_core, m) {
    auto m_io = m.def_submodule("io", "IO submodule.");
    register_io(m_io);

    m.def("to_sample_major_int8", &sk::to_sample_major<int8_t>); 
    m.def("calldata_sum_int8", &sk::calldata_sum<int8_t>); 
    m.def("column_mean", &sk::column_mean<int8_t, double>);
    m.def("calldata_subset_rows_cols_int8", &sk::calldata_subset_rows_cols<int8_t, int32_t>);
}