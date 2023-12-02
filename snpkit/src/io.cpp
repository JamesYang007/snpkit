#include "decl.hpp"
#include <snpkit_core/io/msp_reader.hpp>
#include <snpkit_core/util/types.hpp>

namespace sk = snpkit_core;

void msp_reader(py::module_& m)
{
    using io_t = sk::io::MSPReader;
    using string_t = typename io_t::string_t;
    py::class_<io_t>(m, "MSPReader")
        .def(py::init<const string_t&>(),
            py::arg("filename")
        )
        .def("read", &io_t::read,
            py::arg("max_rows"),
            py::arg("delimiter"),
            py::arg("hap_ids_indices"),
            py::arg("buffer_size"),
            py::arg("n_rows_hint"),
            py::arg("n_threads")
        )
        .def_readonly("ancestry_map", &io_t::_ancestry_map)
        .def_readonly("haplotype_IDs", &io_t::_haplotype_IDs)
        .def_property_readonly("sample_IDs", [](const io_t& s) { 
            return Eigen::Map<const sk::util::rowvec_type<int32_t>>(
                s._sample_IDs.data(),
                s._sample_IDs.size()
            );
        })
        .def_property_readonly("chm", [](const io_t& s) {
            return Eigen::Map<const sk::util::rowvec_type<int32_t>>(
                s._chm.data(),
                s._chm.size()
            );
        })
        .def_property_readonly("pos", [](const io_t& s) {
            return Eigen::Map<const sk::util::rowarr_type<int32_t>>(
                s._pos.data(),
                s._pos.size() / 2,
                2
            );
        })
        .def_property_readonly("gpos", [](const io_t& s) {
            return Eigen::Map<const sk::util::rowarr_type<double>>(
                s._gpos.data(),
                s._gpos.size() / 2,
                2
            );
        })
        .def_property_readonly("n_snps", [](const io_t& s) {
            return Eigen::Map<const sk::util::rowvec_type<int32_t>>(
                s._n_snps.data(),
                s._n_snps.size()
            );
        })
        .def_property_readonly("lai", [](const io_t& s) {
            return Eigen::Map<const sk::util::rowarr_type<int8_t>>(
                s._lai.data(),
                s._chm.size(),
                s._haplotype_IDs.size()
            );
        })
        ;
}


void register_io(py::module_& m)
{
    msp_reader(m);
}