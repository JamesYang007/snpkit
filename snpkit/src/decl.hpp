#pragma once
#pragma once
// Ignore all warnings for pybind + Eigen
#if defined(_MSC_VER)
#pragma warning( push, 0 )
#elif defined(__GNUC__) 
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall" 
#pragma GCC diagnostic ignored "-Wextra" 
#elif defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall" 
#pragma GCC diagnostic ignored "-Wextra" 
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#if defined(_MSC_VER)
#pragma warning( pop )
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#endif
#include <snpkit_core/util/types.hpp>

namespace py = pybind11;

void register_io(py::module_&);