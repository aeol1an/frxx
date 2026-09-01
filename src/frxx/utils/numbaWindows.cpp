#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <utility>

#include <frxx/utils/numbaWindows.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_numbaWindows, module) {
    using frxx::utils::i64;
    module.def("rectangular", [](i64 size) {
        return py::cast(frxx::utils::rectangular(size));
    }, py::arg("N"));
    module.def("hanning", [](i64 size) {
        return py::cast(frxx::utils::hanning(size));
    }, py::arg("N"));
    module.def("hamming", [](i64 size) {
        return py::cast(frxx::utils::hamming(size));
    }, py::arg("N"));
    module.def("blackman", [](i64 size) {
        return py::cast(frxx::utils::blackman(size));
    }, py::arg("N"));
    module.def("bartlett", [](i64 size) {
        return py::cast(frxx::utils::bartlett(size));
    }, py::arg("N"));
}
