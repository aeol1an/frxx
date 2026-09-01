#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <vector>

#include <frxx/utils/integer.hpp>
#include <frxx/utils/numbaWindows.hpp>

namespace py = pybind11;

namespace {

py::array vector_to_array(const std::vector<double>& values) {
    py::array_t<double> output(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), output.mutable_data());
    return output;
}

}  // namespace

PYBIND11_MODULE(_numbaWindows, module) {
    using frxx::utils::i64;
    module.def("rectangular", [](i64 size) {
        return vector_to_array(frxx::utils::rectangular(size));
    }, py::arg("N"));
    module.def("hanning", [](i64 size) {
        return vector_to_array(frxx::utils::hanning(size));
    }, py::arg("N"));
    module.def("hamming", [](i64 size) {
        return vector_to_array(frxx::utils::hamming(size));
    }, py::arg("N"));
    module.def("blackman", [](i64 size) {
        return vector_to_array(frxx::utils::blackman(size));
    }, py::arg("N"));
    module.def("bartlett", [](i64 size) {
        return vector_to_array(frxx::utils::bartlett(size));
    }, py::arg("N"));
}
