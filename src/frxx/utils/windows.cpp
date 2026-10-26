#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <utility>

#include <frxx/utils/windows.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_windows, module) {
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
    module.def("blackman_nuttall", &frxx::utils::blackman_nuttall,
        py::arg("N"), "Return a periodic Blackman-Nuttall (Nuttall4c) window.");
    module.def("rectangular_sdft_coefficients",
        &frxx::utils::rectangular_sdft_coefficients,
        "Return [1], the rectangular M-point DFT window kernel.");
    module.def("hanning_sdft_coefficients",
        &frxx::utils::hanning_sdft_coefficients,
        "Return [-0.25, 0.5, -0.25] at offsets -1, 0, +1 (NFT=M).");
    module.def("hamming_sdft_coefficients",
        &frxx::utils::hamming_sdft_coefficients,
        "Return [-0.23, 0.54, -0.23] at offsets -1, 0, +1 (NFT=M).");
    module.def("blackman_sdft_coefficients",
        &frxx::utils::blackman_sdft_coefficients,
        "Return [0.04, -0.25, 0.42, -0.25, 0.04] at offsets -2..+2 (NFT=M).");
    module.def("blackman_nuttall_sdft_coefficients",
        &frxx::utils::blackman_nuttall_sdft_coefficients,
        "Return the seven-tap periodic Blackman-Nuttall kernel at offsets -3..+3 (NFT=M).");
}
