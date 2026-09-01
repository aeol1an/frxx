#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <utility>

#include <frxx/proc/algs/bootstrapDPSD.hpp>
#include <frxx/pybind/eigen.hpp>
#include <frxx/utils/integer.hpp>

namespace py = pybind11;

namespace {

namespace native = frxx::proc::algs::bootstrap_dpsd;
using frxx::utils::i64;

py::tuple computeSingleSpectrum_py(
    py::array VH,
    py::array VV,
    py::array w,
    i64 M,
    i64 NFT,
    double B,
    double r
) {
    auto typed_VH = frxx::pybind::require_c_array<std::complex<float>, 1>(
        VH, "VH", " must have dtype complex64");
    auto typed_VV = frxx::pybind::require_c_array<std::complex<float>, 1>(
        VV, "VV", " must have dtype complex64");
    auto typed_w = frxx::pybind::require_c_array<double, 1>(
        w, "w", " must have dtype float64");
    auto eigen_VH = frxx::pybind::map_const_vector(typed_VH);
    auto eigen_VV = frxx::pybind::map_const_vector(typed_VV);
    auto eigen_w = frxx::pybind::map_const_vector(typed_w);

    native::SingleSpectrumResult result;
    {
        py::gil_scoped_release release;
        result = native::computeSingleSpectrum(
            eigen_VH, eigen_VV, eigen_w, M, NFT, B, r);
    }
    return py::make_tuple(
        py::cast(std::move(result.SHi)),
        py::cast(std::move(result.SVi)),
        py::cast(std::move(result.SXi)));
}

py::tuple computeMultipleSpectra_py(
    py::array VH,
    py::array VV,
    py::array w,
    i64 NK,
    i64 M,
    i64 NFT,
    double B,
    double r
) {
    auto typed_VH = frxx::pybind::require_c_array<std::complex<float>, 2>(
        VH, "VH", " must have dtype complex64");
    auto typed_VV = frxx::pybind::require_c_array<std::complex<float>, 2>(
        VV, "VV", " must have dtype complex64");
    auto typed_w = frxx::pybind::require_c_array<double, 1>(
        w, "w", " must have dtype float64");
    auto eigen_VH = frxx::pybind::map_const_matrix(typed_VH);
    auto eigen_VV = frxx::pybind::map_const_matrix(typed_VV);
    auto eigen_w = frxx::pybind::map_const_vector(typed_w);

    native::MultipleSpectraResult result;
    {
        py::gil_scoped_release release;
        result = native::computeMultipleSpectra(
            eigen_VH, eigen_VV, eigen_w, NK, M, NFT, B, r);
    }
    return py::make_tuple(
        py::cast(std::move(result.SH)),
        py::cast(std::move(result.SV)),
        py::cast(std::move(result.SX)));
}

py::tuple processRay_S_py(
    py::array iqh,
    py::array iqv,
    py::array window,
    double nBootstraps,
    i64 K,
    i64 NFT
) {
    auto typed_iqh = frxx::pybind::require_c_array<std::complex<float>, 2>(
        iqh, "iqh", " must have dtype complex64");
    auto typed_iqv = frxx::pybind::require_c_array<std::complex<float>, 2>(
        iqv, "iqv", " must have dtype complex64");
    auto typed_window = frxx::pybind::require_c_array<double, 1>(
        window, "window", " must have dtype float64");
    auto eigen_iqh = frxx::pybind::map_const_matrix(typed_iqh);
    auto eigen_iqv = frxx::pybind::map_const_matrix(typed_iqv);
    auto eigen_window = frxx::pybind::map_const_vector(typed_window);

    native::ProcessRayResult result;
    {
        py::gil_scoped_release release;
        result = native::processRay_S(
            eigen_iqh, eigen_iqv, eigen_window, nBootstraps, K, NFT);
    }
    return py::make_tuple(
        py::cast(std::move(result.PSDH)),
        py::cast(std::move(result.PSDV)),
        py::cast(std::move(result.sZDR)),
        py::cast(std::move(result.sRHOHV)));
}

}  // namespace

PYBIND11_MODULE(_bootstrapDPSD, module) {
    module.def(
        "_computeSingleSpectrum", &computeSingleSpectrum_py,
        py::arg("VH"), py::arg("VV"), py::arg("w"),
        py::arg("M"), py::arg("NFT"), py::arg("B"), py::arg("r"));
    module.def(
        "_computeMultipleSpectra", &computeMultipleSpectra_py,
        py::arg("VH"), py::arg("VV"), py::arg("w"),
        py::arg("NK"), py::arg("M"), py::arg("NFT"),
        py::arg("B"), py::arg("r"));
    module.def(
        "processRay_S", &processRay_S_py,
        py::arg("iqh"), py::arg("iqv"), py::arg("window"),
        py::arg("nBootstraps"), py::arg("K") = 1, py::arg("NFT") = 1);
}
