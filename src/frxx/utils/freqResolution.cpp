#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <utility>

#include <frxx/utils/freqResolution.hpp>

namespace py = pybind11;

namespace {

py::object velocity_axis_py(
    frxx::utils::i64 size, py::object nyquist_velocity, bool flip_velocity,
    frxx::utils::i64 left_unfolds, frxx::utils::i64 right_unfolds
) {
    const py::dtype dtype = py::dtype::from_args(nyquist_velocity);
    if (dtype.is(py::dtype::of<float>())) {
        auto result = frxx::utils::velocity_axis(
            size, nyquist_velocity.cast<float>(), flip_velocity,
            left_unfolds, right_unfolds);
        return py::cast(std::move(result));
    }
    if (dtype.is(py::dtype::of<double>())) {
        auto result = frxx::utils::velocity_axis(
            size, nyquist_velocity.cast<double>(), flip_velocity,
            left_unfolds, right_unfolds);
        return py::cast(std::move(result));
    }
    throw py::type_error("va must have dtype float32 or float64");
}

}  // namespace

PYBIND11_MODULE(_freqResolution, module) {
    module.def("velResolution", &frxx::utils::velocity_resolution,
        py::arg("nPulses"), py::arg("prf") = 4000.0,
        py::arg("wavelength") = 0.0308);
    module.def("velResolutionTonPulses", &frxx::utils::velocity_resolution_to_pulses,
        py::arg("delta_v"), py::arg("prf") = 4000.0,
        py::arg("wavelength") = 0.0308);
    module.def("velocityAxis", &velocity_axis_py,
        py::arg("NFT"), py::arg("va"), py::arg("flipVel"),
        py::arg("leftUnfolds") = 0, py::arg("rightUnfolds") = 0);
    module.def("velSpanToNumBins", &frxx::utils::velocity_span_to_bins,
        py::arg("delta_v"), py::arg("nFFT"), py::arg("prf") = 4000.0,
        py::arg("wavelength") = 0.0308);
}
