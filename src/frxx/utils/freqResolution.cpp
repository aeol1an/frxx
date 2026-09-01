#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <vector>

#include <frxx/utils/freqResolution.hpp>
#include <frxx/utils/integer.hpp>

namespace py = pybind11;

namespace {

template <typename T>
py::array typed_vector_to_array(const std::vector<T>& values) {
    py::array_t<T> output(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), output.mutable_data());
    return output;
}

py::array velocity_axis(
    frxx::utils::i64 size, py::object nyquist_velocity, bool flip_velocity,
    frxx::utils::i64 left_unfolds, frxx::utils::i64 right_unfolds
) {
    const py::dtype dtype = py::dtype::from_args(nyquist_velocity);
    if (dtype.is(py::dtype::of<float>())) {
        return typed_vector_to_array(frxx::utils::velocity_axis(
            size, nyquist_velocity.cast<float>(), flip_velocity,
            left_unfolds, right_unfolds));
    }
    if (dtype.is(py::dtype::of<double>())) {
        return typed_vector_to_array(frxx::utils::velocity_axis(
            size, nyquist_velocity.cast<double>(), flip_velocity,
            left_unfolds, right_unfolds));
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
    module.def("velocityAxis", &velocity_axis,
        py::arg("NFT"), py::arg("va"), py::arg("flipVel"),
        py::arg("leftUnfolds") = 0, py::arg("rightUnfolds") = 0);
    module.def("velSpanToNumBins", &frxx::utils::velocity_span_to_bins,
        py::arg("delta_v"), py::arg("nFFT"), py::arg("prf") = 4000.0,
        py::arg("wavelength") = 0.0308);
}
