#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <utility>

#include <frxx/pybind/eigen.hpp>
#include <frxx/utils/angleSplitter.hpp>

namespace py = pybind11;

namespace {

py::tuple find_pulse_boundaries_py(
    py::array angle, float pixel_width_degrees, float beam_overlap_degrees
) {
    auto typed_angle = frxx::pybind::require_array<float, 1>(
        angle, "angle", " must have dtype float32");
    auto eigen_angle = frxx::pybind::map_const_vector(typed_angle);
    frxx::utils::PulseBoundaries result;
    {
        py::gil_scoped_release release;
        result = frxx::utils::find_pulse_boundaries(
            eigen_angle, pixel_width_degrees, beam_overlap_degrees);
    }
    return py::make_tuple(
        py::cast(std::move(result.indices)),
        py::cast(std::move(result.angles)));
}

std::int64_t trim_surveillance_py(py::array angle) {
    auto typed_angle = frxx::pybind::require_array<float, 1>(
        angle, "angle", " must have dtype float32");
    auto eigen_angle = frxx::pybind::map_const_vector(typed_angle);
    py::gil_scoped_release release;
    return frxx::utils::trim_surveillance(eigen_angle);
}

}  // namespace

PYBIND11_MODULE(_angleSplitter, module) {
    module.def("inDegreeRange", &frxx::utils::in_degree_range,
        py::arg("val"), py::arg("low"), py::arg("high"));
    module.def("findPulseBoundaries", &find_pulse_boundaries_py,
        py::arg("angle"), py::arg("pixelWidthDeg"), py::arg("beamOverlapDeg"));
    module.def("trimSurveillance", &trim_surveillance_py, py::arg("angle"));
}
