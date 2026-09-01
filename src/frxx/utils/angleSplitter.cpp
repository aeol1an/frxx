#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <utility>

#include <frxx/eigen.hpp>
#include <frxx/utils/angleSplitter.hpp>

namespace py = pybind11;

namespace {

py::array_t<float> require_float32_1d(py::array array, const char* name) {
    if (!array.dtype().is(py::dtype::of<float>())) {
        throw py::type_error(std::string(name) + " must have dtype float32");
    }
    if (array.ndim() != 1) {
        throw py::value_error(std::string(name) + " must have 1 dimension");
    }
    return py::reinterpret_borrow<py::array_t<float>>(array);
}

auto map_const_vector(const py::array_t<float>& array) {
    return Eigen::Map<
        const frxx::eigen::Array1D<float>, 0, frxx::eigen::DynamicInnerStride>(
            array.data(), array.shape(0),
            frxx::eigen::DynamicInnerStride(
                array.strides(0) / static_cast<py::ssize_t>(sizeof(float))));
}

py::tuple find_pulse_boundaries_py(
    py::array angle, float pixel_width_degrees, float beam_overlap_degrees
) {
    auto typed_angle = require_float32_1d(angle, "angle");
    auto eigen_angle = map_const_vector(typed_angle);
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

}  // namespace

PYBIND11_MODULE(_angleSplitter, module) {
    module.def("inDegreeRange", &frxx::utils::in_degree_range,
        py::arg("val"), py::arg("low"), py::arg("high"));
    module.def("findPulseBoundaries", &find_pulse_boundaries_py,
        py::arg("angle"), py::arg("pixelWidthDeg"), py::arg("beamOverlapDeg"));
}
