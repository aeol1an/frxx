#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <vector>

#include <frxx/utils/angleSplitter.hpp>
#include <frxx/utils/integer.hpp>
#include <frxx/utils/pybind_numpy.hpp>

namespace py = pybind11;

namespace {

py::tuple find_pulse_boundaries(
    py::array angle, float pixel_width_degrees, float beam_overlap_degrees
) {
    auto typed_angle = frxx::utils::require_array<float, 1>(angle, "angle");
    auto angle_view = typed_angle.unchecked<1>();
    frxx::utils::PulseBoundaries result;
    {
        py::gil_scoped_release release;
        result = frxx::utils::find_pulse_boundaries(
            angle_view, typed_angle.shape(0), pixel_width_degrees, beam_overlap_degrees);
    }

    const py::ssize_t groups = static_cast<py::ssize_t>(result.angles.size());
    py::array_t<frxx::utils::i64> boundaries(
        std::vector<py::ssize_t>{groups, py::ssize_t{2}});
    std::copy(result.indices.begin(), result.indices.end(), boundaries.mutable_data());
    py::array_t<float> angles(groups);
    std::copy(result.angles.begin(), result.angles.end(), angles.mutable_data());
    return py::make_tuple(boundaries, angles);
}

}  // namespace

PYBIND11_MODULE(_angleSplitter, module) {
    module.def("inDegreeRange", &frxx::utils::in_degree_range,
        py::arg("val"), py::arg("low"), py::arg("high"));
    module.def("findPulseBoundaries", &find_pulse_boundaries,
        py::arg("angle"), py::arg("pixelWidthDeg"), py::arg("beamOverlapDeg"));
}
