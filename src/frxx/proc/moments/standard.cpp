#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

#include <frxx/proc/moments/standard.hpp>
#include <frxx/pybind/eigen.hpp>

namespace py = pybind11;

namespace {

using Complex = std::complex<double>;
using ComplexArray = frxx::eigen::Array2D<Complex>;
using ComplexArrayRef = frxx::proc::moments::standard::Complex128Array2DRef;

py::array_t<std::int32_t> require_lags(py::object lags) {
    if (!lags.is_none()) {
        return frxx::pybind::require_c_array<std::int32_t, 1>(
            py::cast<py::array>(lags), "lags", " must have dtype int32");
    }
    py::array_t<std::int32_t> defaults(2);
    auto values = defaults.mutable_unchecked<1>();
    values(0) = 0;
    values(1) = 1;
    return defaults;
}

py::tuple process_rays_py(
    py::array iqh,
    py::array iqv,
    py::array pulseBoundaries,
    py::object lags
) {
    auto typed_iqh = frxx::pybind::require_c_array<
        std::complex<float>, 2>(
            iqh, "iqh", " must have dtype complex64");
    auto typed_iqv = frxx::pybind::require_c_array<
        std::complex<float>, 2>(
            iqv, "iqv", " must have dtype complex64");
    auto typed_pulseBoundaries = frxx::pybind::require_c_array<
        frxx::utils::i64, 2>(
            pulseBoundaries, "pulseBoundaries", " must have dtype int64");
    auto typed_lags = require_lags(std::move(lags));

    const py::ssize_t lag_count = typed_lags.shape(0);
    const py::ssize_t time_count = typed_pulseBoundaries.shape(0);
    const py::ssize_t range_count = typed_iqh.shape(0);
    py::array_t<Complex> RH(
        std::vector<py::ssize_t>{lag_count, time_count, range_count});
    py::array_t<Complex> RV(
        std::vector<py::ssize_t>{time_count, range_count});
    py::array_t<Complex> RX(
        std::vector<py::ssize_t>{time_count, range_count});

    std::vector<ComplexArrayRef> RH_views;
    RH_views.reserve(static_cast<std::size_t>(lag_count));
    const py::ssize_t lag_size = time_count * range_count;
    for (py::ssize_t lag = 0; lag < lag_count; ++lag) {
        Eigen::Map<ComplexArray> RH_view(
            RH.mutable_data() + lag * lag_size,
            time_count, range_count);
        RH_views.emplace_back(RH_view);
    }

    auto eigen_iqh = frxx::pybind::map_const_matrix(typed_iqh);
    auto eigen_iqv = frxx::pybind::map_const_matrix(typed_iqv);
    auto eigen_pulseBoundaries =
        frxx::pybind::map_const_matrix(typed_pulseBoundaries);
    auto eigen_lags = frxx::pybind::map_const_vector(typed_lags);
    auto RV_view = frxx::pybind::map_mutable_matrix(RV);
    auto RX_view = frxx::pybind::map_mutable_matrix(RX);
    {
        py::gil_scoped_release release;
        frxx::proc::moments::standard::process_rays(
            eigen_iqh, eigen_iqv, eigen_pulseBoundaries, eigen_lags,
            RH_views, RV_view, RX_view);
    }
    return py::make_tuple(std::move(RH), std::move(RV), std::move(RX));
}

}  // namespace

PYBIND11_MODULE(_standard, module) {
    module.def("_processRays", &process_rays_py,
        py::arg("iqh"), py::arg("iqv"), py::arg("pulseBoundaries"),
        py::arg("lags") = py::none());
}
