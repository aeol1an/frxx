#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <cstdint>
#include <utility>

#include <frxx/proc/algs/ACF.hpp>
#include <frxx/pybind/eigen.hpp>

namespace py = pybind11;

namespace {

py::array compute_ray_m_py(py::array X1, py::array X2, std::int32_t lag) {
    auto typed_X1 = frxx::pybind::require_c_array<
        std::complex<float>, 2>(X1, "X1", " must have dtype complex64");
    auto typed_X2 = frxx::pybind::require_c_array<
        std::complex<float>, 2>(X2, "X2", " must have dtype complex64");
    auto eigen_X1 = frxx::pybind::map_const_matrix(typed_X1);
    auto eigen_X2 = frxx::pybind::map_const_matrix(typed_X2);
    frxx::eigen::Array1D<std::complex<double>> result;
    {
        py::gil_scoped_release release;
        result = frxx::proc::algs::acf::compute_ray_m(
            eigen_X1, eigen_X2, lag);
    }
    return py::cast(std::move(result));
}

}  // namespace

PYBIND11_MODULE(_ACF, module) {
    module.def("computeRay_M", &compute_ray_m_py,
        py::arg("X1"), py::arg("X2"), py::arg("lag") = 0);
}
