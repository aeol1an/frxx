#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <optional>
#include <frxx/pybind/eigen.hpp>
#include <frxx/utils/integer.hpp>
#include <frxx/utils/cppHelpers.hpp>

namespace py = pybind11;

namespace {

using frxx::utils::i64;

template <int Dimensions, typename Function>
decltype(auto) dispatch_float(py::array array, const char* name, Function&& function) {
    return frxx::pybind::dispatch_array<float, double, Dimensions>(
        array, name, " must have dtype float32 or float64",
        std::forward<Function>(function));
}

i64 unwrap_i64_py(py::object value, i64 default_value) {
    return frxx::utils::unwrap_i64(
        value.is_none() ? std::nullopt : std::optional<i64>{value.cast<i64>()},
        default_value);
}

py::object get_masked_py(py::array array, py::array mask) {
    auto typed_mask = frxx::pybind::require_array<bool, 2>(mask, "mask");
    auto eigen_mask = frxx::pybind::map_const_matrix(typed_mask);
    return dispatch_float<2>(array, "arr", [&](auto typed_array, auto) {
        auto eigen_array = frxx::pybind::map_const_matrix(typed_array);
        decltype(frxx::utils::get_masked_float2d(eigen_array, eigen_mask)) result;
        {
            py::gil_scoped_release release;
            result = frxx::utils::get_masked_float2d(eigen_array, eigen_mask);
        }
        return py::cast(std::move(result));
    });
}

void set_masked_scalar_py(py::array array, py::array mask, py::object value) {
    frxx::pybind::require_writable(array, "arr");
    auto typed_mask = frxx::pybind::require_array<bool, 2>(mask, "mask");
    auto eigen_mask = frxx::pybind::map_const_matrix(typed_mask);
    dispatch_float<2>(array, "arr", [&](auto typed_array, auto tag) {
        using T = typename decltype(tag)::type;
        auto eigen_array = frxx::pybind::map_mutable_matrix(typed_array);
        const T typed_value = value.cast<T>();
        py::gil_scoped_release release;
        frxx::utils::set_masked_float2d_scalar(
            eigen_array, eigen_mask, typed_value);
    });
}

void set_masked_array_py(py::array array, py::array mask, py::array values) {
    frxx::pybind::require_writable(array, "arr");
    auto typed_mask = frxx::pybind::require_array<bool, 2>(mask, "mask");
    auto eigen_mask = frxx::pybind::map_const_matrix(typed_mask);
    dispatch_float<2>(array, "arr", [&](auto typed_array, auto tag) {
        using T = typename decltype(tag)::type;
        auto typed_values = frxx::pybind::require_array<T, 1>(values, "val");
        auto eigen_array = frxx::pybind::map_mutable_matrix(typed_array);
        auto eigen_values = frxx::pybind::map_const_vector(typed_values);
        py::gil_scoped_release release;
        frxx::utils::set_masked_float2d_array(
            eigen_array, eigen_mask, eigen_values);
    });
}

i64 nanargmax_py(py::array array) {
    return dispatch_float<1>(array, "arr", [](auto typed_array, auto) {
        auto eigen_array = frxx::pybind::map_const_vector(typed_array);
        py::gil_scoped_release release;
        return frxx::utils::nanargmax(eigen_array);
    });
}

i64 nanargmin_py(py::array array) {
    return dispatch_float<1>(array, "arr", [](auto typed_array, auto) {
        auto eigen_array = frxx::pybind::map_const_vector(typed_array);
        py::gil_scoped_release release;
        return frxx::utils::nanargmin(eigen_array);
    });
}

}  // namespace

PYBIND11_MODULE(_cppHelpers, module) {
    module.def("unwrap_i64", &unwrap_i64_py, py::arg("opt"), py::arg("default"));
    module.def("get_masked_float2d", &get_masked_py, py::arg("arr"), py::arg("mask"));
    module.def("set_masked_float2d_scalar", &set_masked_scalar_py,
        py::arg("arr"), py::arg("mask"), py::arg("val"));
    module.def("set_masked_float2d_array", &set_masked_array_py,
        py::arg("arr"), py::arg("mask"), py::arg("val"));
    module.def("nanargmax", &nanargmax_py, py::arg("arr"));
    module.def("nanargmin", &nanargmin_py, py::arg("arr"));
}
