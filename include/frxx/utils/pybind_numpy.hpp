#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <string>
#include <utility>

#include <frxx/utils/integer.hpp>

namespace frxx::utils {

namespace py = pybind11;

template <typename T>
struct type_tag {
    using type = T;
};

template <typename T, int Dimensions>
py::array_t<T> require_array(py::array array, const char* name) {
    if (!array.dtype().is(py::dtype::of<T>())) {
        throw py::type_error(std::string(name) + " has an unsupported dtype");
    }
    if (array.ndim() != Dimensions) {
        throw py::value_error(
            std::string(name) + " must have " + std::to_string(Dimensions) +
            " dimensions");
    }
    return py::reinterpret_borrow<py::array_t<T>>(array);
}

template <int Dimensions, typename Function>
decltype(auto) dispatch_float(py::array array, const char* name, Function&& function) {
    if (array.dtype().is(py::dtype::of<float>())) {
        return std::forward<Function>(function)(
            require_array<float, Dimensions>(array, name), type_tag<float>{});
    }
    if (array.dtype().is(py::dtype::of<double>())) {
        return std::forward<Function>(function)(
            require_array<double, Dimensions>(array, name), type_tag<double>{});
    }
    throw py::type_error(std::string(name) + " must have dtype float32 or float64");
}

template <int Dimensions, typename Function>
decltype(auto) dispatch_complex(py::array array, const char* name, Function&& function) {
    if (array.dtype().is(py::dtype::of<std::complex<float>>())) {
        return std::forward<Function>(function)(
            require_array<std::complex<float>, Dimensions>(array, name),
            type_tag<std::complex<float>>{});
    }
    if (array.dtype().is(py::dtype::of<std::complex<double>>())) {
        return std::forward<Function>(function)(
            require_array<std::complex<double>, Dimensions>(array, name),
            type_tag<std::complex<double>>{});
    }
    throw py::type_error(
        std::string(name) + " must have dtype complex64 or complex128");
}

inline i64 normalize_index(i64 index, py::ssize_t size, const char* name) {
    if (index < 0) {
        index += static_cast<i64>(size);
    }
    if (index < 0 || index >= static_cast<i64>(size)) {
        throw py::index_error(std::string(name) + " is out of bounds");
    }
    return index;
}

}  // namespace frxx::utils
