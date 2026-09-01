#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <utility>

#include <frxx/eigen.hpp>

namespace frxx::pybind {

namespace py = pybind11;

template <typename T>
struct TypeTag {
    using type = T;
};

/// Require an exact NumPy dtype and dimensionality without copying the array.
template <typename T, int Dimensions>
py::array_t<T> require_array(
    py::array array,
    const char* name,
    const char* dtype_error = " has an unsupported dtype"
) {
    if (!array.dtype().is(py::dtype::of<T>())) {
        throw py::type_error(std::string(name) + dtype_error);
    }
    if (array.ndim() != Dimensions) {
        throw py::value_error(
            std::string(name) + " must have " + std::to_string(Dimensions) +
            " dimensions");
    }
    return py::reinterpret_borrow<py::array_t<T>>(array);
}

/// Require an exact dtype, dimensionality, and C-contiguous memory layout.
template <typename T, int Dimensions>
py::array_t<T> require_c_array(
    py::array array,
    const char* name,
    const char* dtype_error = " has an unsupported dtype"
) {
    auto typed = require_array<T, Dimensions>(array, name, dtype_error);
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
    return typed;
}

/// Dispatch an array with one of two supported dtypes after validating its rank.
template <typename First, typename Second, int Dimensions, typename Function>
decltype(auto) dispatch_array(
    py::array array,
    const char* name,
    const char* dtype_error,
    Function&& function
) {
    if (array.dtype().is(py::dtype::of<First>())) {
        return std::forward<Function>(function)(
            require_array<First, Dimensions>(array, name), TypeTag<First>{});
    }
    if (array.dtype().is(py::dtype::of<Second>())) {
        return std::forward<Function>(function)(
            require_array<Second, Dimensions>(array, name), TypeTag<Second>{});
    }
    throw py::type_error(std::string(name) + dtype_error);
}

inline void require_writable(const py::array& array, const char* name) {
    if (!array.writeable()) {
        throw py::value_error(std::string(name) + " must be writable");
    }
}

template <typename T>
eigen::DynamicStride matrix_stride(const py::array_t<T>& array) {
    return {
        array.strides(0) / static_cast<py::ssize_t>(sizeof(T)),
        array.strides(1) / static_cast<py::ssize_t>(sizeof(T)),
    };
}

/// Map an arbitrary-stride NumPy matrix as the requested row-major Eigen type.
template <typename EigenType, typename T>
auto map_const_matrix_as(const py::array_t<T>& array) {
    return Eigen::Map<const EigenType, 0, eigen::DynamicStride>(
        array.data(), array.shape(0), array.shape(1), matrix_stride(array));
}

template <typename EigenType, typename T>
auto map_mutable_matrix_as(py::array_t<T>& array) {
    return Eigen::Map<EigenType, 0, eigen::DynamicStride>(
        array.mutable_data(), array.shape(0), array.shape(1), matrix_stride(array));
}

template <typename T>
auto map_const_matrix(const py::array_t<T>& array) {
    return map_const_matrix_as<eigen::Array2D<T>>(array);
}

template <typename T>
auto map_mutable_matrix(py::array_t<T>& array) {
    return map_mutable_matrix_as<eigen::Array2D<T>>(array);
}

/// Map an arbitrary-stride NumPy vector as the requested Eigen vector type.
template <typename EigenType, typename T>
auto map_const_vector_as(const py::array_t<T>& array) {
    return Eigen::Map<const EigenType, 0, eigen::DynamicInnerStride>(
        array.data(), array.shape(0),
        eigen::DynamicInnerStride(
            array.strides(0) / static_cast<py::ssize_t>(sizeof(T))));
}

template <typename T>
auto map_const_vector(const py::array_t<T>& array) {
    return map_const_vector_as<eigen::Array1D<T>>(array);
}

}  // namespace frxx::pybind
