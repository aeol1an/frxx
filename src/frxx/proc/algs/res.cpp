#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <string>
#include <utility>

#include <frxx/proc/algs/res.hpp>
#include <frxx/utils/integer.hpp>

namespace py = pybind11;

namespace {

namespace native = frxx::proc::algs::res;
using frxx::utils::i64;

template <typename Scalar>
struct MatrixTypes;

template <>
struct MatrixTypes<std::complex<float>> {
    using Matrix = native::Complex64Matrix;
    using Result = native::Complex64SubsetResult;
};

template <>
struct MatrixTypes<std::complex<double>> {
    using Matrix = native::Complex128Matrix;
    using Result = native::Complex128SubsetResult;
};

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

template <typename Function>
decltype(auto) dispatch_complex(py::array array, const char* name, Function&& function) {
    if (array.dtype().is(py::dtype::of<std::complex<float>>())) {
        return std::forward<Function>(function)(
            require_array<std::complex<float>, 2>(array, name),
            type_tag<std::complex<float>>{});
    }
    if (array.dtype().is(py::dtype::of<std::complex<double>>())) {
        return std::forward<Function>(function)(
            require_array<std::complex<double>, 2>(array, name),
            type_tag<std::complex<double>>{});
    }
    throw py::type_error(
        std::string(name) + " must have dtype complex64 or complex128");
}

template <typename Function>
decltype(auto) translate_native_errors(Function&& function) {
    try {
        return std::forward<Function>(function)();
    } catch (const native::BoundsError& error) {
        throw py::index_error(error.what());
    } catch (const native::ArgumentError& error) {
        throw py::value_error(error.what());
    }
}

template <typename Scalar>
native::DynamicStride matrix_stride(const py::array_t<Scalar>& array) {
    return {
        array.strides(0) / static_cast<py::ssize_t>(sizeof(Scalar)),
        array.strides(1) / static_cast<py::ssize_t>(sizeof(Scalar)),
    };
}

template <typename Scalar>
auto map_const_matrix(const py::array_t<Scalar>& array) {
    using Matrix = typename MatrixTypes<Scalar>::Matrix;
    return Eigen::Map<const Matrix, 0, native::DynamicStride>(
        array.data(), array.shape(0), array.shape(1), matrix_stride(array));
}

template <typename Scalar>
auto map_mutable_matrix(py::array_t<Scalar>& array) {
    using Matrix = typename MatrixTypes<Scalar>::Matrix;
    return Eigen::Map<Matrix, 0, native::DynamicStride>(
        array.mutable_data(), array.shape(0), array.shape(1), matrix_stride(array));
}

auto map_const_int_matrix(const py::array_t<i64>& array) {
    return Eigen::Map<const native::Int64Matrix, 0, native::DynamicStride>(
        array.data(), array.shape(0), array.shape(1),
        native::DynamicStride(
            array.strides(0) / static_cast<py::ssize_t>(sizeof(i64)),
            array.strides(1) / static_cast<py::ssize_t>(sizeof(i64))));
}

auto map_const_int_vector(const py::array_t<i64>& array) {
    return Eigen::Map<
        const native::Int64Vector, 0, Eigen::InnerStride<Eigen::Dynamic>>(
            array.data(), array.shape(0),
            Eigen::InnerStride<Eigen::Dynamic>(
                array.strides(0) / static_cast<py::ssize_t>(sizeof(i64))));
}

void range_subset_py(
    py::array iq,
    py::array result,
    i64 K,
    i64 Koffset,
    i64 range_count,
    i64 start_range,
    i64 first_pulse,
    i64 last_pulse
) {
    if (!result.writeable()) {
        throw py::value_error("result must be writable");
    }
    dispatch_complex(iq, "iq", [&](auto typed_iq, auto tag) {
        using Scalar = typename decltype(tag)::type;
        auto typed_result = require_array<Scalar, 2>(result, "result");
        auto input = map_const_matrix(typed_iq);
        auto output = map_mutable_matrix(typed_result);
        translate_native_errors([&] {
            py::gil_scoped_release release;
            native::range_subset(
                input, output, K, Koffset, range_count,
                start_range, first_pulse, last_pulse);
        });
    });
}

void azimuth_subset_py(
    py::array iq,
    py::array result,
    i64 range_count,
    i64 start_range,
    py::array first_pulses,
    py::array last_pulses
) {
    if (!result.writeable()) {
        throw py::value_error("result must be writable");
    }
    auto typed_first = require_array<i64, 1>(first_pulses, "fps");
    auto typed_last = require_array<i64, 1>(last_pulses, "lps");
    auto first = map_const_int_vector(typed_first);
    auto last = map_const_int_vector(typed_last);
    dispatch_complex(iq, "iq", [&](auto typed_iq, auto tag) {
        using Scalar = typename decltype(tag)::type;
        auto typed_result = require_array<Scalar, 2>(result, "result");
        auto input = map_const_matrix(typed_iq);
        auto output = map_mutable_matrix(typed_result);
        translate_native_errors([&] {
            py::gil_scoped_release release;
            native::azimuth_subset(
                input, output, range_count, start_range, first, last);
        });
    });
}

py::tuple subset_iq_py(
    py::array iq,
    i64 azimuth_index,
    i64 azimuth_count,
    bool azimuth_increasing,
    py::array pulse_boundaries,
    py::array ranges,
    i64 swath_pulses,
    i64 K,
    i64 Koffset,
    i64 average_strategy,
    bool shape_only
) {
    auto typed_boundaries = require_array<i64, 2>(
        pulse_boundaries, "pulseBoundaries");
    auto typed_ranges = require_array<i64, 1>(ranges, "iranges");
    auto boundaries = map_const_int_matrix(typed_boundaries);
    auto range_indices = map_const_int_vector(typed_ranges);

    return dispatch_complex(iq, "iq", [&](auto typed_iq, auto tag) {
        using Scalar = typename decltype(tag)::type;
        using Result = typename MatrixTypes<Scalar>::Result;
        auto input = map_const_matrix(typed_iq);
        Result result;
        translate_native_errors([&] {
            py::gil_scoped_release release;
            result = native::subset_iq(
                input, azimuth_index, azimuth_count, azimuth_increasing,
                boundaries, range_indices, swath_pulses, K, Koffset,
                average_strategy, shape_only);
        });
        return py::make_tuple(
            py::cast(std::move(result.values)),
            result.range_count,
            result.pulse_count);
    });
}

}  // namespace

PYBIND11_MODULE(_res, module) {
    module.doc() = "Python bindings for the native Eigen IQ subsetting API.";
    module.def(
        "_rangeSubsetIQ", &range_subset_py,
        py::arg("iq"), py::arg("result"), py::arg("K"), py::arg("Koffset"),
        py::arg("NR"), py::arg("startRange"), py::arg("fp"), py::arg("lp"));
    module.def(
        "_azSubsetIQ", &azimuth_subset_py,
        py::arg("iq"), py::arg("result"), py::arg("NR"), py::arg("startRange"),
        py::arg("fps"), py::arg("lps"));
    module.def(
        "subsetIQcpp", &subset_iq_py,
        py::arg("iq"), py::arg("iaz"), py::arg("naz"), py::arg("azIncreasing"),
        py::arg("pulseBoundaries"), py::arg("iranges"), py::arg("swathPulses") = -1,
        py::arg("K") = 1, py::arg("KOffset") = 0, py::arg("avgStrat") = 1,
        py::arg("shapeOnly") = false);
}
