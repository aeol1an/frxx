#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <optional>
#include <vector>

#include <frxx/utils/integer.hpp>
#include <frxx/utils/numbaHelpers.hpp>
#include <frxx/utils/pybind_numpy.hpp>

namespace py = pybind11;

namespace {

using frxx::utils::i64;

void require_same_shape(const py::array& array, const py::array& mask) {
    if (array.shape(0) != mask.shape(0) || array.shape(1) != mask.shape(1)) {
        throw py::value_error("arr and mask must have the same shape");
    }
}

i64 unwrap_i64(py::object value, i64 default_value) {
    return frxx::utils::unwrap_i64(
        value.is_none() ? std::nullopt : std::optional<i64>{value.cast<i64>()},
        default_value);
}

py::array get_masked(py::array array, py::array mask) {
    auto typed_mask = frxx::utils::require_array<bool, 2>(mask, "mask");
    require_same_shape(array, mask);
    return frxx::utils::dispatch_float<2>(array, "arr", [&](auto typed_array, auto tag) {
        using T = typename decltype(tag)::type;
        auto array_view = typed_array.template unchecked<2>();
        auto mask_view = typed_mask.template unchecked<2>();
        std::vector<T> values;
        {
            py::gil_scoped_release release;
            values = frxx::utils::get_masked_float2d<T>(
                array_view, mask_view, typed_array.shape(0), typed_array.shape(1));
        }
        py::array_t<T> output(static_cast<py::ssize_t>(values.size()));
        std::copy(values.begin(), values.end(), output.mutable_data());
        return py::array(std::move(output));
    });
}

void set_masked_scalar(py::array array, py::array mask, py::object value) {
    if (!array.writeable()) {
        throw py::value_error("arr must be writable");
    }
    auto typed_mask = frxx::utils::require_array<bool, 2>(mask, "mask");
    require_same_shape(array, mask);
    frxx::utils::dispatch_float<2>(array, "arr", [&](auto typed_array, auto tag) {
        using T = typename decltype(tag)::type;
        auto array_view = typed_array.template mutable_unchecked<2>();
        auto mask_view = typed_mask.template unchecked<2>();
        const T typed_value = value.cast<T>();
        py::gil_scoped_release release;
        frxx::utils::set_masked_float2d_scalar<T>(
            array_view, mask_view, typed_array.shape(0), typed_array.shape(1), typed_value);
    });
}

void set_masked_array(py::array array, py::array mask, py::array values) {
    if (!array.writeable()) {
        throw py::value_error("arr must be writable");
    }
    auto typed_mask = frxx::utils::require_array<bool, 2>(mask, "mask");
    require_same_shape(array, mask);
    frxx::utils::dispatch_float<2>(array, "arr", [&](auto typed_array, auto tag) {
        using T = typename decltype(tag)::type;
        auto typed_values = frxx::utils::require_array<T, 1>(values, "val");
        auto array_view = typed_array.template mutable_unchecked<2>();
        auto mask_view = typed_mask.template unchecked<2>();
        auto value_view = typed_values.template unchecked<1>();

        i64 selected = 0;
        for (py::ssize_t row = 0; row < typed_mask.shape(0); ++row) {
            for (py::ssize_t column = 0; column < typed_mask.shape(1); ++column) {
                selected += mask_view(row, column) ? 1 : 0;
            }
        }
        if (typed_values.shape(0) < selected) {
            throw py::value_error("val does not contain enough elements");
        }

        py::gil_scoped_release release;
        frxx::utils::set_masked_float2d_array(
            array_view, mask_view, value_view,
            typed_array.shape(0), typed_array.shape(1));
    });
}

i64 nanargmax(py::array array) {
    return frxx::utils::dispatch_float<1>(array, "arr", [](auto typed_array, auto) {
        auto view = typed_array.template unchecked<1>();
        py::gil_scoped_release release;
        return frxx::utils::nanargmax(view, typed_array.shape(0));
    });
}

i64 nanargmin(py::array array) {
    return frxx::utils::dispatch_float<1>(array, "arr", [](auto typed_array, auto) {
        auto view = typed_array.template unchecked<1>();
        py::gil_scoped_release release;
        return frxx::utils::nanargmin(view, typed_array.shape(0));
    });
}

}  // namespace

PYBIND11_MODULE(_numbaHelpers, module) {
    module.def("unwrap_i64", &unwrap_i64, py::arg("opt"), py::arg("default"));
    module.def("get_masked_float2d", &get_masked, py::arg("arr"), py::arg("mask"));
    module.def("set_masked_float2d_scalar", &set_masked_scalar,
        py::arg("arr"), py::arg("mask"), py::arg("val"));
    module.def("set_masked_float2d_array", &set_masked_array,
        py::arg("arr"), py::arg("mask"), py::arg("val"));
    module.def("nanargmax", &nanargmax, py::arg("arr"));
    module.def("nanargmin", &nanargmin, py::arg("arr"));
}
