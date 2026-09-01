#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <utility>

#include <frxx/proc/algs/fuzzyDCA.hpp>
#include <frxx/pybind/eigen.hpp>

namespace py = pybind11;

namespace {

namespace native = frxx::proc::algs::fuzzy_dca;
using frxx::utils::i64;

template <int Dimensions, typename Function>
decltype(auto) dispatch_float(
    py::array array, const char* name, Function&& function
) {
    return frxx::pybind::dispatch_array<float, double, Dimensions>(
        array, name, " must have dtype float32 or float64",
        std::forward<Function>(function));
}

py::object calc_variance_py(py::array field, i64 points) {
    return dispatch_float<2>(field, "field", [&](auto typed_field, auto) {
        auto eigen_field = frxx::pybind::map_const_matrix(typed_field);
        decltype(native::calc_variance(eigen_field, points)) result;
        {
            py::gil_scoped_release release;
            result = native::calc_variance(eigen_field, points);
        }
        return py::cast(std::move(result));
    });
}

template <typename T>
py::object membership_fn_line_typed(
    py::array values, T x1, T x2, i64 sign
) {
    if (values.ndim() == 1) {
        auto typed_values = frxx::pybind::require_array<T, 1>(values, "x");
        auto eigen_values = frxx::pybind::map_const_vector(typed_values);
        frxx::eigen::ConstArray1DRef<T> eigen_ref(eigen_values);
        frxx::eigen::Array1D<T> result;
        {
            py::gil_scoped_release release;
            result = native::membership_fn_line(eigen_ref, x1, x2, sign);
        }
        return py::cast(std::move(result));
    }
    if (values.ndim() == 2) {
        auto typed_values = frxx::pybind::require_array<T, 2>(values, "x");
        auto eigen_values = frxx::pybind::map_const_matrix(typed_values);
        frxx::eigen::ConstArray2DRef<T> eigen_ref(eigen_values);
        frxx::eigen::Array2D<T> result;
        {
            py::gil_scoped_release release;
            result = native::membership_fn_line(eigen_ref, x1, x2, sign);
        }
        return py::cast(std::move(result));
    }
    throw py::value_error("x must have 1 or 2 dimensions");
}

py::object membership_fn_line_py(
    py::array values, py::object x1, py::object x2, i64 sign
) {
    if (values.dtype().is(py::dtype::of<float>())) {
        return membership_fn_line_typed<float>(
            values, x1.cast<float>(), x2.cast<float>(), sign);
    }
    if (values.dtype().is(py::dtype::of<double>())) {
        return membership_fn_line_typed<double>(
            values, x1.cast<double>(), x2.cast<double>(), sign);
    }
    throw py::type_error("x must have dtype float32 or float64");
}

py::object membership_py(py::array values, i64 scatterer_class, i64 field) {
    return dispatch_float<2>(values, "x", [&](auto typed_values, auto) {
        auto eigen_values = frxx::pybind::map_const_matrix(typed_values);
        decltype(native::membership(eigen_values, scatterer_class, field)) result;
        {
            py::gil_scoped_release release;
            result = native::membership(eigen_values, scatterer_class, field);
        }
        return py::cast(std::move(result));
    });
}

py::tuple calc_aggregation_py(
    py::array zdr,
    py::array rhohv,
    py::array zdr_variance,
    py::array rhohv_variance,
    py::array psd,
    py::object filter_strength
) {
    return dispatch_float<2>(zdr, "sZDR", [&](auto typed_zdr, auto tag) {
        using T = typename decltype(tag)::type;
        auto typed_rhohv = frxx::pybind::require_array<T, 2>(rhohv, "sRHOHV");
        auto typed_zdrv = frxx::pybind::require_array<T, 2>(
            zdr_variance, "sZDRv");
        auto typed_rhohvv = frxx::pybind::require_array<T, 2>(
            rhohv_variance, "sRHOHVv");
        auto typed_psd = frxx::pybind::require_array<T, 2>(psd, "PSDH");
        auto eigen_zdr = frxx::pybind::map_const_matrix(typed_zdr);
        auto eigen_rhohv = frxx::pybind::map_const_matrix(typed_rhohv);
        auto eigen_zdrv = frxx::pybind::map_const_matrix(typed_zdrv);
        auto eigen_rhohvv = frxx::pybind::map_const_matrix(typed_rhohvv);
        auto eigen_psd = frxx::pybind::map_const_matrix(typed_psd);
        const T typed_filter_strength = filter_strength.cast<T>();
        native::AggregationResult<T> result;
        {
            py::gil_scoped_release release;
            result = native::calc_aggregation(
                eigen_zdr, eigen_rhohv, eigen_zdrv, eigen_rhohvv, eigen_psd,
                typed_filter_strength);
        }
        return py::make_tuple(
            py::cast(std::move(result.rain)),
            py::cast(std::move(result.normalized_rain)),
            py::cast(std::move(result.filtered_psd)));
    });
}

py::tuple process_ray_s_py(
    py::array psd,
    py::array zdr,
    py::array rhohv,
    i64 points,
    py::object filter_strength
) {
    return dispatch_float<2>(psd, "PSDH", [&](auto typed_psd, auto tag) {
        using T = typename decltype(tag)::type;
        auto typed_zdr = frxx::pybind::require_array<T, 2>(zdr, "sZDR");
        auto typed_rhohv = frxx::pybind::require_array<T, 2>(rhohv, "sRHOHV");
        auto eigen_psd = frxx::pybind::map_const_matrix(typed_psd);
        auto eigen_zdr = frxx::pybind::map_const_matrix(typed_zdr);
        auto eigen_rhohv = frxx::pybind::map_const_matrix(typed_rhohv);
        const T typed_filter_strength = filter_strength.cast<T>();
        native::SpectralRayResult<T> result;
        {
            py::gil_scoped_release release;
            result = native::process_ray_s(
                eigen_psd, eigen_zdr, eigen_rhohv, points,
                typed_filter_strength);
        }
        return py::make_tuple(
            py::cast(std::move(result.zdr_variance)),
            py::cast(std::move(result.rhohv_variance)),
            py::cast(std::move(result.rain)),
            py::cast(std::move(result.normalized_rain)),
            py::cast(std::move(result.filtered_psd)));
    });
}

py::object db_to_linear_2d_py(py::array values) {
    return dispatch_float<2>(values, "arr", [&](auto typed_values, auto) {
        auto eigen_values = frxx::pybind::map_const_matrix(typed_values);
        decltype(native::db_to_linear_2d(eigen_values)) result;
        {
            py::gil_scoped_release release;
            result = native::db_to_linear_2d(eigen_values);
        }
        return py::cast(std::move(result));
    });
}

py::tuple process_ray_m_py(
    py::array filtered_psd,
    py::array psd,
    py::array acf_velocity,
    py::object nyquist_velocity,
    bool flip_velocity
) {
    return dispatch_float<2>(filtered_psd, "PSDHFdb", [&](auto typed_filtered, auto tag) {
        using T = typename decltype(tag)::type;
        auto typed_psd = frxx::pybind::require_array<T, 2>(psd, "PSDHdb");
        auto typed_acf = frxx::pybind::require_array<T, 1>(acf_velocity, "vACF");
        auto eigen_filtered = frxx::pybind::map_const_matrix(typed_filtered);
        auto eigen_psd = frxx::pybind::map_const_matrix(typed_psd);
        auto eigen_acf = frxx::pybind::map_const_vector(typed_acf);
        const T typed_nyquist_velocity = nyquist_velocity.cast<T>();
        native::MomentRayResult<T> result;
        {
            py::gil_scoped_release release;
            result = native::process_ray_m(
                eigen_filtered, eigen_psd, eigen_acf,
                typed_nyquist_velocity, flip_velocity);
        }
        return py::make_tuple(
            py::cast(std::move(result.velocity)),
            py::cast(std::move(result.correction)));
    });
}

}  // namespace

PYBIND11_MODULE(_fuzzyDCA, module) {
    module.def("calcVariance", &calc_variance_py,
        py::arg("field"), py::arg("pts") = 9);
    module.def("_membershipFnLine", &membership_fn_line_py,
        py::arg("x"), py::arg("x1"), py::arg("x2"), py::arg("sign"));
    module.def("_membership", &membership_py,
        py::arg("x"), py::arg("scattererClass"), py::arg("field"));
    module.def("calcAggregation", &calc_aggregation_py,
        py::arg("sZDR"), py::arg("sRHOHV"), py::arg("sZDRv"),
        py::arg("sRHOHVv"), py::arg("PSDH"), py::arg("filterStrength") = 8.0);
    module.def("processRay_S", &process_ray_s_py,
        py::arg("PSDH"), py::arg("sZDR"), py::arg("sRHOHV"),
        py::arg("pts") = 9, py::arg("filterStrength") = 8.0);
    module.def("db_to_linear_2d", &db_to_linear_2d_py, py::arg("arr"));
    module.def("processRay_M", &process_ray_m_py,
        py::arg("PSDHFdb"), py::arg("PSDHdb"), py::arg("vACF"),
        py::arg("va"), py::arg("flipVel"));
}
