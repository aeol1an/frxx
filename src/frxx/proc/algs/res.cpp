#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

#include <frxx/utils/integer.hpp>
#include <frxx/utils/pybind_numpy.hpp>

namespace py = pybind11;

namespace {

using frxx::utils::floor_div;
using frxx::utils::i64;
using frxx::utils::normalize_index;
using frxx::utils::require_array;

template <typename Complex>
void range_subset_impl(
    py::array_t<Complex> iq,
    py::array_t<Complex> result,
    i64 K,
    i64 Koffset,
    i64 NR,
    i64 start_range,
    i64 first_pulse,
    i64 last_pulse
) {
    if (!result.writeable()) {
        throw py::value_error("result must be writable");
    }
    if (K < 0 || NR < 0) {
        throw py::value_error("K and NR must be non-negative");
    }

    const i64 range_gates = static_cast<i64>(iq.shape(0));
    const i64 pulses = static_cast<i64>(iq.shape(1));
    const i64 pulse_count = last_pulse - first_pulse + 1;
    if (range_gates == 0 || first_pulse < 0 || last_pulse >= pulses || pulse_count < 0) {
        throw py::index_error("IQ subset indices are out of bounds");
    }
    if (result.shape(0) < K * NR || result.shape(1) != pulse_count) {
        throw py::value_error("result has an incompatible shape");
    }

    auto input = iq.template unchecked<2>();
    auto output = result.template mutable_unchecked<2>();
    py::gil_scoped_release release;
    for (i64 r = 0; r < NR; ++r) {
        for (i64 k = 0; k < K; ++k) {
            i64 source_range = k + r - (K / 2 - Koffset) + start_range;
            source_range = std::clamp(source_range, i64{0}, range_gates - 1);
            const i64 output_range = r * K + k;
            for (i64 pulse = 0; pulse < pulse_count; ++pulse) {
                output(output_range, pulse) = input(source_range, first_pulse + pulse);
            }
        }
    }
}

template <typename Complex>
void az_subset_impl(
    py::array_t<Complex> iq,
    py::array_t<Complex> result,
    i64 NR,
    i64 start_range,
    py::array_t<i64> first_pulses,
    py::array_t<i64> last_pulses
) {
    if (!result.writeable()) {
        throw py::value_error("result must be writable");
    }
    if (first_pulses.shape(0) != last_pulses.shape(0)) {
        throw py::value_error("fps and lps must have the same length");
    }

    const i64 azimuth_count = static_cast<i64>(first_pulses.shape(0));
    const i64 pulse_count = result.shape(1);
    if (NR < 0 || result.shape(0) < NR * azimuth_count) {
        throw py::value_error("result has an incompatible shape");
    }
    if (start_range < 0 || start_range + NR > iq.shape(0)) {
        throw py::index_error("range indices are out of bounds");
    }

    auto input = iq.template unchecked<2>();
    auto output = result.template mutable_unchecked<2>();
    auto first = first_pulses.template unchecked<1>();
    auto last = last_pulses.template unchecked<1>();
    const i64 available_pulses = static_cast<i64>(iq.shape(1));

    for (i64 azimuth = 0; azimuth < azimuth_count; ++azimuth) {
        if (first(azimuth) < 0 || last(azimuth) >= available_pulses ||
            last(azimuth) - first(azimuth) + 1 != pulse_count) {
            throw py::value_error("pulse bounds are incompatible with result");
        }
    }

    py::gil_scoped_release release;
    for (i64 r = 0; r < NR; ++r) {
        for (i64 azimuth = 0; azimuth < azimuth_count; ++azimuth) {
            const i64 output_range = r * azimuth_count + azimuth;
            for (i64 pulse = 0; pulse < pulse_count; ++pulse) {
                output(output_range, pulse) = input(
                    r + start_range, first(azimuth) + pulse);
            }
        }
    }
}

template <typename Complex>
py::tuple subset_iq_impl(
    py::array_t<Complex> iq,
    i64 iaz,
    i64 naz,
    bool az_increasing,
    py::array_t<i64> pulse_boundaries,
    py::array_t<i64> iranges,
    i64 swath_pulses,
    i64 K,
    i64 Koffset,
    i64 avg_strat,
    bool shape_only
) {
    if (K < 1) {
        throw py::value_error("K must be greater than 0.");
    }
    if (Koffset != 0 && Koffset != 1) {
        throw py::value_error("Valid values for KOffset: {0(low), 1(high)}");
    }
    if (avg_strat != 0 && avg_strat != 1) {
        throw py::value_error("Valid values for avgStrat: {0(range), 1(azimuth)}");
    }
    if (pulse_boundaries.shape(1) < 2) {
        throw py::value_error("pulseBoundaries must have at least two columns");
    }
    if (iranges.shape(0) < 2) {
        throw py::value_error("iranges must contain a start and end index");
    }

    auto boundaries = pulse_boundaries.template unchecked<2>();
    auto ranges = iranges.template unchecked<1>();
    const i64 pulse_count = static_cast<i64>(iq.shape(1));
    const i64 start_range = ranges(0);
    const i64 NR = ranges(1) + 1 - start_range;
    if (NR < 0) {
        throw py::value_error("negative dimensions are not allowed");
    }

    py::array result = py::array_t<Complex>(
        std::vector<py::ssize_t>{py::ssize_t{0}, py::ssize_t{0}});

    if (K > 1 && avg_strat == 0) {
        const i64 boundary_index = normalize_index(
            iaz, pulse_boundaries.shape(0), "iaz");
        const i64 boundary_start = boundaries(boundary_index, 0);
        const i64 boundary_end = boundaries(boundary_index, 1);
        const i64 center_pulse = boundary_start + floor_div(
            boundary_end + 1 - boundary_start, 2);
        if (center_pulse < 0 || center_pulse >= pulse_count) {
            throw py::value_error("Center pulse out of bounds.");
        }
        if (swath_pulses < 2) {
            swath_pulses = boundary_end + 1 - boundary_start;
        }

        i64 first_pulse = center_pulse - floor_div(swath_pulses, 2);
        i64 last_pulse = swath_pulses % 2 != 0
            ? center_pulse + floor_div(swath_pulses, 2)
            : center_pulse + floor_div(swath_pulses, 2) - 1;
        first_pulse = std::max(i64{0}, first_pulse);
        last_pulse = std::min(pulse_count - 1, last_pulse);
        swath_pulses = last_pulse - first_pulse + 1;

        if (!shape_only) {
            auto typed_result = py::array_t<Complex>(std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(K * NR),
                static_cast<py::ssize_t>(swath_pulses),
            });
            range_subset_impl(
                iq, typed_result, K, Koffset, NR, start_range, first_pulse, last_pulse);
            result = std::move(typed_result);
        }
    } else if (K > 1) {
        if (naz < 1) {
            throw py::value_error("naz must be greater than 0");
        }

        std::vector<i64> azimuth_indices(static_cast<std::size_t>(K));
        const i64 decreasing_shift = (K + 1) / 2 - std::abs(Koffset - 1);
        for (i64 index = 0; index < K; ++index) {
            i64 azimuth = az_increasing
                ? index - (K / 2 - Koffset) + iaz
                : (K - 1 - index) - decreasing_shift + iaz;
            azimuth_indices[static_cast<std::size_t>(index)] =
                std::clamp(azimuth, i64{0}, naz - 1);
        }

        std::vector<i64> first_pulses(static_cast<std::size_t>(K));
        std::vector<i64> last_pulses(static_cast<std::size_t>(K));
        if (swath_pulses < 2) {
            swath_pulses = std::numeric_limits<i64>::max();
        }

        for (i64 index = 0; index < K; ++index) {
            const i64 boundary_index = normalize_index(
                azimuth_indices[static_cast<std::size_t>(index)],
                pulse_boundaries.shape(0), "azimuth index");
            const i64 boundary_start = boundaries(boundary_index, 0);
            const i64 boundary_end = boundaries(boundary_index, 1);
            const i64 center_pulse = boundary_start + floor_div(
                boundary_end + 1 - boundary_start, 2);
            if (center_pulse < 0 || center_pulse >= pulse_count) {
                throw py::value_error("A center pulse is out of bounds.");
            }
            if (swath_pulses == std::numeric_limits<i64>::max()) {
                first_pulses[static_cast<std::size_t>(index)] =
                    boundary_end + 1 - boundary_start;
            }
        }

        if (swath_pulses == std::numeric_limits<i64>::max()) {
            swath_pulses = *std::min_element(first_pulses.begin(), first_pulses.end());
        }

        i64 common_pulses = std::numeric_limits<i64>::max();
        for (i64 index = 0; index < K; ++index) {
            const i64 boundary_index = azimuth_indices[static_cast<std::size_t>(index)];
            const i64 boundary_start = boundaries(boundary_index, 0);
            const i64 boundary_end = boundaries(boundary_index, 1);
            const i64 center_pulse = boundary_start + floor_div(
                boundary_end + 1 - boundary_start, 2);
            i64 first_pulse = center_pulse - floor_div(swath_pulses, 2);
            i64 last_pulse = swath_pulses % 2 != 0
                ? center_pulse + floor_div(swath_pulses, 2)
                : center_pulse + floor_div(swath_pulses, 2) - 1;
            first_pulse = std::max(i64{0}, first_pulse);
            last_pulse = std::min(pulse_count - 1, last_pulse);
            first_pulses[static_cast<std::size_t>(index)] = first_pulse;
            last_pulses[static_cast<std::size_t>(index)] = last_pulse;
            common_pulses = std::min(common_pulses, last_pulse - first_pulse + 1);
        }
        swath_pulses = common_pulses;
        for (i64 index = 0; index < K; ++index) {
            last_pulses[static_cast<std::size_t>(index)] =
                first_pulses[static_cast<std::size_t>(index)] + swath_pulses - 1;
        }

        if (!shape_only) {
            auto typed_result = py::array_t<Complex>(std::vector<py::ssize_t>{
                static_cast<py::ssize_t>(K * NR),
                static_cast<py::ssize_t>(swath_pulses),
            });
            auto first_array = py::array_t<i64>(
                static_cast<py::ssize_t>(K), first_pulses.data());
            auto last_array = py::array_t<i64>(
                static_cast<py::ssize_t>(K), last_pulses.data());
            az_subset_impl(
                iq, typed_result, NR, start_range, first_array, last_array);
            result = std::move(typed_result);
        }
    } else {
        const i64 boundary_index = normalize_index(
            iaz, pulse_boundaries.shape(0), "iaz");
        const i64 boundary_start = boundaries(boundary_index, 0);
        const i64 boundary_end = boundaries(boundary_index, 1);
        const i64 center_pulse = boundary_start + floor_div(
            boundary_end + 1 - boundary_start, 2);
        if (center_pulse < 0 || center_pulse >= pulse_count) {
            throw py::value_error("Center pulse out of bounds.");
        }
        if (swath_pulses < 2) {
            swath_pulses = boundary_end + 1 - boundary_start;
        }

        i64 first_pulse = center_pulse - floor_div(swath_pulses, 2);
        i64 last_pulse = swath_pulses % 2 != 0
            ? center_pulse + floor_div(swath_pulses, 2)
            : center_pulse + floor_div(swath_pulses, 2) - 1;
        first_pulse = std::max(i64{0}, first_pulse);
        last_pulse = std::min(pulse_count - 1, last_pulse);
        swath_pulses = last_pulse - first_pulse + 1;

        if (!shape_only) {
            result = py::cast<py::array>(iq.attr("__getitem__")(py::make_tuple(
                py::slice(start_range, start_range + NR, 1),
                py::slice(first_pulse, last_pulse + 1, 1))));
        }
    }

    return py::make_tuple(result, NR, swath_pulses);
}

void range_subset(
    py::array iq,
    py::array result,
    i64 K,
    i64 Koffset,
    i64 NR,
    i64 start_range,
    i64 first_pulse,
    i64 last_pulse
) {
    frxx::utils::dispatch_complex<2>(iq, "iq", [&](auto typed_iq, auto tag) {
        using Complex = typename decltype(tag)::type;
        auto typed_result = require_array<Complex, 2>(result, "result");
        range_subset_impl(
            typed_iq, typed_result, K, Koffset, NR, start_range, first_pulse, last_pulse);
    });
}

void az_subset(
    py::array iq,
    py::array result,
    i64 NR,
    i64 start_range,
    py::array fps,
    py::array lps
) {
    auto first_pulses = require_array<i64, 1>(fps, "fps");
    auto last_pulses = require_array<i64, 1>(lps, "lps");
    frxx::utils::dispatch_complex<2>(iq, "iq", [&](auto typed_iq, auto tag) {
        using Complex = typename decltype(tag)::type;
        auto typed_result = require_array<Complex, 2>(result, "result");
        az_subset_impl(
            typed_iq, typed_result, NR, start_range, first_pulses, last_pulses);
    });
}

py::tuple subset_iq(
    py::array iq,
    i64 iaz,
    i64 naz,
    bool az_increasing,
    py::array pulse_boundaries,
    py::array iranges,
    i64 swath_pulses,
    i64 K,
    i64 Koffset,
    i64 avg_strat,
    bool shape_only
) {
    auto boundaries = require_array<i64, 2>(pulse_boundaries, "pulseBoundaries");
    auto ranges = require_array<i64, 1>(iranges, "iranges");
    return frxx::utils::dispatch_complex<2>(iq, "iq", [&](auto typed_iq, auto) {
        return subset_iq_impl(
            typed_iq, iaz, naz, az_increasing,
            boundaries, ranges, swath_pulses, K, Koffset, avg_strat, shape_only);
    });
}

}  // namespace

PYBIND11_MODULE(_res, module) {
    module.doc() = "C++ implementations of IQ subsetting kernels.";
    module.def(
        "_rangeSubsetIQ", &range_subset,
        py::arg("iq"), py::arg("result"), py::arg("K"), py::arg("Koffset"),
        py::arg("NR"), py::arg("startRange"), py::arg("fp"), py::arg("lp"));
    module.def(
        "_azSubsetIQ", &az_subset,
        py::arg("iq"), py::arg("result"), py::arg("NR"), py::arg("startRange"),
        py::arg("fps"), py::arg("lps"));
    module.def(
        "subsetIQcpp", &subset_iq,
        py::arg("iq"), py::arg("iaz"), py::arg("naz"), py::arg("azIncreasing"),
        py::arg("pulseBoundaries"), py::arg("iranges"), py::arg("swathPulses") = -1,
        py::arg("K") = 1, py::arg("KOffset") = 0, py::arg("avgStrat") = 1,
        py::arg("shapeOnly") = false);
}
