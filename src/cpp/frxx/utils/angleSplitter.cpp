#include <frxx/utils/angleSplitter.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace frxx::utils {

namespace {

float positive_degrees(float value) {
    const float result = std::fmod(value, 360.0F);
    return result < 0.0F ? result + 360.0F : result;
}

float wrapped_difference(float current, float previous) {
    return std::remainder(current - previous, 360.0F);
}

}  // namespace

bool in_degree_range(double value, double low, double high) {
    return low < high ? value > low && value < high : value > low || value < high;
}

i64 trim_surveillance(frxx::eigen::ConstArray1DRef<float> angle) {
    const i64 ray_count = static_cast<i64>(angle.size());
    if (ray_count < 2) {
        return ray_count;
    }

    double direction_sum = 0.0;
    for (i64 index = 1; index < ray_count; ++index) {
        if (!std::isfinite(angle(index)) || !std::isfinite(angle(index - 1))) {
            throw std::invalid_argument("angle must contain only finite values");
        }
        direction_sum += wrapped_difference(angle(index), angle(index - 1));
    }
    if (direction_sum == 0.0) {
        return ray_count;
    }

    const float direction = direction_sum > 0.0 ? 1.0F : -1.0F;
    double rotation = 0.0;
    for (i64 index = 1; index < ray_count; ++index) {
        rotation += direction * wrapped_difference(
            angle(index), angle(index - 1));
        if (rotation >= 360.0) {
            return index;
        }
    }
    return ray_count;
}

PulseBoundaries find_pulse_boundaries(
    frxx::eigen::ConstArray1DRef<float> angle,
    float pixel_width_degrees,
    float beam_overlap_degrees
) {
    const i64 pulse_count = static_cast<i64>(angle.size());
    if (pulse_count < 1) {
        throw std::invalid_argument("angle must contain at least one pulse");
    }
    const float half_swath = 0.5F *
        (pixel_width_degrees + 2.0F * beam_overlap_degrees);
    const float angle_spacing = half_swath;

    frxx::eigen::Array1D<float> discrete(pulse_count);
    for (i64 index = 0; index < pulse_count; ++index) {
        discrete(index) = positive_degrees(
            std::nearbyint(angle(index) / angle_spacing) * angle_spacing +
            0.5F * angle_spacing);
    }

    frxx::eigen::Array1D<float> unique(pulse_count);
    i64 group_count = 1;
    unique(0) = discrete(0);
    for (i64 index = 1; index < pulse_count; ++index) {
        if (discrete(index) != discrete(index - 1)) {
            unique(group_count++) = discrete(index);
        }
    }
    unique.conservativeResize(group_count);

    frxx::eigen::Array1D<float> low(group_count);
    frxx::eigen::Array1D<float> high(group_count);
    for (i64 group = 0; group < group_count; ++group) {
        low(group) = positive_degrees(unique(group) - half_swath);
        high(group) = positive_degrees(unique(group) + half_swath);
    }

    frxx::eigen::Array2D<i64> boundaries =
        frxx::eigen::Array2D<i64>::Zero(group_count, 2);
    frxx::eigen::Array1D<std::int8_t> state =
        frxx::eigen::Array1D<std::int8_t>::Zero(group_count);
    double direction_sum = 0.0;
    for (i64 index = 1; index < pulse_count; ++index) {
        const float difference = angle(index) - angle(index - 1);
        direction_sum += difference > 0.0F ? 1.0 : difference < 0.0F ? -1.0 : 0.0;
    }
    const bool increasing = pulse_count > 1 &&
        direction_sum / static_cast<double>(pulse_count - 1) > 0.0;

    if (increasing) {
        for (i64 index = 0; index < pulse_count; ++index) {
            for (i64 group = 0; group < group_count; ++group) {
                if (state(group) == 2) {
                    continue;
                }
                const bool inside = in_degree_range(
                    angle(index), low(group), high(group));
                if (state(group) == 0 && inside) {
                    boundaries(group, 0) = index;
                    state(group) = 1;
                } else if (state(group) == 1 && !inside) {
                    boundaries(group, 1) = index - 1;
                    state(group) = 2;
                }
            }
        }
        for (i64 group = 0; group < group_count; ++group) {
            if (state(group) == 1) {
                boundaries(group, 1) = pulse_count - 1;
            }
        }
    } else {
        for (i64 index = pulse_count - 1; index >= 0; --index) {
            for (i64 group = 0; group < group_count; ++group) {
                if (state(group) == 2) {
                    continue;
                }
                const bool inside = in_degree_range(
                    angle(index), low(group), high(group));
                if (state(group) == 0 && inside) {
                    boundaries(group, 1) = index;
                    state(group) = 1;
                } else if (state(group) == 1 && !inside) {
                    boundaries(group, 0) = index + 1;
                    state(group) = 2;
                }
            }
        }
        for (i64 group = 0; group < group_count; ++group) {
            if (state(group) == 1) {
                boundaries(group, 0) = 0;
            }
        }
    }
    return {std::move(boundaries), std::move(unique)};
}

}  // namespace frxx::utils
