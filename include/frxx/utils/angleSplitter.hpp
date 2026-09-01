#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <frxx/utils/integer.hpp>

namespace frxx::utils {

inline bool in_degree_range(double value, double low, double high) {
    return low < high ? value > low && value < high : value > low || value < high;
}

inline float positive_degrees(float value) {
    float result = std::fmod(value, 360.0F);
    return result < 0.0F ? result + 360.0F : result;
}

struct PulseBoundaries {
    std::vector<i64> indices;
    std::vector<float> angles;
};

template <typename AngleArray>
PulseBoundaries find_pulse_boundaries(
    const AngleArray& angle, i64 pulse_count,
    float pixel_width_degrees, float beam_overlap_degrees
) {
    if (pulse_count < 1) {
        throw std::invalid_argument("angle must contain at least one pulse");
    }
    const float half_swath = 0.5F *
        (pixel_width_degrees + 2.0F * beam_overlap_degrees);
    const float angle_spacing = half_swath;

    std::vector<float> discrete(static_cast<std::size_t>(pulse_count));
    for (i64 index = 0; index < pulse_count; ++index) {
        discrete[static_cast<std::size_t>(index)] = positive_degrees(
            std::nearbyint(angle(index) / angle_spacing) * angle_spacing +
            0.5F * angle_spacing);
    }

    std::vector<float> unique{discrete.front()};
    for (i64 index = 1; index < pulse_count; ++index) {
        if (discrete[static_cast<std::size_t>(index)] !=
            discrete[static_cast<std::size_t>(index - 1)]) {
            unique.push_back(discrete[static_cast<std::size_t>(index)]);
        }
    }

    const i64 group_count = static_cast<i64>(unique.size());
    std::vector<float> low(static_cast<std::size_t>(group_count));
    std::vector<float> high(static_cast<std::size_t>(group_count));
    for (i64 group = 0; group < group_count; ++group) {
        low[static_cast<std::size_t>(group)] = positive_degrees(
            unique[static_cast<std::size_t>(group)] - half_swath);
        high[static_cast<std::size_t>(group)] = positive_degrees(
            unique[static_cast<std::size_t>(group)] + half_swath);
    }

    std::vector<i64> boundaries(static_cast<std::size_t>(group_count * 2), 0);
    std::vector<std::int8_t> state(static_cast<std::size_t>(group_count), 0);
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
                if (state[static_cast<std::size_t>(group)] == 2) {
                    continue;
                }
                const bool inside = in_degree_range(
                    angle(index), low[static_cast<std::size_t>(group)],
                    high[static_cast<std::size_t>(group)]);
                if (state[static_cast<std::size_t>(group)] == 0 && inside) {
                    boundaries[static_cast<std::size_t>(group * 2)] = index;
                    state[static_cast<std::size_t>(group)] = 1;
                } else if (state[static_cast<std::size_t>(group)] == 1 && !inside) {
                    boundaries[static_cast<std::size_t>(group * 2 + 1)] = index - 1;
                    state[static_cast<std::size_t>(group)] = 2;
                }
            }
        }
        for (i64 group = 0; group < group_count; ++group) {
            if (state[static_cast<std::size_t>(group)] == 1) {
                boundaries[static_cast<std::size_t>(group * 2 + 1)] = pulse_count - 1;
            }
        }
    } else {
        for (i64 index = pulse_count - 1; index >= 0; --index) {
            for (i64 group = 0; group < group_count; ++group) {
                if (state[static_cast<std::size_t>(group)] == 2) {
                    continue;
                }
                const bool inside = in_degree_range(
                    angle(index), low[static_cast<std::size_t>(group)],
                    high[static_cast<std::size_t>(group)]);
                if (state[static_cast<std::size_t>(group)] == 0 && inside) {
                    boundaries[static_cast<std::size_t>(group * 2 + 1)] = index;
                    state[static_cast<std::size_t>(group)] = 1;
                } else if (state[static_cast<std::size_t>(group)] == 1 && !inside) {
                    boundaries[static_cast<std::size_t>(group * 2)] = index + 1;
                    state[static_cast<std::size_t>(group)] = 2;
                }
            }
        }
        for (i64 group = 0; group < group_count; ++group) {
            if (state[static_cast<std::size_t>(group)] == 1) {
                boundaries[static_cast<std::size_t>(group * 2)] = 0;
            }
        }
    }
    return {std::move(boundaries), std::move(unique)};
}

}  // namespace frxx::utils
