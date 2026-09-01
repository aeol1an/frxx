#pragma once

#include <cstdint>
#include <optional>
#include <type_traits>

namespace frxx::utils {

using i64 = std::int64_t;

inline i64 unwrap_i64(std::optional<i64> value, i64 default_value) {
    return value.value_or(default_value);
}

template <typename Dividend, typename Divisor>
constexpr auto floor_div(Dividend value, Divisor divisor)
    -> std::common_type_t<Dividend, Divisor> {
    static_assert(
        std::is_integral_v<Dividend> && std::is_integral_v<Divisor>,
        "floor_div requires integer types");
    using Integer = std::common_type_t<Dividend, Divisor>;
    const Integer numerator = static_cast<Integer>(value);
    const Integer denominator = static_cast<Integer>(divisor);
    Integer quotient = numerator / denominator;
    const Integer remainder = numerator % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0))) {
        --quotient;
    }
    return quotient;
}

}  // namespace frxx::utils
