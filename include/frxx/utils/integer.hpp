#pragma once

#include <cstdint>
#include <optional>

namespace frxx::utils {

using i64 = std::int64_t;

/// Return `value` when present, otherwise return `default_value`.
///
/// @param value Optional signed 64-bit integer.
/// @param default_value Value used when `value` is empty.
i64 unwrap_i64(std::optional<i64> value, i64 default_value);

/// Divide two signed integers and round the result toward negative infinity.
///
/// @param value Dividend.
/// @param divisor Non-zero divisor.
/// @return The mathematical floor of `value / divisor`.
i64 floor_div(i64 value, i64 divisor);

}  // namespace frxx::utils
