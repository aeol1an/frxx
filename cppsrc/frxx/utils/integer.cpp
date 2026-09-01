#include <frxx/utils/integer.hpp>

#include <stdexcept>

namespace frxx::utils {

i64 unwrap_i64(std::optional<i64> value, i64 default_value) {
    return value.value_or(default_value);
}

i64 floor_div(i64 value, i64 divisor) {
    if (divisor == 0) {
        throw std::invalid_argument("floor_div divisor must not be zero");
    }
    i64 quotient = value / divisor;
    const i64 remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0))) {
        --quotient;
    }
    return quotient;
}

}  // namespace frxx::utils
