from numba import njit

@njit('int64(optional(int64), int64)', cache=True)
def unwrap_i64(opt, default):
    if opt is not None:
        return opt
    return default