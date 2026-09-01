from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    FloatArray = NDArray[np.float32] | NDArray[np.float64]
    BoolArray = NDArray[np.bool_]

    def unwrap_i64(opt: int | None, default: int) -> int: ...
    def get_masked_float2d(arr: FloatArray, mask: BoolArray) -> FloatArray: ...
    def set_masked_float2d_scalar(
        arr: FloatArray, mask: BoolArray, val: float
    ) -> None: ...
    def set_masked_float2d_array(
        arr: FloatArray, mask: BoolArray, val: FloatArray
    ) -> None: ...
    def nanargmax(arr: FloatArray) -> int: ...
    def nanargmin(arr: FloatArray) -> int: ...
else:
    from ._numbaHelpers import (
        get_masked_float2d,
        nanargmax,
        nanargmin,
        set_masked_float2d_array,
        set_masked_float2d_scalar,
        unwrap_i64,
    )
