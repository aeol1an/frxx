from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]

_membershipThresholds = {
    "rain": {
        "ZDR": ("full", (-1.5, 1.0, 2.0, 4.0)),
        "rhoHV": ("right", (0.79, 0.98)),
        "ZDRvar": ("left", (0.6, 5.0)),
        "rhoHVvar": ("left", (0.00025, 0.027)),
    },
    "debris": {
        "ZDR": ("full", (-19.0, -7.4, 1.7, 10.6)),
        "rhoHV": ("full", (0.0, 0.3, 0.94, 0.99)),
        "ZDRvar": ("right", (0.4, 7.1)),
        "rhoHVvar": ("right", (0.0001, 0.027)),
    },
}

if TYPE_CHECKING:
    def calcVariance(field: FloatArray, pts: int = 9) -> FloatArray: ...
    def _membershipFnLine(
        x: FloatArray, x1: float, x2: float, sign: int
    ) -> FloatArray: ...
    def _membership(
        x: FloatArray, scattererClass: int, field: int
    ) -> FloatArray: ...
    def calcAggregation(
        sZDR: FloatArray,
        sRHOHV: FloatArray,
        sZDRv: FloatArray,
        sRHOHVv: FloatArray,
        PSDH: FloatArray,
        filterStrength: float = 8.0,
    ) -> tuple[FloatArray, FloatArray, FloatArray]: ...
    def processRay_S(
        PSDH: FloatArray,
        sZDR: FloatArray,
        sRHOHV: FloatArray,
        pts: int = 9,
        filterStrength: float = 8.0,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]: ...
    def db_to_linear_2d(arr: FloatArray) -> NDArray[np.float64]: ...
    def processRay_M(
        PSDHFdb: FloatArray,
        PSDHdb: FloatArray,
        vACF: FloatArray,
        va: float,
        flipVel: bool,
    ) -> tuple[FloatArray, FloatArray]: ...
else:
    from ._fuzzyDCA import (
        _membership,
        _membershipFnLine,
        calcAggregation,
        calcVariance,
        db_to_linear_2d,
        processRay_M,
        processRay_S,
    )
