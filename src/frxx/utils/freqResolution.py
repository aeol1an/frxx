from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    FloatArray = NDArray[np.float32] | NDArray[np.float64]

    def velResolution(
        nPulses: float, prf: float = 4000, wavelength: float = 0.0308
    ) -> float: ...
    def velResolutionTonPulses(
        delta_v: float, prf: float = 4000, wavelength: float = 0.0308
    ) -> float: ...
    def velocityAxis(
        NFT: int,
        va: np.float32 | np.float64,
        flipVel: bool,
        leftUnfolds: int = 0,
        rightUnfolds: int = 0,
    ) -> FloatArray: ...
    def velSpanToNumBins(
        delta_v: float,
        nFFT: int,
        prf: float = 4000,
        wavelength: float = 0.0308,
    ) -> int: ...
else:
    from ._freqResolution import (
        velResolution,
        velResolutionTonPulses,
        velocityAxis,
        velSpanToNumBins,
    )
