from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    def inDegreeRange(val: float, low: float, high: float) -> bool: ...
    def findPulseBoundaries(
        angle: NDArray[np.float32],
        pixelWidthDeg: float,
        beamOverlapDeg: float,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]: ...
else:
    from ._angleSplitter import findPulseBoundaries, inDegreeRange
