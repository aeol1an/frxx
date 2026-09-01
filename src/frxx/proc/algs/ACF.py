from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray


Complex64Array: TypeAlias = NDArray[np.complex64]
Complex128Array: TypeAlias = NDArray[np.complex128]

if TYPE_CHECKING:
    def computeRay_M(
        X1: Complex64Array,
        X2: Complex64Array,
        lag: int = 0,
    ) -> Complex128Array: ...
else:
    from ._ACF import computeRay_M
