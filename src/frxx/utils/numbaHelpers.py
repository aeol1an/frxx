import numpy as np
from numba import njit

from typing import List
from numpy.typing import NDArray

from numba.typed import List as ListType

@njit('int64(optional(int64), int64)', cache=True, nogil=True)
def unwrap_i64(opt, default):
    if opt is not None:
        return opt
    return default

@njit(
    [
        'float32[:](float32[:,:], boolean[:,:])',
        'float64[:](float64[:,:], boolean[:,:])',
    ],
    cache=True, nogil=True
)
def get_masked_float2d(arr, mask):
    n = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if mask[i, j]:
                n += 1
    out = np.empty(n, dtype=arr.dtype)
    idx = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if mask[i, j]:
                out[idx] = arr[i, j]
                idx += 1
    return out

@njit(
    [
        'void(float32[:,:], boolean[:,:], float32)',
        'void(float64[:,:], boolean[:,:], float64)',
    ],
    cache=True, nogil=True
)
def set_masked_float2d_scalar(arr, mask, val):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if mask[i, j]:
                arr[i, j] = val

@njit(
    [
        'void(float32[:,:], boolean[:,:], float32[:])',
        'void(float64[:,:], boolean[:,:], float64[:])',
    ],
    cache=True, nogil=True
)
def set_masked_float2d_array(arr, mask, val):
    idx = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if mask[i, j]:
                arr[i, j] = val[idx]
                idx += 1

@njit(
    [
        'int64(float32[:])',
        'int64(float64[:])'
    ], 
    cache=True, nogil=True
)
def nanargmax(arr):
    idx, found = 0, False
    for i in range(len(arr)):
        if arr[i] == arr[i] and (not found or arr[i] > arr[idx]):
            idx, found = i, True
    if not found:
        raise ValueError("All-NaN slice encountered")
    return idx

@njit(
    [
        'int64(float32[:])',
        'int64(float64[:])'
    ],
    cache=True, nogil=True
)
def nanargmin(arr):
    idx, found = 0, False
    for i in range(len(arr)):
        if arr[i] == arr[i] and (not found or arr[i] < arr[idx]):
            idx, found = i, True
    if not found:
        raise ValueError("All-NaN slice encountered")
    return idx

def toNumbaList(list: List[NDArray]):
    ret: List = ListType() # type: ignore
    for l in list:
        ret.append(np.asarray(l))
    return ret