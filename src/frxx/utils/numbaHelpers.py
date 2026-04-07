import numpy as np
from numba import njit

@njit('int64(optional(int64), int64)', cache=True)
def unwrap_i64(opt, default):
    if opt is not None:
        return opt
    return default

@njit(
    [
        'float32[:](float32[:,:], boolean[:,:])', 
        'float32[:](float32[:,:], boolean[:,:])',

        'float64[:](float64[:,:], boolean[:,:])', 
        'float64[:](float64[:,:], boolean[:,:])'
    ], 
    inline='always', cache=True
)
def get_masked_float2d(arr, mask):
    a = arr.ravel()
    m = mask.ravel()
    return a[m]

@njit(
    [
        'void(float32[:,:], boolean[:,:], float32[:])',
        'void(float32[:,:], boolean[:,:], float32)',

        'void(float64[:,:], boolean[:,:], float64[:])', 
        'void(float64[:,:], boolean[:,:], float64)'
    ], 
    inline='always', cache=True
)
def set_masked_float2d(arr, mask, val):
    a = arr.ravel()
    m = mask.ravel()
    a[m] = val

@njit(
    [
        'int64(float32[:])',
        'int64(float64[:])'
    ], 
    inline='always', cache=True
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
    inline='always', cache=True
)
def nanargmin(arr):
    idx, found = 0, False
    for i in range(len(arr)):
        if arr[i] == arr[i] and (not found or arr[i] < arr[idx]):
            idx, found = i, True
    if not found:
        raise ValueError("All-NaN slice encountered")
    return idx