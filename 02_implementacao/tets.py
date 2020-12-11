import concurrent.futures
import numba
import numpy as np

# @numba.njit([numba.float64(numba.float64[:])], parallel=True) # works
@numba.njit(parallel=True) # fails
def plus(arr):
    res = 0
    for ii in numba.prange(len(arr)):
        res += arr[ii]
    return res

xx = np.arange(10000).astype(float)

with concurrent.futures.ThreadPoolExecutor(2) as pool:
    for result in pool.map(plus, [xx, xx]):
        print(result)