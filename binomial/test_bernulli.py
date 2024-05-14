import math
import numba as nb, numpy as np
from functools import reduce
from scipy.stats import binom
from numba import prange

@nb.jit(boundscheck=False, nopython=True, parallel=True, fastmath=True)
def _convolve_nmatches(arr1, arr2):
    sub_arr, arr = arr1, arr2
    if len(arr1) > len(arr2):
        sub_arr, arr = sub_arr, arr
    n_matches = np.zeros(len(arr) - len(sub_arr) + 1, dtype=np.float32)
    for i in prange(len(n_matches)):
        n_matches[i] = (arr[i:i+len(sub_arr)] == sub_arr).sum()
    return n_matches

def convolve_binom_and_matches(arr1, arr2, p):
    k_matches = _convolve_nmatches(arr1, arr2)
    probs = binom.pmf(k_matches, min(len(arr1), len(arr2)), p)
    return probs, k_matches

# arr1 = np.array([0, 1, 1, 0])
# arr2 = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
# probs, matches = convolve_binom_and_matches(arr1, arr2, 1/2)
# import pprint
# pprint.pprint(list(map(tuple, zip(matches, probs))))

import time
start = time.time()
arr1 = np.random.randint(0, 20, 10000)
arr2 = np.random.randint(0, 20, 10000000)

print('size 1', arr1.nbytes/(2**20))
print('size 2', arr2.nbytes/(2**20))

ks = _convolve_nmatches(arr1, arr2)
end = time.time()
print('elapsed', end - start)
print(len(ks))
probs = binom.pmf(ks, min(len(arr1), len(arr2)), 1/20)
end2 = time.time()
print('elapsed binom', end2 - end)
print(probs)
from scipy.signal import argrelextrema

start = time.time()
# for local maxima
maxs = argrelextrema(probs, np.greater)[0]

# for local minima
mins = argrelextrema(probs, np.less)[0]
end = time.time()
print('time maxs/mins', end - start)
print(len(maxs), maxs)
print(len(mins), mins)