import numpy as np, numba as nb

def num_elements_per_diagonal_line(n_rows, n_cols):
    n_rows = n_rows + 1
    n_cols = n_cols + 1
    min_of_dims = min(n_rows, n_cols)
    diff_of_dims = abs(n_rows - n_cols)
    nums_elements = list(np.arange(1, min_of_dims))
    if (repeats := diff_of_dims-1) > 0:
        nums_elements.extend(np.repeat(min_of_dims-1, repeats))
    if diff_of_dims == 0:
        nums_elements.extend(np.arange(min_of_dims - 2, 0, -1))
    else:
        nums_elements.extend(np.arange(min_of_dims - 1, 0, -1))
    return np.array(nums_elements).astype('int32')

def starts_per_diagonal_line(n_rows, n_cols):
    n_rows = n_rows + 1
    n_cols = n_cols + 1

    starts_np = np.zeros((n_rows + n_cols-3, 2), dtype=int)
    starts_np[:n_cols-1, 1] = np.arange(1, n_cols)
    starts_np[:n_cols-1, 0] = 1

    starts_np[n_cols-1:, 0] = np.arange(2, n_rows)
    starts_np[n_cols-1:, 1] = n_cols-1
    return starts_np.astype('int32')

# def process_py(F, a, b, S, d):
#     a_s = ''
#     b_s = ''
#     i = len(a)
#     j = len(b)
#     while i > 0 or j > 0:
#         score = F[i, j]
#         up = F[i - 1, j]
#         left = F[i, j - 1]
#         diag = F[i-1, j-1]
#         _a = a[i-1]
#         _b = b[j-1]
#         sim = S[_a, _b]

#         if score == diag + sim :
#             a_s = inv_mappings[_a] + a_s
#             b_s = inv_mappings[_b] + b_s
#             i -= 1
#             j -= 1
#         elif score == left + d:
#             a_s = '-' + a_s
#             b_s = inv_mappings[_b] + b_s
#             j -= 1
#         elif score == up + d:
#             a_s = inv_mappings[_a] + a_s
#             b_s = '-' + b_s
#             i -= 1
            
#     print(i, j)
#     while i > 0:
#         a_s = inv_mappings[a[i-1]] + a_s
#         b_s = '-' + b_s
#         i -= 1
#     while j > 0:
#         a_s = '-' + a_s
#         b_s = inv_mappings[b[j-1]] + b_s
#         j -= 1
#     return a_s, b_s

@nb.jit(nopython=True)
def process_align(F, a, b, S, d):
    lines_codes = np.empty((len(a) + len(b), 2), dtype='int32')
    score: int = 0
    diag: int = 0
    up: int = 0
    left: int = 0

    i: int = len(a)
    j: int = len(b)
    _a: int = 0
    _b: int = 0
    ind: int = 0

    while (i > 0) or (j > 0):
        score = F[i][j]
        diag = F[i-1][j-1]
        left = F[i][j-1]
        up = F[i-1][j]
        _a = a[i - 1]
        _b = b[j - 1]

        if score == diag + S[_a][_b]:
            lines_codes[ind][0] = _a
            lines_codes[ind][1] = _b
            i -= 1
            j -= 1
        elif score == left + d:
            lines_codes[ind][0] = -1
            lines_codes[ind][1] = _b
            j -= 1
        elif score == up + d:
            lines_codes[ind][0] = _a
            lines_codes[ind][1] = -1
            i -= 1
        ind += 1

    while i > 0:
        _a = a[i - 1]
        lines_codes[ind][0] = _a
        lines_codes[ind][1] = -1
        i -= 1
        ind += 1
    while j > 0:
        _b = b[j - 1]
        lines_codes[ind][0] = -1
        lines_codes[ind][1] = _b
        j -= 1
        ind += 1
    
    res = lines_codes[:ind][::-1]
    return res[:, 0], res[:, 1]


# 2
import numba as nb, numpy as np
from functools import reduce
from scipy.stats import binom
from numba import prange

@nb.jit(boundscheck=False, nopython=True, parallel=True)
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