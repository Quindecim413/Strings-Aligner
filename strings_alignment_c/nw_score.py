from .c_binding import *
from .PCQ import PCQ
import numpy as np


def nw_score_cpu(a, b, S, d):
    # print('nw_score_cpu')
    assert a.dtype == np.int32
    assert b.dtype == np.int32
    assert S.dtype == np.int32
    S_plain = S.ravel()
    scores = np.empty(len(b) + 1, dtype=np.int32)
    c_nw_score_cpu(a, len(a), b, len(b),
        S_plain, len(S), np.int32(d), scores)
    # print('nw_score_cpu DONE')
    return scores


def nw_score_gpu(a, b, S, d, pcq: PCQ):
    # print('nw_score_gpu')
    assert a.dtype == np.int32
    assert b.dtype == np.int32
    assert S.dtype == np.int32
    S_plain = S.ravel()
    scores = np.empty(len(b) + 1, dtype=np.int32)
    err = c_nw_score_gpu(pcq.c_pcq, a, len(a), b, len(b),
        S_plain, len(S), np.int32(d), scores)
    if err:
        raise Exception("nw_score_gpu return error code " + str(err))
    # print('nw_score_gpu DONE')
    return scores


def nw_score_adaptive(a, b, S, d, pcq: PCQ):
    from . import get_options
    NW_SCORE_MINLEN_FOR_GPU = get_options().NW_SCORE_MINLEN_FOR_GPU
    if len(a) >= NW_SCORE_MINLEN_FOR_GPU and len(b) >= NW_SCORE_MINLEN_FOR_GPU:
        return nw_score_gpu(a, b, S, d, pcq)
    else:
        return nw_score_cpu(a, b, S, d)