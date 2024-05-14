from .nw_score import nw_score_adaptive, nw_score_cpu
from .nw_align import nw_align_adaptive, nw_align_cpu

import numpy as np


def process_adaptive(a, b, S, d, pcq):
    return hirschberg_adaptive(a, b, S, d, pcq)

def process_cpu(a, b, S, d):
    return hirschberg_cpu(a, b, S, d)

def hirschberg_adaptive(a, b, S, d, pcq):
    from . import get_options
    HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE = get_options().HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE
    if len(a) == 0:
        z = np.full(len(b), -1)
        w = b
    elif len(b) == 0:
        z = a
        w = np.full(len(a), -1)
    elif len(a) * len(b) <= HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE:
        z, w = nw_align_adaptive(a, b, S, d, pcq)
    else:
        a_l, a_r = a[:len(a)//2], a[len(a)//2:]
        score_l = nw_score_adaptive(a_l, b, S, d, pcq)
        score_r = nw_score_adaptive(a_r[::-1].copy(), b[::-1].copy(), S, d, pcq)
        print('L:', score_l)
        print('R:', score_r)
        
        y_mid = np.argmax(score_l + score_r[::-1])

        z_l, w_l = hirschberg_adaptive(a_l, b[:y_mid], S, d, pcq)
        z_r, w_r = hirschberg_adaptive(a_r, b[y_mid:], S, d, pcq)
        z = np.concatenate((z_l, z_r))
        w = np.concatenate((w_l, w_r))
        
    return z, w


def hirschberg_cpu(a, b, S, d):
    from . import get_options
    HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE = get_options().HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE
    
    if len(a) == 0:
        # print("DUMMY a")
        z = np.full(len(b), -1)
        w = b
    elif len(b) == 0:
        z = a
        w = np.full(len(a), -1)
    elif len(a) * len(b) <= HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE:
        z, w = nw_align_cpu(a, b, S, d)
    else:
        a_l, a_r = a[:len(a)//2], a[len(a)//2:]
        score_l = nw_score_cpu(a_l, b, S, d)
        score_r = nw_score_cpu(a_r[::-1].copy(), b[::-1].copy(), S, d)
        y_mid = np.argmax(score_l + score_r[::-1])

        z_l, w_l = hirschberg_cpu(a_l, b[:y_mid], S, d)
        z_r, w_r = hirschberg_cpu(a_r, b[y_mid:], S, d)
        z = np.concatenate((z_l, z_r))
        w = np.concatenate((w_l, w_r))

    return z, w