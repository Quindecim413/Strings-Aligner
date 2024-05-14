import threading
from .c_binding import *
import numpy as np
from .PCQ import PCQ


def nw_align_cpu(a, b, S, d):
    # print('nw_align_cpu')
    assert a.dtype == np.int32
    assert b.dtype == np.int32
    assert S.dtype == np.int32
    S_plain = S.ravel().copy()
    a_align_buff = np.empty(len(a) + len(b), dtype=np.int32)
    b_align_buff = np.empty(len(a) + len(b), dtype=np.int32)
    len_a = len(a)
    len_b = len(b)
    width_s = len(S)
    d = np.int32(d)

    ident = np.int32(threading.get_ident())
    try:
        used_buff_len = c_nw_align_cpu(ident, a, len_a, b, len_b,
            S_plain, width_s, d,
            a_align_buff, b_align_buff
        )
        
    except Exception as e:
        np.save('a_np.pkl', a, allow_pickle=True)
        np.save('b_np.pkl', b, allow_pickle=True)
        print('ident', ident)
        print(hex(id(a)), a.__array_interface__['data'], 'a', a, 'len', len(a))
        print(hex(id(b)), b.__array_interface__['data'], 'b', b, 'len', len(b))
        print(hex(id(a_align_buff)), a_align_buff.__array_interface__['data'], 'a buff', a_align_buff, 'len', len(a_align_buff))
        print(hex(id(b_align_buff)), b_align_buff.__array_interface__['data'], 'b buff', b_align_buff, 'len', len(b_align_buff))
        print(hex(id(S_plain)), 'S_plain', S_plain)
        print(hex(id(d)), 'd', d)
        
        raise e
    # print('nw_align_cpu DONE')
    a_align_buff = a_align_buff[:used_buff_len][::-1]
    b_align_buff = b_align_buff[:used_buff_len][::-1]
    return a_align_buff, b_align_buff


ind = 0
def nw_align_gpu(a, b, S, d, pcq: PCQ, rng):
    # print('nw_align_gpu')
    global ind
    assert a.dtype == np.int32
    assert b.dtype == np.int32
    assert S.dtype == np.int32
    S_plain = S.ravel()
    a_align_buff = np.empty(len(a) + len(b), dtype=np.int32)
    b_align_buff = np.empty(len(a) + len(b), dtype=np.int32)

    # ident = np.int32(threading.get_ident())
    
    # # F_plain = np.empty((len(a)+1) * (len(b)+1), dtype=np.int32)
    used_buff_len_or_err = c_nw_align_gpu(pcq.c_pcq, a, len(a), b, len(b),
        S_plain, len(S), np.int32(d),
        a_align_buff, b_align_buff
    )
    
    

    if used_buff_len_or_err < 0:
        raise Exception("nw_align_gpu return error code " + str(used_buff_len_or_err))
    # print('nw_align_gpu DONE')
    a_align_buff = a_align_buff[:used_buff_len_or_err][::-1]
    b_align_buff = b_align_buff[:used_buff_len_or_err][::-1]
    
    # array([68691, 74691], dtype=int64)
    if rng[0] == 68691 and rng[1] == 74691:
        sub = ' 1thr'
        np.save('rng'+sub, rng, allow_pickle=True)
        np.save('a'+sub, a, allow_pickle=True)
        np.save('b'+sub, b, allow_pickle=True)
        np.save('a_al'+sub, a_align_buff, allow_pickle=True)
        np.save('b_al'+sub, b_align_buff, allow_pickle=True)
        np.save('S_plain'+sub, S_plain)
        # np.save('F'+sub, F_plain.reshape(-1, len(b)+1), allow_pickle=True)
    ind += 1

    return a_align_buff, b_align_buff


def nw_align_adaptive(a, b, S, d, pcq, rng):
    # from . import get_options
    # NW_ALIGN_MINLEN_FOR_GPU = get_options().NW_ALIGN_MINLEN_FOR_GPU
    # if len(a) >= NW_ALIGN_MINLEN_FOR_GPU and len(b) >= NW_ALIGN_MINLEN_FOR_GPU:
        # return nw_align_gpu(a, b, S, d, pcq, rng)
    # else:
        return nw_align_cpu(a, b, S, d)