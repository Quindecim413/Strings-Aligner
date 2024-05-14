from .nw_score import nw_score_adaptive, nw_score_cpu
from .nw_align import nw_align_adaptive, nw_align_cpu

import numpy as np

# Минимальное количество len(a) * len(b) чтобы рекурсионное выравнивание переключилось на алгоритм NeedlemanWunsch
HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE = 10000*10000

from .concurrent_executor import ConcurrentExecutor

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def process_adaptive(a, b, S, d, pcq, max_workers=8):
    with ConcurrentExecutor(max_workers) as executor:
        results_tree = []
        executor.create_job("hirschberg ROOT", hirschberg_adaptive, None, a, b, S, d, results_tree, pcq, executor)
    # print("EXIT")
    return process_results_tree(results_tree)
    
    
def process_results_tree(results_tree):
    res = flatten(results_tree)
    a_s = np.concatenate(list(map(lambda el: el[0], res)))
    b_s = np.concatenate(list(map(lambda el: el[1], res)))
    return a_s, b_s

def hirschberg_adaptive(a, b, S, d, tree_node, pcq, executor: ConcurrentExecutor):
    from . import get_options
    HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE = get_options().HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE
    
    if len(a) == 0:
        z = np.full(len(b), -1)
        w = b
        tree_node[:] =  [(z, w)]
    elif len(b) == 0:
        z = a
        w = np.full(len(a), -1)
        tree_node[:] =  [(z, w)]
    elif max(len(a), len(b)) <= HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE:
        z_w = nw_align_adaptive(a, b, S, d, pcq)
        tree_node.append(z_w)
    else:
        a_l, a_r = a[:len(a)//2], a[len(a)//2:]
        score_l, score_r = None, None

        def done_l(res_l):
            nonlocal score_l, score_r, a_l, a_r, b
            score_l = res_l
            if score_r is not None:
                next()
        
        def done_r(res_r):
            nonlocal score_l, score_r, a_l, a_r, b
            score_r = res_r

            if score_l is not None:
                next()
        
        executor.create_job('score L', nw_score_adaptive, done_l, a_l, b, S, d, pcq)
        executor.create_job('score R', nw_score_adaptive, done_r, a_r[::-1].copy(), b[::-1].copy(), S, d, pcq)
        
        def next():
            y_mid = np.argmax(score_l + score_r[::-1])
            node_l = []
            node_r = []
            tree_node[:] = [node_l, node_r]

            executor.create_job('hirshberg L', hirschberg_adaptive, None, a_l, b[:y_mid], S, d, node_l, pcq, executor)
            executor.create_job('hirshberg R', hirschberg_adaptive, None, a_r, b[y_mid:], S, d, node_r, pcq, executor)


# def process_cpu(a, b, S, d, max_workers=8):
#     with ConcurrentExecutor(max_workers) as executor:
#         return hirschberg_cpu(a, b, S, d, executor)
# 
# 
# def hirschberg_cpu(a, b, S, d, tree_node, executor):
#     if len(a) == 0:
#         z = np.full(len(b), -1)
#         w = b
#     elif len(b) == 0:
#         z = a
#         w = np.full(len(a), -1)
#     elif len(a) * len(b) >= HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE:
#         z, w = nw_align_cpu(a, b, S, d)
#     else:
#         a_l, a_r = a[:len(a)//2], a[len(a)//2:]
#         score_l = nw_score_cpu(a_l, b, S, d)
#         score_r = nw_score_cpu(a_r[::-1].copy(), b[::-1].copy(), S, d)
#         y_mid = np.argmax(score_l + score_r[::-1])

#         z_l, w_l = hirschberg_cpu(a_l, b[:y_mid], S, d)
#         z_r, w_r = hirschberg_cpu(a_r, b[y_mid:], S, d)
#         z = np.concatenate((z_l, z_r))
#         w = np.concatenate((w_l, w_r))

#     return z, w