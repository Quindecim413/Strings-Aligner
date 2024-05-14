from .nw_score import nw_score_adaptive
from .nw_align import nw_align_adaptive

import numpy as np
import gc

from .concurrent_jobs_executor import ConcurrentJobsExecutor

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])
    
def process_results_tree(results_tree):
    res = flatten(results_tree)
    a_s = np.concatenate(list(map(lambda el: el[0], res)))
    b_s = np.concatenate(list(map(lambda el: el[1], res)))
    return a_s, b_s

def hirschberg_adaptive(a, b, S, d, tree_node, pcq, executor: ConcurrentJobsExecutor, rng=None):
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
        z_w = nw_align_adaptive(a, b, S, d, pcq, rng)
        tree_node.append(z_w)
    else:
        a_l, a_r = a[:len(a)//2] , a[len(a)//2:]
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
        
        executor.execute_subtask(nw_score_adaptive, done_l, a_l, b, S, d, pcq)
        executor.execute_subtask(nw_score_adaptive, done_r, a_r[::-1].copy(), b[::-1].copy(), S, d, pcq)
        
        def next():
            y_mid = np.argmax(score_l + score_r[::-1])
            node_l = []
            node_r = []
            tree_node[:] = [node_l, node_r]

            executor.execute_subtask(hirschberg_adaptive, None, a_l, b[:y_mid], S, d, node_l, pcq, executor)
            executor.execute_subtask(hirschberg_adaptive, None, a_r, b[y_mid:], S, d, node_r, pcq, executor)
            gc.collect()
