import threading
from strings_alignment_c.PCQ import PCQ

from strings_alignment_c.concurrent_jobs_executor import ConcurrentJobsExecutor
from strings_alignment_c.hirschberg_multithreaded2 import hirschberg_adaptive, process_results_tree
from scipy.stats import binom
import numpy as np
import pandas as pd
import os

from .c_binding import c_convolveNMatches
from .progress_counter import ProgressCounter

def convolve_nmatches(arr1, arr2, selected_platform_ind):
    sub_arr, arr = arr1, arr2
    if len(arr1) > len(arr2):
        sub_arr, arr = sub_arr, arr
    n_matches = np.empty(len(arr) - len(sub_arr) + 1, dtype=np.int32)
    with PCQ(selected_platform_ind) as pcq:
        err = c_convolveNMatches(pcq.c_pcq, sub_arr, len(sub_arr), arr, len(arr), n_matches, len(n_matches))
        if err:
            raise Exception("convolve_nmatches return error code " + str(err))
    return n_matches



def get_binomial_pmf(n, p):
    assert 0 <= p <= 1
    assert n > 0
    prob = binom.pmf(np.arange(n+1), n, p)
    return prob

def get_binomial_cdf(n, p):
    assert 0 <= p <= 1
    assert n > 0
    prob = binom.cdf(np.arange(n+1), n, p)
    return prob

def get_cutoff_val(n, p, threshold_p):
    cdf = get_binomial_cdf(n, p)
    n_min = np.absolute(cdf - threshold_p).argmin()
    return n_min


def organize_smat_rows_and_columns(smat):
    smat = smat.copy()
    smat.index = list(map(lambda el: str(el).upper(), smat.index))
    smat.columns = list(map(lambda el: str(el).upper(), smat.columns))
    cols = smat.columns
    cols_upper = list(map(lambda el: el.upper(), smat.columns))
    res = [smat.loc[col] for col in cols]
    return pd.DataFrame(res, index=cols_upper, columns=cols_upper)


def get_process_ranges_for_maxs(nmatches, cutoff, short_arr_size, expand_elements_one_side):
    good_points_inds = np.argwhere(nmatches >= cutoff)
    starts = good_points_inds - expand_elements_one_side
    ends = good_points_inds + short_arr_size + expand_elements_one_side
    mask = (starts >= 0) & (ends <= len(nmatches))
    starts = starts[mask].reshape(-1, 1)
    ends = ends[mask].reshape(-1, 1)
    nmatches = nmatches[good_points_inds]
    nmatches = nmatches[mask]

    return nmatches, np.hstack([starts, ends])


def encode_arr(arr, mappings):
    l_arr = list(arr)
    arr_codes = np.empty(len(l_arr), dtype=np.int32)
    step = 100000
    d_mapping = dict(mappings)
    for i in range((len(l_arr) // step) + ( 0 if len(l_arr) % step == 0 else 1)):
        start = i*step
        limit = min(len(l_arr), start+step)
        arr_codes[start:limit] = pd.Series(l_arr[start:limit]).map(d_mapping).fillna(-1).values.astype('int32')
    # arr_codes2 = pd.Series(l_arr).map(d_mapping).fillna(-1).values.astype('int32')
    # good = (arr_codes == arr_codes2).all()
    return arr_codes

def decode_arr(arr_codes, inv_mapping):
    arr_str = ''.join(pd.Series(arr_codes).map(dict(inv_mapping)).values)
    return arr_str


def get_inverse_mapping(similarity_mat):
    inv_mappings = list(enumerate(map(lambda el: el.upper(), similarity_mat.columns))) + [(-1, '-')]
    return inv_mappings

def get_forward_mapping(similatity_mat):
    inv_mapping = get_inverse_mapping(similatity_mat)
    mapping = list(map(lambda code_chr: (code_chr[1], code_chr[0]), inv_mapping))
    return mapping


def process_after_tree_completes(rng, results_tree, 
                                nmatches, n_matches_prob, short_len,
                                inv_mapping, stats_arr, 
                                save_dir, set_err_save):
    a_codes, b_codes = process_results_tree(results_tree)
    a_str = decode_arr(a_codes, inv_mapping)
    b_str = decode_arr(b_codes, inv_mapping)
    
    a_skips = (a_codes == -1).sum()
    b_skips = (b_codes == -1).sum()
    for ind_l in range(len(a_codes)):
        if a_codes[ind_l] != -1:
            break
    for ind_r in range(len(a_codes)-1, -1, -1):
        if a_codes[ind_r] != -1:
            break

    a_trimmed_len = ind_r - ind_l + 1

    stats_arr.append((rng[0], nmatches, nmatches/short_len, n_matches_prob, a_skips, b_skips, a_trimmed_len))
    # print('done', rng)
    save_file_path = os.path.join(save_dir, '{}.txt'.format(rng[0]))
    try:
        with open(save_file_path, 'w') as f:
            f.write(a_str)
            f.write('\n')
            f.write(b_str)
    except:
        set_err_save(save_file_path)
    


def process_arrays(short_arr, long_arr, similarity_mat, d_penalty,
                    process_ranges, nmatches, binom_probabilities,
                    num_threads,
                    selected_platform,
                    save_dir,
                    counter: ProgressCounter):
    num_threads = int(num_threads)
    assert num_threads > 0

    stats = []
    S = similarity_mat.values
    
    mapping = get_forward_mapping(similarity_mat)
    inv_mapping = get_inverse_mapping(similarity_mat)
    short_arr_codes, long_arr_codes = encode_arr(short_arr, mapping), encode_arr(long_arr, mapping)

    err_save = ''
    def set_err_path_save(filepath):
        nonlocal err_save
        with threading.Lock():
            if filepath is not None:
                err_save = '\n'.join(err_save,filepath)
    with PCQ(selected_platform) as pcq:
        with ConcurrentJobsExecutor(num_threads) as executor:
            for rng, nmatch, prob in zip(process_ranges, nmatches, binom_probabilities):
                result_tree = []
                # executor.create_job(lambda rng=rng: (time.sleep(1), print('TEST WAIT',rng)),
                #                     counter.count_up
                #                         )
                executor.create_job(lambda rng=rng, tree=result_tree: hirschberg_adaptive(short_arr_codes, 
                                                                            long_arr_codes[rng[0]:rng[1]], 
                                                                            S, d_penalty, 
                                                                            tree, pcq, executor, rng),
                                    lambda rng=rng, tree=result_tree, nmatch=nmatch, prob=prob: (
                                        process_after_tree_completes(rng, tree, 
                                                            nmatch, prob, len(short_arr),
                                                            inv_mapping, stats, 
                                                            save_dir, set_err_path_save),
                                        counter.count_up()
                                        ))
                if err_save:
                    return err_save
                # print(rng)
        print('write results')
        # print(stats)
        try:
            df_stats = pd.DataFrame(stats, columns=['Индекс начала свертки', 'Количество совпадений', 
                                                    'Доля совпадений', 'Биномиальная вероятность',
                                                    'К-во пропусков по малому массиву', 'К-во пропусков по участку большого массива',
                                                    'Длина короткого массива после обрезки хвостов'])
            df_stats.to_excel(os.path.join(save_dir, 'stats.xlsx'), index=False, engine='openpyxl')
        except:
            return os.path.join(save_dir, 'stats.xlsx')
