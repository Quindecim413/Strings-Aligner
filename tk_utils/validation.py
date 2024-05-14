import pandas as pd
import numpy as np

def check_two_arrays_simbols(arr1, arr2):
    set_arr1 = set(arr1)
    set_arr2 = set(arr2)
    return set_arr1 == set_arr2, set_arr1.difference(set_arr2), set_arr2.difference(set_arr1)

def check_arrshort_and_arrlong_length(len_arr_short, len_arr_long):
    return len_arr_short <= len_arr_long

def check_smat_simbols_lengths(smat_simbols):
    return all(map(lambda el: len(str(el))==1, smat_simbols))