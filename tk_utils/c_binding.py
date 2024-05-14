import ctypes
from numpy.ctypeslib import ndpointer

import os
path_to_dll = os.path.join(os.getcwd(), 'OpenCLExtension.dll')

clib = ctypes.WinDLL (path_to_dll)
c_convolveNMatches = clib.convolveNMatches
c_convolveNMatches.restype = ctypes.c_int
c_convolveNMatches.argtypes = [
    ctypes.c_void_p,
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, # short_arr, len(short_arr)
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, #long_arr, len(long_arr)
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, #scores, len(scores)
]