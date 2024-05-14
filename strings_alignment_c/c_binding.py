import ctypes
from numpy.ctypeslib import ndpointer

import os
path_to_dll = os.path.join(os.getcwd(), 'OpenCLExtension.dll')

clib = ctypes.WinDLL (path_to_dll)

c_createPCQ = clib.createPCQ
c_createPCQ.restype = ctypes.c_void_p
c_createPCQ.argtypes = [ctypes.c_int]

c_buildPCQ = clib.buildPCQ
c_buildPCQ.restype = ctypes.c_int
c_buildPCQ.argtypes = [ctypes.c_void_p]

c_deletePCQ = clib.deletePCQ
c_deletePCQ.restype = None
c_deletePCQ.argtypes = [ctypes.c_void_p]

c_nw_score_cpu = clib.NWScoreCPU
c_nw_score_cpu.restype = None
c_nw_score_cpu.argtypes = [
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, # a, len(a)
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, #b, len(b)
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, #S, len(s)
    ctypes.c_int, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS") # d, scores
]

c_nw_score_gpu = clib.NWScoreGPU
c_nw_score_gpu.restype = ctypes.c_int
c_nw_score_gpu.argtypes = [
    ctypes.c_void_p, # PCQ pointer
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, # a, len(a)
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, #b, len(b)
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, #S, len(s)
    ctypes.c_int, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS") # d, scores
]

c_nw_align_cpu = clib.NWAlignCPU
c_nw_align_cpu.restype = ctypes.c_int
c_nw_align_cpu.argtypes = [
    ctypes.c_int32, # ident
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, # a, len(a)
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, #b, len(b)
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, #S, len(s)
    ctypes.c_int, # d
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # a_aligned
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS") # b_aligned
]

c_nw_align_gpu = clib.NWAlignGPU
c_nw_align_gpu.restype = ctypes.c_int
c_nw_align_gpu.argtypes = [
    # ctypes.c_int32, # ident
    ctypes.c_void_p, # PCQ pointer
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, # a, len(a)
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, #b, len(b)
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_size_t, #S, len(s)
    ctypes.c_int, # d
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # a_aligned
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), # b_aligned
    # ndpointer(ctypes.c_int, flags="C_CONTIGUOUS") #F
]

c_getPlatformNames = clib.getPlatformNames
c_getPlatformNames.restype = ctypes.c_char_p
c_getPlatformNames.argtypes = []

c_free_chars = clib.freeChars
c_free_chars.restype = None
c_free_chars.argtypes = [ctypes.c_char_p]
