from distutils.command.config import config
from strings_alignment_c import get_options
from strings_alignment_c.PCQ import get_platform_names


def try_set_HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE(val:int):
    get_options().HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE = val
    return val == get_options().HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE

def try_set_NW_ALIGN_MINLEN_FOR_GPU(val:int):
    get_options().NW_ALIGN_MINLEN_FOR_GPU = val
    return val == get_options().NW_ALIGN_MINLEN_FOR_GPU

def try_set_NW_SCORE_MINLEN_FOR_GPU(val:int):
    get_options().NW_SCORE_MINLEN_FOR_GPU = val
    return val == get_options().NW_SCORE_MINLEN_FOR_GPU


def get_platforms():
    return get_platform_names()

def get_max_mem_usage_per_thread(alignment_size):
    # Матрица по NeddleManWunsch: F и массивы (device и host)
    # NSCORE prev curr next (device) + хранение данных по обоим массивам (device и host)
    size_int = 4

    matrix_max_size = min(get_options().HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE + 1, alignment_size + 1)

    return (matrix_max_size**2 + (alignment_size * 2) * 2) * size_int**2 +\
         (3*(alignment_size * 2 - 1) + (alignment_size * 2) * 2) * size_int

import os

CONFIG_FILE = os.path.join(os.getcwd(), 'config.json')

def load_config_from_file():
    import json
    try:
        with open(CONFIG_FILE, 'r') as f:
            conf = json.load(f)
    except:
        return 'Файл не найден'
    err = ''
    if 'HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE' in conf:
        val = conf['HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE']
        try:
            val = int(val)
            if not try_set_HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE(val):
                err = 'Значение параметра\nHIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE\nне входит в допустимый диапазон {}\nи установлено по умолчанию: {}'.format(
                    get_options().HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE_MINMAX_REPR,
                    get_options().HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE)
        except:
            err = 'Значение параметра\nHIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE\nне является целым числом'
    
    if 'NW_SCORE_MINLEN_FOR_GPU' in conf:
            val = conf['NW_SCORE_MINLEN_FOR_GPU']
            try:
                val = int(val)
                if not try_set_NW_SCORE_MINLEN_FOR_GPU(val):
                    if err:
                        err += '\n'
                    err += 'Значение параметра\nNW_SCORE_MINLEN_FOR_GPU\nне входит в допустимый диапазон {}\nи установлено по умолчанию: {}'.format(
                        get_options().NW_SCORE_MINLEN_FOR_GPU_MINMAX_REPR,
                        get_options().NW_SCORE_MINLEN_FOR_GPU)
            except:
                if err:
                    err += '\n'
                err += 'Значение параметра\nNW_SCORE_MINLEN_FOR_GPU\nне является целым числом'
    
    if 'NW_ALIGN_MINLEN_FOR_GPU' in conf:
            val = conf['NW_ALIGN_MINLEN_FOR_GPU']
            try:
                val = int(val)
                if not try_set_NW_ALIGN_MINLEN_FOR_GPU(val):
                    if err:
                        err += '\n'
                    err += 'Значение параметра\nNW_ALIGN_MINLEN_FOR_GPU\nне входит в допустимый диапазон {}\nи установлено по умолчанию: {}'.format(
                        get_options().NW_ALIGN_MINLEN_FOR_GPU_MINMAX_REPR,
                        get_options().NW_ALIGN_MINLEN_FOR_GPU)
            except:
                if err:
                    err += '\n'
                err += 'Значение параметра\nNW_ALIGN_MINLEN_FOR_GPU\nне является целым числом'
    return err
    