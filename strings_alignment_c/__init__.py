class Options:
    @property
    def NW_ALIGN_MINLEN_FOR_GPU(self):
        return self._NW_ALIGN_MINLEN_FOR_GPU
    
    @property
    def NW_ALIGN_MINLEN_FOR_GPU_MINMAX_REPR(self):
        return '[1000; +Inf)'

    @NW_ALIGN_MINLEN_FOR_GPU.setter
    def NW_ALIGN_MINLEN_FOR_GPU(self, value):
        self._NW_ALIGN_MINLEN_FOR_GPU = max(1000, value)

    @property
    def NW_SCORE_MINLEN_FOR_GPU(self):
        return self._NW_SCORE_MINLEN_FOR_GPU

    @property
    def NW_SCORE_MINLEN_FOR_GPU_MINMAX_REPR(self):
        return '[10000; +Inf)'

    @NW_SCORE_MINLEN_FOR_GPU.setter
    def NW_SCORE_MINLEN_FOR_GPU(self, value):
        self._NW_SCORE_MINLEN_FOR_GPU = max(10000, value)

    @property
    def HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE(self):
        return self._HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE

    @property
    def HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE_MINMAX_REPR(self):
        return '[2000; 25000]'

    @HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE.setter
    def HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE(self, value):
        self._HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE = min(max(2000, value), 25000)
    
    _options = None

    def __init__(self) -> None:
        # Минимальная длина минимального массива для проведения выравнивания по алгоритму NeedlemanWunsch на GPU
        self._NW_ALIGN_MINLEN_FOR_GPU = 10000

        # Минимальная длина минимального массива для проведения расчета NWScore на GPU
        self._NW_SCORE_MINLEN_FOR_GPU = 50000

        # Минимальное количество len(a) * len(b) чтобы рекурсивное выравнивание переключилось на алгоритм NeedlemanWunsch
        self._HIRSCHBERG_MAX_SIZE_FOR_NEEDLEMAN_WUNSCH_USE = 10000


def get_options() -> Options:
    if Options._options is not None:
        return Options._options
    Options._options = Options()
    return Options._options