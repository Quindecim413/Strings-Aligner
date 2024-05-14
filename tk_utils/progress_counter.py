class ProgressCounter:
    def __init__(self, start_val, max_val):
        self._val = start_val
        self._max_val = max_val

    def count_up(self):
        self._val += 1
    
    @property
    def processed(self):
        return self._val
    
    @property
    def max_val(self):
        return self._max_val

    @property
    def progress(self):
        return self._val / self._max_val