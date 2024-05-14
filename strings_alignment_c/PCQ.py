from .c_binding import *

class PCQ:
    def __init__(self, platform_ind=0):
        self.platform_ind = platform_ind
        self.c_pcq = None
    
    def build(self):
        # assert self.c_pcq is not None, 'PCQ should be initialized'
        
        self.c_pcq = c_createPCQ(self.platform_ind)
        err_code = c_buildPCQ(self.c_pcq)
        if err_code:
            raise Exception('PCQ.build err code: ' + str(err_code))
        assert self.c_pcq != 0, "Failed to initialize PCQ"
        return self
    
    def free(self):
        c_deletePCQ(self.c_pcq)

    def __enter__(self):
        if self.c_pcq is None:
            self.build()
        return self

    def __exit__(self, *args):
        self.free()

def get_platform_names():
    names_prt = c_getPlatformNames()
    names_str = ctypes.c_char_p(names_prt).value.decode("utf-8") 
    names = names_str.split('\n')
    return names