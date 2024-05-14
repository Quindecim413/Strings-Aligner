import sys, os
from cx_Freeze import setup, Executable


EXECUTABLE_PATH = sys.base_exec_prefix

tcls = [os.path.join(EXECUTABLE_PATH, 'DLLS', el) for el in ['tcl86t.dll', 'tk86t.dll']]
tcls.extend([os.path.join(EXECUTABLE_PATH, 'tcl', el) for el in ['tcl8.6/', 'tk8.6/']])

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {"includes": ["tkinter"],
                    'packages': ['tkinter', 'numpy', 'pandas', 'psutil', 'scipy', 'ctypes'],
                    'include_files': ['OpenCLExtension.dll', *tcls]}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name = "Strings aligner",
    version = "1.0",
    description = "Strings aligning app from lab of medical cybernetics for biological researchers",
    options = {"build_exe": build_exe_options},
    executables = [Executable("tk_app.py", base = base)])