#!/usr/bin/python
import sys
import numpy as np
sys.path.insert(0, "...")

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_module = Extension(
"core_process",
["core_process.pyx"],#要转换的pyx文件

)

setup(
cmdclass={'build_ext': build_ext},
ext_modules=[ext_module],
include_dirs=[np.get_include()]
)
