import sys
import numpy as np

A = sys.path.insert(0, "..")
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_module = Extension(
    "core_process",
    ["core_process.pyx"],
    extra_compile_args=["/openmp"],
    extra_link_args=["/openmp"],
)

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
    # 注意这一句一定要有，不然只编译成C代码，无法编译成pyd文件
    include_dirs=[np.get_include()]
)
