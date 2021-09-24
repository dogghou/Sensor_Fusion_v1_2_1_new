# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 20:53:57 2020

@author: Administrator
"""

import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules=cythonize(Extension(
           'lidar_movement',
           language='c',
           sources = ['lidar_movement.pyx'],
           include_dirs=[np.get_include()],
           library_dirs=[],
           libraries=[],
           extra_compile_args=[],
           extra_link_args=[])))
