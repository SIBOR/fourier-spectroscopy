# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 06:47:34 2017

@author: jaymz
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("fourierTransC.pyx"),
    include_dirs=[np.get_include()]
)