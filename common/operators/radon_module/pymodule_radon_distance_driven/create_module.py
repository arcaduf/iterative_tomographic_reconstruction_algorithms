from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("radon_distance_driven",
                             sources=[ "radon_distance_driven.pyx" ,
                                       "radon_dd.c" 
                                       ],
                             include_dirs=[numpy.get_include()],libraries=['gcov'],
                             extra_compile_args=['-w','-O3','-march=native','-ffast-math','-fprofile-generate','-fopenmp'],
                             extra_link_args=['-fprofile-generate'])],
)
