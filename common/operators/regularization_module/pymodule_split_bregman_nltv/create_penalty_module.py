from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("split_bregman_nltv",
                             sources=[ "split_bregman_nltv.pyx" ,
                                       "nl_weights.c" ,
                                       "sbnltv.c" ,
                                       ],
                             include_dirs=[numpy.get_include()],libraries=['gcov'],extra_compile_args=['-w','-O3','-march=native','-ffast-math','-fprofile-generate'],extra_link_args=['-fprofile-generate'])],
)
