from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("gridrec_v4",
                             sources=[ "gridrec_v4.pyx" ,
                                       "gridrec_v4_backproj.c" ,
                                       "gridrec_v4_forwproj.c" ,
                                       "filters.c" ,
                                       "fft.c"
                                       ],
                             include_dirs=[numpy.get_include()],libraries=['fftw3f','gcov'],extra_compile_args=['-w','-O3','-march=native','-ffast-math','-fprofile-generate'],extra_link_args=['-fprofile-generate'])],
)


'''
import gridrec_v4
gridrec_v4.createFFTWWisdomFile(2016, "profile.wis")

import os
import sys
os.system(sys.executable + " profile.py")

os.remove('gridrec_v4.so')

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("gridrec_v4",
                             sources=[ "gridrec_v4.pyx" ,
                                       "gridrec_v4_backproj.c" ,
                                       "gridrec_v4_forwproj.c" ,
                                       "filters.c" , 
                                       "fft.c"
                                       ],
                             include_dirs=[numpy.get_include()],libraries=['fftw3f'],extra_compile_args=['-O3','-march=native','-ffast-math','-fprofile-use'],extra_link_args=['-fprofile-use'])],
)
'''
