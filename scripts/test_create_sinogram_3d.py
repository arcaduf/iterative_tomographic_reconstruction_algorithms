##  In this test a 3D Shepp-Logan phantom, consisting of z-slices saved
##  in separate files, is forward projected with the forward gridding
##  projector with minimal oversmapling.
##  The outcome is a new folder containing sinograms with 304 views for
##  each z-slice.

from __future__ import division , print_function
import os

os.chdir( '../algorithms' )
command = 'python -W ignore gridding_forward.py -Di ../data/sl3d/sl3d_256_phantom -Do ../data/sl3d/sl3d_256_sino_ang0304 -n 304'
print( '\n\nCommand line:\n', command )
os.system( command )
