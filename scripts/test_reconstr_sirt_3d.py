##  In this test a sinogram of Shepp-Logan is reconstructed with the ADMM
##  Plug-and-Play, using the Split-Bregman TV method as regularization 
##  (-r pp-breg), the gridding projectors with minimal oversampling
##  (-pr grid-kb), 4 iterations (-n1 4), 4 CG-subiterations (-n2 4),
##  physical constraints, i.e., setting to zero all negative pixels
##  at each iteration (-pc 0.0).

from __future__ import division , print_function
import os

os.chdir( '../algorithms' )
command = 'python -W ignore art.py -a sirt -Di ../data/sl3d/sl3d_128_sino_ang0200/ -Do ../data/sl3d/sl3d_128_reco_ang0200/ -pr grid-kb -n50 -pc 0.0'
print( '\n\nCommand line:\n', command )
os.system( command )
