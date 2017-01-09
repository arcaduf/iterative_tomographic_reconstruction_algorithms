##  In this test a sinogram of Shepp-Logan is reconstructed with the ADMM
##  Plug-and-Play, using the SPlit-Bregman non-local TV method as regularization 
##  (-r pp-nltv), the projectors based on a cubic B-spline tensor product
##  (-pr bspline), 4 iterations (-n1 3), 4 CG-subiterations (-n2 4),
##  physical constraints, i.e., setting to zero all negative pixels
##  at each iteration (-pc 0.0).

from __future__ import division , print_function
import os

os.chdir( '../algorithms' )
command = 'python -W ignore admm.py -Di ../data/ -i shepp_logan_pix0256_ang0304_sino.tif  -o shepp_logan_pix0256_ang0304_sino_reco.tif -pr bspline -r pp-nltv -n1 3 -n2 4 -pp 1.0:1.0:1.0:1.0 -pc 0.0'
print( '\n\nCommand line:\n', command )
os.system( command )
