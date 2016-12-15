##  In this test a sinogram of Shepp-Logan is reconstructed with the ADMM
##  LASSO-TV, using the TV shrinkage-thresholding method as regularization 
##  (-r lasso-tv), the projectors based on the distance-driven approach
##  (-pr dist-driv), 4 iterations (-n1 3), 4 CG-subiterations (-n2 4),
##  physical constraints, i.e., setting to zero all negative pixels
##  at each iteration (-pc 0.0), using as center of rotation 128.5 
##  instead of the default mid pixel, i.e. 128 (-c 128.5).

##  N.B.: the input sinogram in this example was created with gridding
##  projectors with minimal oversampling that are not centered as the
##  projectors based on the distance-driven approach. This is why the
##  test requires to specify the rotation center shifted by half pixel.
##  It is just a minor problem regarding the implementations of these
##  projectors.

from __future__ import division , print_function
import os

os.chdir( '../algorithms' )
command = 'python -W ignore admm.py -Di ../data/ -i shepp_logan_pix0256_ang0304_sino.DMP  -o shepp_logan_pix0256_ang0304_sino_reco.DMP -pr dist-driv -r lasso-tv -n1 3 -n2 4 -pp 1.0:1.0:1.0:1.0 -pc 0.0 -c 128.5 -p'
print( '\n\nCommand line:\n', command )
os.system( command )
