##  In this test a sinogram of Shepp-Logan with 256 pixels X 304 views is 
##  reconstructed with the adjoint gridding projector with minimal oversampling
##  using a parzen window superimposed to the ramp filter.

from __future__ import division , print_function
import os

os.chdir( '../algorithms' )
command = 'python -W ignore gridding_adjoint.py -Di ../data/ -i shepp_logan_pix0256_ang0304_sino.tif -f parzen -p'
print( '\n\nCommand line:\n', command )
os.system( command )
