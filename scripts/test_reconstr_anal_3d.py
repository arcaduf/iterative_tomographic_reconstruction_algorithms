##  In this test the forward projection of a 3D Shepp-Logan with 304 views
##  is reconstructed with the adjoint gridding projector with minimal 
##  oversampling using a hanning window superimposed to the ramp filter.
##  The sinograms correspond to separate files in ../data/sl3d/sl3d_256_sino_ang0304.
##  The reconstructions correspond to separate files as well, saved in the newly
##  created folder ../data/sl3d/sl3d_256_reco_ang0304.

from __future__ import division , print_function
import os

os.chdir( '../algorithms' )
command = 'python -W ignore gridding_adjoint.py -Di ../data/sl3d/sl3d_256_sino_ang0304 -Do ../data/sl3d/sl3d_256_reco_ang0304 -f hanning'
print( '\n\nCommand line:\n', command )
os.system( command )
