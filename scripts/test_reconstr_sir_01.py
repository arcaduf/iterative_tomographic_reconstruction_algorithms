##  In this test a sinogram of Shepp-Logan is reconstructed with the
##  Maximum-Likelihood Expectation-Maximization (-a em) the gridding
##  projectors with minimal oversampling (-pr grid-kb), 100 iterations
##  (-n 100).

from __future__ import division , print_function
import os

os.chdir( '../algorithms' )
command = 'python -W ignore sir.py -a em -Di ../data/ -i shepp_logan_pix0256_ang0304_sino.DMP  -o shepp_logan_pix0256_ang0304_sino_reco.DMP -pr grid-pswf -n 12 -p'
print( '\n\nCommand line:\n', command )
os.system( command )
