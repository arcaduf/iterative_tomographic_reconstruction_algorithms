##  In this test a Shepp-Logan phantom is forward projected with the
##  forward gridding projector with minimal oversampling for 304
##  different angles in [0,180)

from __future__ import division , print_function
import os

os.chdir( '../algorithms' )
command = 'python -W ignore gridding_forward.py -Di ../data/ -i shepp_logan_pix0256.DMP -n 304 -p'
print( '\n\nCommand line:\n', command )
os.system( command )
