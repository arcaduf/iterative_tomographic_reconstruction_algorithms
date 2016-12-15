import os
import shutil


if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.isfile( 'radon_distance_driven.c' ) is True:
    os.remove( 'radon_distance_driven.c' ) 

if os.path.isfile( 'radon_distance_driven.so' ) is True:
    os.remove( 'radon_distance_driven.so' )
