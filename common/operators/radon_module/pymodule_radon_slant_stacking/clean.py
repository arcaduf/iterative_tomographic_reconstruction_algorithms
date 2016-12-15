import os
import shutil


if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.isfile( 'radon_slant_stacking.c' ) is True:
    os.remove( 'radon_slant_stacking.c' ) 

if os.path.isfile( 'radon_slant_stacking.so' ) is True:
    os.remove( 'radon_slant_stacking.so' )
