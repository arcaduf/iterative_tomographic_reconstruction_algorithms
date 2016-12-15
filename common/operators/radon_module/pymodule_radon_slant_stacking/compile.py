import os
import shutil


if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.isfile( 'radon_slant_stacking.c' ) is True:
    os.remove( 'radon_slant_stacking.c' ) 

os.system('python create_module.py build_ext --inplace')
