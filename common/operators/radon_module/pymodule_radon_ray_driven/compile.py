import os
import shutil


if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.isfile( 'radon_ray_driven.c' ) is True:
    os.remove( 'radon_ray_driven.c' ) 

os.system('python create_module.py build_ext --inplace')
