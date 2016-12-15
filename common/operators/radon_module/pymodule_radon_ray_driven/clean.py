import os
import shutil


if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.isfile( 'radon_ray_driven.c' ) is True:
    os.remove( 'radon_ray_driven.c' ) 

if os.path.isfile( 'radon_ray_driven.so' ) is True:
    os.remove( 'radon_ray_driven.so' )
