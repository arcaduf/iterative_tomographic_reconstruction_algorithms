import os
import shutil


if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.isfile( 'genradon.c' ) is True:
    os.remove( 'genradon.c' )

if os.path.isfile( 'genradon.so' ) is True:
    os.remove( 'genradon.so' )   

os.system('python create_module.py build_ext --inplace')
