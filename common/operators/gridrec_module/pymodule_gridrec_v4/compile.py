import os
import shutil


if os.path.exists('build'):
    shutil.rmtree('build')

if os.path.isfile( 'gridrec_v4.c' ) is True:
    os.remove( 'gridrec_v4.c' )

if os.path.isfile( 'gridrec_v4.so' ) is True:
    os.remove( 'gridrec_v4.so' )   

os.system('python create_module.py build_ext --inplace')
